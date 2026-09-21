"""The REST surface.

Handlers are plain ``def``, not ``async def``, and deliberately so — the same
rule ``syncai_backend`` follows and for the same reason. Synthesis is CPU-bound
for a few hundred milliseconds (plus a one-time ~310 MB model load on the first
request), and ``?wait=true`` blocks for the length of an utterance. Declared
``def``, FastAPI runs them in its worker threadpool; declared ``async def`` they
would sit on the event loop and stall every other request for the duration.

Routes:

======  ============================  ====================================
GET     /api/v1/voices                the loaded model's voice ids
POST    /api/v1/synthesize            text -> WAV bytes, nothing is played
POST    /api/v1/speak                 queue an utterance; 202 + a job id
GET     /api/v1/speak                 recent jobs, newest first
GET     /api/v1/speak/{id}            one job's state
DELETE  /api/v1/speak/{id}            cancel it, queued or playing
======  ============================  ====================================
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, Query, Response, status
from pydantic import BaseModel, Field

from syncai_tts.config import MAX_SPEED, MAX_TEXT_LENGTH, MIN_SPEED, Settings
from syncai_tts.engine import KokoroEngine
from syncai_tts.player import JobView, SpeechPlayer

# How long past the utterance's own length a `wait=true` caller is held before
# being told to poll instead. Generous, because the wait also covers time spent
# queued behind other utterances; a caller that does not want to wait at all
# simply omits the flag.
_DEFAULT_WAIT_SLACK_S = 30.0


class SynthesizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The text to speak. English only (kokoro's G2P is English).",
    )
    voice: Optional[str] = Field(
        None,
        description=(
            "Kokoro voice id; the list is at GET /api/v1/voices. "
            "Defaults to TTS_DEFAULT_VOICE."
        ),
    )
    speed: float = Field(
        1.0, ge=MIN_SPEED, le=MAX_SPEED, description="Playback rate multiplier."
    )


class SpeakRequest(SynthesizeRequest):
    wait: bool = Field(
        False,
        description=(
            "Hold the response until playback finishes. Off by default: the "
            "intended caller polls GET /api/v1/speak/{id} instead, which is what "
            "lets a Temporal SPEAK activity heartbeat and be cancelled."
        ),
    )
    wait_timeout: Optional[float] = Field(
        None,
        ge=0.0,
        le=600.0,
        description=(
            "Seconds to hold a wait=true response before answering with the "
            "job's current state. Defaults to the utterance's length plus slack."
        ),
    )


class JobResponse(BaseModel):
    id: str = Field(..., description="Opaque job id; poll or cancel with it.")
    status: str = Field(
        ..., description="queued | playing | done | failed | cancelled."
    )
    text: str = Field(..., description="The text this job speaks.")
    voice: str = Field(..., description="The voice it was rendered with.")
    speed: float = Field(..., description="The rate it was rendered at.")
    duration: float = Field(..., description="Length of the rendered audio, in seconds.")
    queue_position: Optional[int] = Field(
        None, description="0 is next up. Null unless the job is still queued."
    )
    queued_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = Field(
        None, description="Why it failed or was cancelled; null otherwise."
    )
    code: Optional[str] = Field(
        None,
        description=(
            "Stable discriminator for `error`, e.g. playback_failed. Read this "
            "rather than matching on the sentence."
        ),
    )

    @classmethod
    def of(cls, view: JobView) -> "JobResponse":
        return cls(
            id=view.id,
            status=view.status.value,
            text=view.text,
            voice=view.voice,
            speed=view.speed,
            duration=view.duration_s,
            queue_position=view.queue_position,
            queued_at=view.queued_at,
            started_at=view.started_at,
            finished_at=view.finished_at,
            error=view.error,
            code=view.code.value if view.code else None,
        )


class ListVoicesResponse(BaseModel):
    voices: List[str] = Field(..., description="The voice ids the loaded model carries.")


class ListJobsResponse(BaseModel):
    jobs: List[JobResponse] = Field(..., description="Newest first.")


def init_routes(
    logger: structlog.stdlib.BoundLogger,
    settings: Settings,
    engine: KokoroEngine,
    player: SpeechPlayer,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["TTS"])

    @router.get("/voices", response_model=ListVoicesResponse)
    def list_voices():
        return ListVoicesResponse(voices=engine.voices())

    # No response_model: the body is the WAV itself, so the schema is declared
    # through `responses` instead.
    @router.post(
        "/synthesize",
        responses={200: {"content": {"audio/wav": {}}, "description": "The rendered WAV."}},
        response_class=Response,
    )
    def synthesize(request: SynthesizeRequest):
        """Render text and hand back the audio. Nothing reaches the speaker."""
        utterance = engine.synthesize(
            text=request.text,
            voice=request.voice or settings.default_voice,
            speed=request.speed,
        )
        return Response(
            content=utterance.wav,
            media_type="audio/wav",
            headers={
                # So a caller can size a timeout without decoding the WAV.
                "X-Audio-Duration": f"{utterance.duration_s:.3f}",
            },
        )

    @router.post(
        "/speak",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def speak(request: SpeakRequest, response: Response):
        """Render text and queue it for the speaker.

        Synthesis happens inline, so the two failures the caller can act on — an
        unknown voice (400) and missing weights (503) — are answered
        synchronously rather than discovered by polling. Only playback, the part
        that takes as long as the utterance, becomes a job.
        """
        voice = request.voice or settings.default_voice
        utterance = engine.synthesize(text=request.text, voice=voice, speed=request.speed)
        view = player.enqueue(
            utterance=utterance, text=request.text, voice=voice, speed=request.speed
        )

        if not request.wait:
            return JobResponse.of(view)

        timeout = request.wait_timeout
        if timeout is None:
            timeout = (
                utterance.duration_s
                + settings.playback_timeout_margin
                + _DEFAULT_WAIT_SLACK_S
            )
        view = player.wait(view.id, timeout=timeout)
        if view.status.terminal:
            # It really is finished, so this is no longer "accepted for later".
            response.status_code = status.HTTP_200_OK
        return JobResponse.of(view)

    @router.get("/speak", response_model=ListJobsResponse)
    def list_jobs(
        limit: int = Query(20, ge=1, le=200, description="How many jobs to return."),
    ):
        return ListJobsResponse(jobs=[JobResponse.of(v) for v in player.recent(limit)])

    @router.get("/speak/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return JobResponse.of(player.get(job_id))

    @router.delete("/speak/{job_id}", response_model=JobResponse)
    def cancel_job(job_id: str):
        """Stop a job, whether it is waiting or already on the speaker.

        Cancelling an already-finished job answers 200 with its terminal state
        rather than an error: a caller racing its own cancel against the end of
        an utterance should not have to tell those two outcomes apart.
        """
        return JobResponse.of(player.cancel(job_id))

    return router
