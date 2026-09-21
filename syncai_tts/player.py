"""The speaker: one utterance at a time, queued, cancellable.

This is the object the whole split exists for. While ``TtsGateway`` lived inside
``syncai_backend``, "one utterance on the device at a time" was enforced by a
``threading.Lock`` held across the ``aplay`` subprocess — which works only as
long as every caller is in that one process. The moment the Temporal worker
moves to a process of its own, two ``TtsGateway`` instances hold two locks and
nothing stops a scheduled SPEAK step and a manual ``POST /tts/speak`` from
opening two streams on the same pcm node. Making the speaker's owner a service
puts the lock back in front of the single device.

Two things changed while moving it here, both of which the in-process version
could not have:

**Playback is a job, not a blocking call.** ``POST /api/v1/speak`` answers 202
with an id and the audio plays behind it. The backend's SPEAK activity can then
poll that id once a second, which is a heartbeat — today the activity holds one
blocking ``speak()`` call, cannot heartbeat at all, and so runs on
``start_to_close`` alone and is uncancellable mid-utterance. A cancel now has
somewhere to land: ``DELETE /api/v1/speak/{id}``.

**The queue is FIFO and bounded.** A lock wakes waiters in whatever order the OS
chooses, so three queued utterances could play in any order; a deque plays them
in the order they were accepted, and refuses the ninth instead of accepting work
it will speak minutes later.
"""

import os
import re
import subprocess
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, List, Optional

import structlog

from syncai_tts.config import Settings
from syncai_tts.engine import Utterance
from syncai_tts.errors import ConflictError, Failure, NotFoundError


# How long to wait for aplay to die after a cancel before escalating to SIGKILL.
# aplay handles SIGTERM promptly; this is the "it did not" budget, kept short
# because the speaker is blocked for its duration.
_TERMINATE_GRACE_S = 2.0


class JobStatus(str, Enum):
    QUEUED = "queued"
    PLAYING = "playing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass(frozen=True)
class JobView:
    """An immutable snapshot of a job, taken under the player's lock.

    The API renders these. Handing out the live job object instead would let a
    response be serialised while the playback thread mutates it half way.
    """

    id: str
    status: JobStatus
    text: str
    voice: str
    speed: float
    duration_s: float
    queue_position: Optional[int]
    queued_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error: Optional[str]
    code: Optional[Failure]


class _Job:
    """Mutable playback job. Every field is written under the player's lock."""

    __slots__ = (
        "id",
        "text",
        "voice",
        "speed",
        "duration_s",
        "status",
        "queued_at",
        "started_at",
        "finished_at",
        "error",
        "code",
        "wav",
        "cancel_requested",
        "done",
    )

    def __init__(self, utterance: Utterance, text: str, voice: str, speed: float):
        self.id = uuid.uuid4().hex
        self.text = text
        self.voice = voice
        self.speed = speed
        self.duration_s = utterance.duration_s
        self.status = JobStatus.QUEUED
        self.queued_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.code: Optional[Failure] = None
        # Dropped the moment the job reaches a terminal state: a minute of
        # 24 kHz mono is ~2.8 MB, and the history keeps 64 records.
        self.wav: Optional[bytes] = utterance.wav
        self.cancel_requested = False
        self.done = threading.Event()


def resolve_playback_device(pcm_link: str, fallback: str) -> str:
    """ALSA name of the speaker, via the udev symlink when it exists.

    ``/dev/syncai/speaker_pcm -> ../snd/pcmC2D0p`` encodes the live card and
    device number in its target's name; realpath follows it and the regex lifts
    the numbers out. Any miss — link absent (rules not installed, speaker
    unplugged) or a target that is not a pcm node — falls back to the by-name
    device rather than raising: aplay itself is the error reporter with an
    actionable message, and a broken link should not turn that into a traceback.

    Resolved per utterance rather than once at startup, because a replug moves
    the card number and the link with it, and a name resolved at construction
    would go stale.
    """
    target = os.path.realpath(pcm_link)
    match = re.fullmatch(r"pcmC(\d+)D(\d+)p", os.path.basename(target))
    if match is None:
        return fallback
    return f"plughw:{match.group(1)},{match.group(2)}"


class SpeechPlayer:
    def __init__(self, logger: structlog.stdlib.BoundLogger, settings: Settings):
        self._logger = logger
        self._settings = settings

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)

        self._pending: Deque[_Job] = deque()
        self._current: Optional[_Job] = None
        self._proc: Optional[subprocess.Popen] = None
        self._jobs: "OrderedDict[str, _Job]" = OrderedDict()
        self._stopping = False
        self._thread: Optional[threading.Thread] = None

    # --- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        # Cleared so a stop/start pair is a real restart rather than a thread
        # that wakes, sees the old stop flag and returns. Safe because stop()
        # joins before it returns, so no previous worker is still running here.
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="speech-player", daemon=True
        )
        self._thread.start()
        self._logger.info("[SpeechPlayer] Playback thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop playing, drop the queue, and join the thread.

        Called from the app's lifespan shutdown, which uvicorn runs on SIGTERM —
        so a container stop cuts the utterance instead of leaving an orphaned
        aplay holding the pcm node open. Queued utterances are marked cancelled
        rather than spoken: nothing is listening to a robot that is shutting
        down, and draining them would delay the stop by their total length.
        """
        with self._wake:
            if self._stopping:
                return
            self._stopping = True
            while self._pending:
                job = self._pending.popleft()
                self._finish_locked(job, JobStatus.CANCELLED, "service is shutting down")
            # Marked as well as signalled: the worker may be between popping a
            # job and spawning aplay, where there is no process to terminate
            # yet. The flag is what it checks before spawning, so the utterance
            # is cut in that window too instead of starting after the stop.
            if self._current is not None:
                self._current.cancel_requested = True
            proc = self._proc
            self._wake.notify_all()

        if proc is not None:
            self._terminate(proc)

        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._logger.warning("[SpeechPlayer] Playback thread did not stop in time")
        self._thread = None
        self._logger.info("[SpeechPlayer] Playback thread stopped")

    # --- Queue ---------------------------------------------------------------

    def enqueue(self, utterance: Utterance, text: str, voice: str, speed: float) -> JobView:
        """Accept an utterance for playback. Returns the job as it stands now."""
        job = _Job(utterance=utterance, text=text, voice=voice, speed=speed)

        with self._wake:
            if self._stopping:
                raise ConflictError(
                    "The speech service is shutting down and is not accepting "
                    "new utterances.",
                    code=Failure.QUEUE_FULL,
                )
            if len(self._pending) >= self._settings.max_queue:
                raise ConflictError(
                    f"{len(self._pending)} utterances are already waiting "
                    f"(TTS_MAX_QUEUE={self._settings.max_queue}). Wait for the "
                    "queue to drain, or cancel one.",
                    code=Failure.QUEUE_FULL,
                )

            self._jobs[job.id] = job
            self._pending.append(job)
            self._evict_locked()
            self._wake.notify()
            view = self._view_locked(job)

        self._logger.info(
            "[SpeechPlayer] Utterance queued",
            job_id=job.id,
            voice=voice,
            duration_s=round(job.duration_s, 2),
            queue_position=view.queue_position,
        )
        return view

    def get(self, job_id: str) -> JobView:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(
                    f"No speech job {job_id!r}. It never existed, or it finished "
                    f"more than {self._settings.job_history} jobs ago.",
                    code=Failure.JOB_NOT_FOUND,
                )
            return self._view_locked(job)

    def recent(self, limit: int = 20) -> List[JobView]:
        """Newest first. Queued and playing jobs are in here too."""
        with self._lock:
            jobs = list(self._jobs.values())[-limit:]
            return [self._view_locked(job) for job in reversed(jobs)]

    def cancel(self, job_id: str) -> JobView:
        """Stop a job, whether it is waiting or already on the speaker.

        Cancelling a terminal job is not an error — it is the answer the caller
        wanted, and a SPEAK activity racing its own cancel against the end of
        the utterance should not have to tell the two apart.
        """
        with self._wake:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(
                    f"No speech job {job_id!r}.", code=Failure.JOB_NOT_FOUND
                )

            if job.status.terminal:
                return self._view_locked(job)

            job.cancel_requested = True

            if job.status is JobStatus.QUEUED:
                # Not on the device yet: drop it out of the queue and finish it
                # here. The worker skips ids it no longer finds pending.
                try:
                    self._pending.remove(job)
                except ValueError:
                    pass
                self._finish_locked(job, JobStatus.CANCELLED, "cancelled before playback")
                self._logger.info("[SpeechPlayer] Queued utterance cancelled", job_id=job.id)
                return self._view_locked(job)

            # Playing. _proc may still be None if the worker has popped the job
            # but not spawned aplay yet; the flag above is what it checks before
            # spawning, so the cancel is not lost either way.
            proc = self._proc
            view = self._view_locked(job)

        if proc is not None:
            self._terminate(proc)
            self._logger.info("[SpeechPlayer] Playing utterance cancelled", job_id=job_id)
        return view

    def wait(self, job_id: str, timeout: float) -> JobView:
        """Block until the job is terminal, or the timeout expires.

        Returns whatever state it is in either way — a timeout here is not an
        error, it is "still going, ask again". Only ``?wait=true`` callers use
        this; the polling path never blocks a request.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(
                    f"No speech job {job_id!r}.", code=Failure.JOB_NOT_FOUND
                )
            event = job.done

        event.wait(timeout=timeout)
        return self.get(job_id)

    def snapshot(self) -> dict:
        """What /health says about the speaker."""
        with self._lock:
            current = self._current
            return {
                "state": current.status.value if current else "idle",
                "job_id": current.id if current else None,
                "queue_depth": len(self._pending),
                "max_queue": self._settings.max_queue,
                "thread_alive": bool(self._thread and self._thread.is_alive()),
            }

    # --- Worker --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._pending and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                job = self._pending.popleft()
                if job.status.terminal:
                    # Cancelled between being queued and being picked up.
                    continue
                job.status = JobStatus.PLAYING
                job.started_at = datetime.now(timezone.utc)
                self._current = job

            try:
                self._play(job)
            except Exception as exc:
                # A playback thread that dies takes the speaker with it and says
                # nothing: every later utterance would queue forever while the
                # API kept answering 202. Finish the job that broke, log it, and
                # stay in the loop.
                self._logger.error(
                    "[SpeechPlayer] Playback raised", job_id=job.id, exc_info=True
                )
                with self._lock:
                    self._proc = None
                    self._current = None
                    if not job.status.terminal:
                        self._finish_locked(
                            job,
                            JobStatus.FAILED,
                            f"playback thread error: {exc}",
                            code=Failure.PLAYBACK_FAILED,
                        )

    def _play(self, job: _Job) -> None:
        device = resolve_playback_device(
            self._settings.speaker_pcm_link, self._settings.fallback_device
        )

        with self._lock:
            if job.cancel_requested:
                self._finish_locked(job, JobStatus.CANCELLED, "cancelled before playback")
                self._current = None
                return
            wav = job.wav or b""

        try:
            # aplay reads the WAV, header included, from stdin. -q so its
            # per-file banner does not land in the log on every utterance.
            proc = subprocess.Popen(
                ["aplay", "-q", "-D", device, "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            with self._lock:
                self._finish_locked(
                    job,
                    JobStatus.FAILED,
                    "aplay not found — alsa-utils is not installed in this image",
                    code=Failure.PLAYER_UNAVAILABLE,
                )
                self._current = None
            self._logger.error("[SpeechPlayer] aplay is not installed", job_id=job.id)
            return

        with self._lock:
            self._proc = proc
            cancel_now = job.cancel_requested
        if cancel_now:
            self._terminate(proc)

        # The budget starts here, not at enqueue, so a queued utterance is never
        # charged for the one it waited on.
        budget = job.duration_s + self._settings.playback_timeout_margin
        timed_out = False
        try:
            _, stderr = proc.communicate(input=wav, timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            _, stderr = proc.communicate()

        with self._lock:
            self._proc = None
            self._current = None

            if job.cancel_requested:
                self._finish_locked(job, JobStatus.CANCELLED, "cancelled during playback")
            elif timed_out:
                self._finish_locked(
                    job,
                    JobStatus.FAILED,
                    f"aplay timed out after {budget:.1f}s (device wedged?)",
                    code=Failure.PLAYBACK_TIMEOUT,
                )
            elif proc.returncode != 0:
                detail = (stderr or b"").decode(errors="replace").strip()
                self._finish_locked(
                    job,
                    JobStatus.FAILED,
                    f"aplay failed: {detail}" if detail else "aplay failed",
                    code=Failure.PLAYBACK_FAILED,
                )
            else:
                self._finish_locked(job, JobStatus.DONE)

            status = job.status
            error = job.error

        if status is JobStatus.DONE:
            self._logger.info(
                "[SpeechPlayer] Utterance spoken",
                job_id=job.id,
                duration_s=round(job.duration_s, 2),
                device=device,
            )
        elif status is JobStatus.FAILED:
            self._logger.error(
                "[SpeechPlayer] Playback failed",
                job_id=job.id,
                device=device,
                error=error,
            )

    def _terminate(self, proc: subprocess.Popen) -> None:
        """SIGTERM, then SIGKILL if aplay is still holding the device."""
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            self._logger.warning("[SpeechPlayer] aplay ignored SIGTERM; killing")
            proc.kill()
        except (ProcessLookupError, OSError):
            # Exited between poll() and terminate(); nothing to stop.
            pass

    # --- Internals (caller holds the lock) -----------------------------------

    def _finish_locked(
        self,
        job: _Job,
        status: JobStatus,
        error: Optional[str] = None,
        code: Optional[Failure] = None,
    ) -> None:
        job.status = status
        job.finished_at = datetime.now(timezone.utc)
        job.error = error
        job.code = code
        job.wav = None  # the record is kept; the audio is not
        job.done.set()
        # Deliberately no eviction here. Trimming on finish means that with a
        # small history the job that just finished can be the one evicted, and
        # the caller polling the id it was handed gets a 404 for work that
        # succeeded. Room is made when new work arrives instead; see
        # _evict_locked.

    def _evict_locked(self) -> None:
        """Trim finished jobs down to the history cap, oldest first.

        Called when a job is accepted, not when one ends, so a job that has just
        finished is always still readable by whoever is polling it.

        Only terminal jobs are evictable: a queued or playing job must stay
        readable however long the queue gets, which means `_jobs` can sit above
        the cap while a backlog drains. That is the intended trade — the cap
        bounds history, not work in flight.
        """
        if len(self._jobs) <= self._settings.job_history:
            return
        for job_id, job in list(self._jobs.items()):
            if len(self._jobs) <= self._settings.job_history:
                break
            if job.status.terminal:
                del self._jobs[job_id]

    def _view_locked(self, job: _Job) -> JobView:
        position = None
        if job.status is JobStatus.QUEUED:
            try:
                position = self._pending.index(job)
            except ValueError:
                position = None
        return JobView(
            id=job.id,
            status=job.status,
            text=job.text,
            voice=job.voice,
            speed=job.speed,
            duration_s=job.duration_s,
            queue_position=position,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            code=job.code,
        )
