"""The speaker's rules: one at a time, in order, and stoppable.

This is the module the split exists for, so it carries the most tests. The
regression at the centre of them is the one syncai_backend hit in 79fa196: a
lock that covered synthesis but not playback let two callers into `aplay` at
once, which is two streams into one pcm node. Here the guarantee is structural
(a single playback thread), and these tests pin it so a future rewrite that
reintroduces a pool has to argue with them.
"""

import os
import threading

import pytest

from syncai_tts.engine import Utterance
from syncai_tts.errors import ConflictError, Failure, NotFoundError
from syncai_tts.player import JobStatus, SpeechPlayer, resolve_playback_device

from .conftest import make_settings, wait_until


def utterance(duration_s: float = 0.1) -> Utterance:
    return Utterance(wav=b"RIFF____fake wav", duration_s=duration_s, sample_rate=24000)


def speak(player, text: str = "hello", voice: str = "af_heart"):
    return player.enqueue(utterance(), text=text, voice=voice, speed=1.0)


def finished(player, job_id: str) -> bool:
    return player.get(job_id).status.terminal


# --- The core guarantee -----------------------------------------------------


def test_only_one_utterance_reaches_the_speaker_at_a_time(player, aplay):
    """Four callers, one speaker: never two aplay processes at once."""
    aplay.play_time = 0.05
    jobs = [speak(player, text=f"line {i}") for i in range(4)]

    assert wait_until(lambda: all(finished(player, job.id) for job in jobs))
    assert aplay.max_inside == 1
    assert len(aplay.calls) == 4
    assert [player.get(job.id).status for job in jobs] == [JobStatus.DONE] * 4


def test_utterances_play_in_the_order_they_were_accepted(player, aplay):
    """FIFO, which a lock could not promise: it wakes waiters in any order."""
    aplay.play_time = 0.05
    jobs = [speak(player, text=f"line {i}") for i in range(3)]

    assert wait_until(lambda: all(finished(player, job.id) for job in jobs))

    finish_times = [player.get(job.id).finished_at for job in jobs]
    assert finish_times == sorted(finish_times)


def test_queue_position_counts_only_what_is_still_waiting(player, aplay):
    """0 is next up.

    The utterance on the device has left the queue, so the first job behind it
    reports 0 rather than 1, and the one playing reports None. That is what a
    console showing "2 ahead of you" has to render.
    """
    aplay.play_time = 5.0
    playing = speak(player, text="playing")
    assert wait_until(lambda: player.get(playing.id).status is JobStatus.PLAYING)
    assert player.get(playing.id).queue_position is None

    first = speak(player, text="q0")
    second = speak(player, text="q1")

    assert first.queue_position == 0
    assert second.queue_position == 1


def test_a_finished_job_keeps_its_record_but_drops_its_audio(player, aplay):
    job = speak(player)
    assert wait_until(lambda: finished(player, job.id))

    view = player.get(job.id)
    assert view.status is JobStatus.DONE
    assert view.duration_s == pytest.approx(0.1)
    assert view.started_at is not None and view.finished_at is not None
    assert view.error is None and view.code is None
    # The audio itself is released; only the small record is retained.
    assert player._jobs[job.id].wav is None


# --- Cancellation -----------------------------------------------------------


def test_a_queued_utterance_can_be_cancelled_before_it_is_spoken(player, aplay):
    aplay.play_time = 1.0
    first = speak(player, text="first")
    second = speak(player, text="second")

    assert wait_until(lambda: player.get(first.id).status is JobStatus.PLAYING)
    view = player.cancel(second.id)

    assert view.status is JobStatus.CANCELLED
    assert wait_until(lambda: finished(player, first.id))
    # Only the first ever reached aplay.
    assert len(aplay.calls) == 1


def test_a_playing_utterance_is_stopped_at_the_device(player, aplay):
    aplay.play_time = 5.0
    job = speak(player)
    # Wait for audio to actually be going into aplay, not merely for the job to
    # be marked PLAYING: the worker marks it before it spawns the process, and
    # a cancel landing in between is a different path (covered below).
    assert wait_until(lambda: aplay.inside >= 1)

    player.cancel(job.id)

    assert wait_until(lambda: finished(player, job.id))
    assert player.get(job.id).status is JobStatus.CANCELLED
    assert aplay.terminated >= 1


def test_a_cancel_during_spawn_still_reaches_the_device(player, aplay):
    """The window between marking a job PLAYING and having a process to signal.

    The worker sets the status under the lock and only then spawns aplay, so a
    cancel can arrive while `_proc` is still None and find nothing to terminate.
    What saves it is the request flag, re-checked once the process exists. Held
    open deliberately here with a gate, because it is narrow enough that it only
    shows up as an intermittent failure otherwise.
    """
    gate = threading.Event()
    aplay.gate = gate
    aplay.play_time = 5.0

    job = speak(player)
    assert wait_until(lambda: len(aplay.calls) == 1)  # inside the spawn, blocked

    player.cancel(job.id)
    gate.set()

    assert wait_until(lambda: finished(player, job.id))
    assert player.get(job.id).status is JobStatus.CANCELLED
    assert aplay.terminated >= 1


def test_cancelling_a_finished_job_answers_its_terminal_state(player, aplay):
    """Not an error: a caller racing its own cancel against the end of an
    utterance should not have to tell the two outcomes apart."""
    job = speak(player)
    assert wait_until(lambda: finished(player, job.id))

    view = player.cancel(job.id)
    assert view.status is JobStatus.DONE


def test_cancelling_an_unknown_job_is_a_not_found(player):
    with pytest.raises(NotFoundError) as excinfo:
        player.cancel("nosuchjob")
    assert excinfo.value.code is Failure.JOB_NOT_FOUND


def test_a_cancel_that_lands_before_playback_starts_is_not_lost(logger, settings, aplay):
    """A cancel taken while nothing is consuming the queue still holds.

    The job is removed from the queue and finished on the spot, so when the
    worker does start it has nothing to speak. The worker additionally skips any
    job it pops that is already terminal, which is the same guarantee from the
    other side.
    """
    spk = SpeechPlayer(logger=logger, settings=settings)
    job = spk.enqueue(utterance(), text="hi", voice="af_heart", speed=1.0)

    # Cancel while nothing is consuming the queue at all.
    view = spk.cancel(job.id)
    assert view.status is JobStatus.CANCELLED

    spk.start()
    try:
        # The worker pops it, sees a terminal job, and skips it.
        assert wait_until(lambda: len(aplay.calls) == 0, timeout=0.5)
        assert spk.get(job.id).status is JobStatus.CANCELLED
    finally:
        spk.stop(timeout=5.0)


# --- Failures ---------------------------------------------------------------


def test_a_missing_aplay_fails_the_job_with_a_code(logger, settings, aplay):
    aplay.missing = True
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        job = spk.enqueue(utterance(), text="hi", voice="af_heart", speed=1.0)
        assert wait_until(lambda: spk.get(job.id).status.terminal)
        view = spk.get(job.id)
        assert view.status is JobStatus.FAILED
        assert view.code is Failure.PLAYER_UNAVAILABLE
        assert "alsa-utils" in view.error
    finally:
        spk.stop(timeout=5.0)


def test_a_nonzero_aplay_exit_fails_the_job_and_keeps_its_stderr(player, aplay):
    aplay.returncode = 1
    aplay.stderr = b"aplay: main:831: audio open error"

    job = speak(player)
    assert wait_until(lambda: finished(player, job.id))

    view = player.get(job.id)
    assert view.status is JobStatus.FAILED
    assert view.code is Failure.PLAYBACK_FAILED
    assert "audio open error" in view.error


def test_a_wedged_device_is_killed_and_reported_as_a_timeout(player, aplay):
    """The budget is the utterance's own length plus the configured margin."""
    aplay.hang = True

    job = speak(player)  # 0.1 s of audio, 0.3 s margin in the test settings
    assert wait_until(lambda: finished(player, job.id), timeout=5.0)

    view = player.get(job.id)
    assert view.status is JobStatus.FAILED
    assert view.code is Failure.PLAYBACK_TIMEOUT
    assert aplay.killed >= 1


def test_the_playback_thread_survives_an_unexpected_error(player, aplay, monkeypatch):
    """A thread that dies would queue every later utterance forever."""
    from syncai_tts import player as player_module

    boom = {"raised": False}
    real_resolve = player_module.resolve_playback_device

    def _explode(*args, **kwargs):
        if not boom["raised"]:
            boom["raised"] = True
            raise RuntimeError("udev exploded")
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(player_module, "resolve_playback_device", _explode)

    first = speak(player, text="doomed")
    assert wait_until(lambda: finished(player, first.id))
    assert player.get(first.id).status is JobStatus.FAILED

    second = speak(player, text="fine")
    assert wait_until(lambda: finished(player, second.id))
    assert player.get(second.id).status is JobStatus.DONE


# --- Queue management -------------------------------------------------------


def test_the_queue_refuses_work_it_would_only_speak_much_later(logger, aplay):
    settings = make_settings(max_queue=2)
    aplay.play_time = 5.0
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        spk.enqueue(utterance(), text="playing", voice="af_heart", speed=1.0)
        assert wait_until(lambda: spk.snapshot()["state"] == "playing")

        spk.enqueue(utterance(), text="q1", voice="af_heart", speed=1.0)
        spk.enqueue(utterance(), text="q2", voice="af_heart", speed=1.0)

        with pytest.raises(ConflictError) as excinfo:
            spk.enqueue(utterance(), text="q3", voice="af_heart", speed=1.0)
        assert excinfo.value.code is Failure.QUEUE_FULL
    finally:
        spk.stop(timeout=5.0)


def test_history_is_bounded_but_never_drops_a_live_job(logger, aplay):
    """A backlog may push `_jobs` past the cap; the cap bounds history, not work
    in flight. Every id a caller is holding has to stay resolvable."""
    settings = make_settings(job_history=2, max_queue=10)
    aplay.play_time = 5.0
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        playing = spk.enqueue(utterance(), text="playing", voice="af_heart", speed=1.0)
        assert wait_until(lambda: spk.get(playing.id).status is JobStatus.PLAYING)

        queued = [
            spk.enqueue(utterance(), text=f"q{i}", voice="af_heart", speed=1.0)
            for i in range(4)
        ]

        for job in [playing, *queued]:
            assert spk.get(job.id).id == job.id
    finally:
        spk.stop(timeout=5.0)


def test_a_job_that_just_finished_is_still_readable(logger, aplay):
    """The reason eviction runs on enqueue rather than on finish.

    With a history of one, trimming as a job ended would delete the job that
    just ended, and a caller polling the id it was handed would get a 404 for
    work that succeeded.
    """
    settings = make_settings(job_history=1)
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        job = spk.enqueue(utterance(), text="hi", voice="af_heart", speed=1.0)
        assert wait_until(lambda: spk.get(job.id).status.terminal)
        assert spk.get(job.id).status is JobStatus.DONE
    finally:
        spk.stop(timeout=5.0)


def test_old_finished_jobs_are_dropped_as_new_work_arrives(logger, aplay):
    settings = make_settings(job_history=2)
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        jobs = []
        for i in range(4):
            job = spk.enqueue(utterance(), text=f"line {i}", voice="af_heart", speed=1.0)
            jobs.append(job)
            assert wait_until(lambda: spk.get(job.id).status.terminal)

        # The two oldest have been evicted to make room for the two newest.
        for job in jobs[:2]:
            with pytest.raises(NotFoundError):
                spk.get(job.id)
        for job in jobs[2:]:
            assert spk.get(job.id).id == job.id
    finally:
        spk.stop(timeout=5.0)


def test_stopping_drops_the_queue_and_cuts_the_utterance(logger, settings, aplay):
    """What a SIGTERM to the container does, via the app's lifespan shutdown."""
    aplay.play_time = 5.0
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    playing = spk.enqueue(utterance(), text="playing", voice="af_heart", speed=1.0)
    # Audio actually going into aplay, so the stop below has a process to reap
    # rather than landing in the pre-spawn window (covered separately).
    assert wait_until(lambda: aplay.inside >= 1)
    queued = spk.enqueue(utterance(), text="queued", voice="af_heart", speed=1.0)

    spk.stop(timeout=5.0)

    assert spk.get(queued.id).status is JobStatus.CANCELLED
    assert spk.get(playing.id).status is JobStatus.CANCELLED
    assert aplay.terminated >= 1
    # And it refuses new work rather than accepting what it will never speak.
    with pytest.raises(ConflictError):
        spk.enqueue(utterance(), text="late", voice="af_heart", speed=1.0)


def test_stopping_before_playback_starts_still_cuts_the_utterance(logger, settings, aplay):
    """The same pre-spawn window as a cancel, reached by a shutdown instead.

    stop() marks the job in flight as well as signalling the process, so an
    utterance cannot start playing after the service has been told to stop.
    """
    gate = threading.Event()
    aplay.gate = gate
    aplay.play_time = 5.0

    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    playing = spk.enqueue(utterance(), text="playing", voice="af_heart", speed=1.0)
    assert wait_until(lambda: len(aplay.calls) == 1)  # inside the spawn, blocked

    stopper = threading.Thread(target=spk.stop, kwargs={"timeout": 5.0})
    stopper.start()
    gate.set()
    stopper.join(timeout=10.0)

    assert not stopper.is_alive()
    assert spk.get(playing.id).status is JobStatus.CANCELLED


def test_wait_returns_the_terminal_state(player, aplay):
    job = speak(player)
    view = player.wait(job.id, timeout=5.0)
    assert view.status is JobStatus.DONE


def test_wait_that_times_out_reports_the_job_still_going(player, aplay):
    aplay.play_time = 5.0
    job = speak(player)
    view = player.wait(job.id, timeout=0.1)
    assert view.status in (JobStatus.QUEUED, JobStatus.PLAYING)


def test_snapshot_reports_what_health_needs(player, aplay):
    aplay.play_time = 1.0
    job = speak(player)
    assert wait_until(lambda: player.snapshot()["state"] == "playing")

    snapshot = player.snapshot()
    assert snapshot["job_id"] == job.id
    assert snapshot["thread_alive"] is True
    assert snapshot["max_queue"] == 4


def test_concurrent_callers_all_get_a_job(player, aplay):
    """Four threads enqueueing at once: no lost ids, no duplicate ids."""
    aplay.play_time = 0.01
    ids = []
    ids_lock = threading.Lock()

    def _enqueue():
        view = speak(player)
        with ids_lock:
            ids.append(view.id)

    threads = [threading.Thread(target=_enqueue) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(set(ids)) == 4
    assert wait_until(lambda: all(finished(player, job_id) for job_id in ids))


# --- Device resolution ------------------------------------------------------


def test_the_udev_symlink_decides_the_alsa_device(tmp_path):
    """/dev/syncai/speaker_pcm -> ../snd/pcmC2D0p means plughw:2,0."""
    (tmp_path / "pcmC2D0p").write_text("")
    link = tmp_path / "speaker_pcm"
    os.symlink(tmp_path / "pcmC2D0p", link)

    assert resolve_playback_device(str(link), "fallback") == "plughw:2,0"


def test_a_missing_symlink_falls_back_rather_than_raising(tmp_path):
    """A speaker unplugged, or udev rules not installed. aplay is the error
    reporter with an actionable message; a broken link must not become a
    traceback."""
    missing = tmp_path / "speaker_pcm"
    assert resolve_playback_device(str(missing), "plughw:CARD=X,DEV=0") == "plughw:CARD=X,DEV=0"


def test_a_link_to_something_that_is_not_a_pcm_node_falls_back(tmp_path):
    (tmp_path / "not_a_pcm").write_text("")
    link = tmp_path / "speaker_pcm"
    os.symlink(tmp_path / "not_a_pcm", link)

    assert resolve_playback_device(str(link), "fallback") == "fallback"


def test_the_device_is_resolved_for_every_utterance(player, aplay, tmp_path):
    """A replug moves the card number, so a name resolved once would go stale."""
    job_one = speak(player)
    assert wait_until(lambda: finished(player, job_one.id))

    job_two = speak(player)
    assert wait_until(lambda: finished(player, job_two.id))

    assert len(aplay.devices) == 2
    assert all(device == "plughw:CARD=TEST,DEV=0" for device in aplay.devices)


def test_a_stopped_player_can_be_started_again(logger, settings, aplay):
    """Not used in production, where the lifespan starts it once — but a stop
    flag left set would make a restart look alive while speaking nothing."""
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    spk.stop(timeout=5.0)

    spk.start()
    try:
        job = spk.enqueue(utterance(), text="again", voice="af_heart", speed=1.0)
        assert wait_until(lambda: spk.get(job.id).status is JobStatus.DONE)
    finally:
        spk.stop(timeout=5.0)
