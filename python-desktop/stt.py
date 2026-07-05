"""Push-to-talk voice capture: hold a key, record, transcribe offline.

Voice is the DECLENCHEUR (trigger) of the whole causal chain: no new
transcript means `agent.py`'s OBSERVE step returns nothing, which freezes
the hold-state. Everything native/network here (`pynput`, `sounddevice`,
`faster-whisper`) is imported lazily so this module stays importable
without those packages installed.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any, Callable

from config import Settings

logger = logging.getLogger(__name__)

# A press-and-release shorter than this cannot contain speech: skip the
# transcription entirely. Whisper handed near-empty audio tends to
# hallucinate plausible-sounding commands out of nothing.
MIN_UTTERANCE_S = 0.3


class MicrophoneError(Exception):
    """Raised when audio recording cannot start or produced no data."""


def resolve_key(name: str) -> Any:
    """Maps a settings key name (e.g. "f8", "esc") to a `pynput` key object.

    Args:
        name: Key name from Settings (PTT_KEY / KILL_KEY).

    Returns:
        A `pynput.keyboard.Key` (special key) or `KeyCode` (single char).
    """
    from pynput import keyboard  # lazy: native input dependency

    lowered = name.strip().lower()
    special = getattr(keyboard.Key, lowered, None)
    if special is not None:
        return special
    return keyboard.KeyCode.from_char(lowered)


class AudioRecorder:
    """Records mono 16-bit PCM audio from the default microphone."""

    def __init__(self, sample_rate: int = 16000) -> None:
        """Prepares a recorder without opening any audio stream yet.

        Args:
            sample_rate: Capture sample rate in Hz.
        """
        self._sample_rate = sample_rate
        self._frames: list[Any] = []
        self._stream = None

    def start(self) -> None:
        """Opens the microphone input stream and starts buffering frames."""
        import sounddevice as sd  # lazy: native audio dependency

        self._frames = []

        def _callback(indata, _frames, _time_info, _status) -> None:
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16", callback=_callback
        )
        self._stream.start()

    def stop_and_wav_bytes(self) -> tuple[bytes, float]:
        """Stops recording and returns the buffered audio as in-memory WAV bytes.

        Returns:
            A (wav_bytes, duration_seconds) tuple. No temp file touches disk:
            the bytes go straight into the transcriber (mirroring the
            wav-bytes -> transcribe shape of the artemis LiveKit wrapper).

        Raises:
            MicrophoneError: If no stream was ever started.
        """
        import numpy as np  # lazy: ships alongside sounddevice

        if self._stream is None:
            raise MicrophoneError("stop_and_wav_bytes() called before start()")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        audio = np.concatenate(self._frames) if self._frames else np.zeros((0, 1), dtype="int16")
        duration_s = len(audio) / self._sample_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio.tobytes())
        return buffer.getvalue(), duration_s


class Transcriber:
    """Offline speech-to-text via faster-whisper.

    Deliberately un-primed: an `initial_prompt` makes whisper echo the prompt
    back verbatim on silent/near-silent audio, which the agent then treats as
    a real spoken instruction. `vad_filter=True` stands in for the upstream
    VAD the artemis LiveKit wrapper gets from Silero -- here the push-to-talk
    key is the only other gate, so whisper's own VAD must strip the silence.
    """

    def __init__(self, settings: Settings) -> None:
        """Stores settings; the whisper model loads lazily on first use.

        Args:
            settings: Application settings (drives the priming language).
        """
        self._settings = settings
        self._model = None

    def _ensure_model(self) -> Any:
        """Lazily loads the configured faster-whisper model (int8 for CPU speed)."""
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # lazy: heavy ML dependency

        self._model = WhisperModel(self._settings.whisper_model, compute_type="int8")
        return self._model

    def warm_up(self) -> None:
        """Loads the whisper model now, so the FIRST push-to-talk is not the
        one paying the multi-second (or download) model load inside the
        keyboard callback thread."""
        self._ensure_model()

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribes in-memory WAV audio.

        Args:
            wav_bytes: A mono 16-bit PCM WAV payload.

        Returns:
            The transcript text, stripped, or "" if nothing was recognized
            (silence removed by the VAD yields zero segments, not a
            hallucinated command).
        """
        model = self._ensure_model()
        segments, _info = model.transcribe(
            io.BytesIO(wav_bytes),
            language=self._settings.language,
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class PushToTalkListener:
    """Binds a global hotkey to start/stop recording and emit transcripts.

    While the configured `push_to_talk_key` is held, audio is recorded; on
    release it is transcribed and handed to `on_transcript`. This is the
    only source of new `instruction` observations feeding `agent.py`.
    """

    def __init__(self, settings: Settings, on_transcript: Callable[[str], None]) -> None:
        """Wires the listener to its settings and transcript callback.

        Args:
            settings: Application settings (push_to_talk_key, language).
            on_transcript: Called with the recognized text on key release.
        """
        self._settings = settings
        self._on_transcript = on_transcript
        self._recorder = AudioRecorder()
        self._transcriber = Transcriber(settings)
        self._held = False
        self._listener = None
        self._target_key = None

    def _on_press(self, key: Any) -> None:
        if self._held or key != self._target_key:
            return
        self._held = True
        try:
            self._recorder.start()
        except Exception:  # noqa: BLE001 - a pynput callback must never raise
            # An exception escaping a pynput callback kills the listener
            # thread silently: after that, no voice would ever be heard again
            # with zero visible error. Swallow + log instead.
            logger.exception("microphone recording failed to start")
            self._held = False

    def _on_release(self, key: Any) -> None:
        if not self._held or key != self._target_key:
            return
        self._held = False
        try:
            wav_bytes, duration_s = self._recorder.stop_and_wav_bytes()
            if duration_s < MIN_UTTERANCE_S:
                logger.info("push-to-talk released after %.2fs; too short, ignored", duration_s)
                return
            text = self._transcriber.transcribe(wav_bytes)
            if text:
                self._on_transcript(text)
        except Exception:  # noqa: BLE001 - a pynput callback must never raise
            logger.exception("push-to-talk transcription failed; hold the key and retry")

    def start(self) -> None:
        """Starts the background keyboard listener thread.

        Also warms up the whisper model NOW: the first model load can take
        seconds (or a download), and paying it lazily inside the keyboard
        callback would freeze the very first spoken instruction of the demo.
        """
        from pynput import keyboard  # lazy: native input dependency

        self._transcriber.warm_up()
        self._target_key = resolve_key(self._settings.push_to_talk_key)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        """Stops the background keyboard listener thread, if running."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
