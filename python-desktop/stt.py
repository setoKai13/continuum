"""Push-to-talk voice capture: hold a key, record, transcribe offline.

Voice is the DECLENCHEUR (trigger) of the whole causal chain: no new
transcript means `agent.py`'s OBSERVE step returns nothing, which freezes
the hold-state. Everything native/network here (`pynput`, `sounddevice`,
`faster-whisper`) is imported lazily so this module stays importable
without those packages installed.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

from config import Settings

logger = logging.getLogger(__name__)

PRIMING_PROMPT_FR = (
    "Commandes pour piloter un Mac reel: ouvrir une application, cliquer, taper, "
    "assigner un ticket a une equipe, reprendre une tache, annuler, arreter."
)
PRIMING_PROMPT_EN = (
    "Commands to control a real Mac: open an application, click, type, "
    "assign a ticket to a team, resume a task, cancel, stop."
)


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

    def stop_and_save(self) -> Path:
        """Stops recording and writes the buffered audio to a temp WAV file.

        Returns:
            Path to the written WAV file (caller/OS is responsible for cleanup).

        Raises:
            MicrophoneError: If no stream was ever started.
        """
        import numpy as np  # lazy: ships alongside sounddevice

        if self._stream is None:
            raise MicrophoneError("stop_and_save() called before start()")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        audio = np.concatenate(self._frames) if self._frames else np.zeros((0, 1), dtype="int16")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio.tobytes())
        return Path(tmp.name)


class Transcriber:
    """Offline speech-to-text via faster-whisper, primed for Mac-control commands."""

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

    def transcribe(self, wav_path: Path) -> str:
        """Transcribes a WAV file, primed with a Mac-control vocabulary hint.

        Args:
            wav_path: Path to a mono 16-bit PCM WAV file.

        Returns:
            The transcript text, stripped, or "" if nothing was recognized.
        """
        model = self._ensure_model()
        prompt = PRIMING_PROMPT_FR if self._settings.language == "fr" else PRIMING_PROMPT_EN
        segments, _info = model.transcribe(
            str(wav_path), language=self._settings.language, initial_prompt=prompt
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
        wav_path: Path | None = None
        try:
            wav_path = self._recorder.stop_and_save()
            text = self._transcriber.transcribe(wav_path)
            if text:
                self._on_transcript(text)
        except Exception:  # noqa: BLE001 - a pynput callback must never raise
            logger.exception("push-to-talk transcription failed; hold the key and retry")
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

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
