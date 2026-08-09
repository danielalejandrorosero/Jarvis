"""Tests para el núcleo puro de transcripción de audio a texto (`transcribe`).

No pegan a la API real de OpenAI (`load_stt_client`): `transcribe()` recibe un cliente `OpenAI`
stub, controlado por el test, cuyo `.audio.transcriptions.create()` está mockeado (mismo enfoque
que `tests/test_llm_client.py`/`tests/audio/test_wake_word.py`: núcleo puro con dependencia
externa stubeada), así que corren rápido, sin red y sin `OPENAI_API_KEY` real (CI-safe).
"""

from __future__ import annotations

import wave
from typing import Any, cast

import numpy as np
from openai import Omit, OpenAI

from jarvis.audio import stt
from jarvis.audio.stt import transcribe

SAMPLE_RATE = 16_000


class _FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _FakeTranscription:
    """Stub de la respuesta de `client.audio.transcriptions.create()`: expone `.text` y
    `.logprobs` (default `None`, igual que la API real cuando no se pide `include=["logprobs"]`
    o el modelo no los devuelve), lo único que usa `transcribe()`."""

    def __init__(
        self, text: str, *, logprobs: list[_FakeLogprob] | None = None
    ) -> None:
        self.text = text
        self.logprobs = logprobs


class _FakeTranscriptions:
    """Stub de `client.audio.transcriptions`.

    `.create()` devuelve una respuesta predefinida y registra con qué kwargs fue llamada en
    `.calls`, para poder inspeccionarla desde el test (spy) sin pegarle a la red real.
    """

    def __init__(self, response: _FakeTranscription) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeTranscription:
        self.calls.append(kwargs)
        return self._response


class _FakeAudio:
    def __init__(self, transcriptions: _FakeTranscriptions) -> None:
        self.transcriptions = transcriptions


class _FakeOpenAIClient:
    """Stub de `openai.OpenAI`: solo expone `.audio.transcriptions`, lo único que usa
    `transcribe()`."""

    def __init__(self, response: _FakeTranscription) -> None:
        self.audio = _FakeAudio(_FakeTranscriptions(response))


def _client_with(
    text: str, *, logprobs: list[_FakeLogprob] | None = None
) -> tuple[OpenAI, _FakeTranscriptions]:
    # El stub no implementa la superficie completa de OpenAI, solo lo que `transcribe()` usa
    # (`.audio.transcriptions.create()`). Se castea para satisfacer la firma de
    # `transcribe(..., client: OpenAI)`.
    fake_client = _FakeOpenAIClient(_FakeTranscription(text, logprobs=logprobs))
    return cast(OpenAI, fake_client), fake_client.audio.transcriptions


def test_transcribe_returns_stripped_result_text() -> None:
    """Caso aceptado: el `.text` de la respuesta de la API se devuelve stripeado."""
    client, _spy = _client_with("  hola cómo estás  \n")
    audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

    result = transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert result == "hola cómo estás"


def test_transcribe_returns_empty_string_when_result_text_is_blank() -> None:
    """Silencio / sin voz detectada: la API puede devolver texto vacío o solo espacios, no es
    un error, `transcribe()` simplemente devuelve el string (stripeado, posiblemente vacío)."""
    client, _spy = _client_with("   ")
    audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

    result = transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert result == ""


def test_transcribe_passes_default_language_through_to_create() -> None:
    """El parámetro `language` (default "es") se reenvía tal cual a `create()`."""
    client, spy = _client_with("ok")
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert len(spy.calls) == 1
    assert spy.calls[0]["language"] == "es"


def test_transcribe_passes_explicit_language_through_to_create() -> None:
    """Un `language` explícito (distinto del default) también se reenvía tal cual."""
    client, spy = _client_with("ok")
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, client=client, sample_rate=SAMPLE_RATE, language="en")

    assert spy.calls[0]["language"] == "en"


def test_transcribe_with_language_none_sends_omit_instead_of_string() -> None:
    """`language=None`: `create()` recibe un `Omit()` (para dejar que la API autodetecte el
    idioma), nunca el string `"es"` del default ni el literal `None`."""
    client, spy = _client_with("ok")
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, client=client, sample_rate=SAMPLE_RATE, language=None)

    sent_language = spy.calls[0]["language"]
    assert isinstance(sent_language, Omit)


def test_transcribe_sends_model_kwarg() -> None:
    """El modelo enviado a `create()` es el configurado en el módulo (`stt.MODEL`), no hardcodeado
    acá, para no tener que tocar este test cada vez que se cambie de modelo."""
    client, spy = _client_with("ok")
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert spy.calls[0]["model"] == stt.MODEL


def test_transcribe_sends_deterministic_decoding_and_logprobs_kwargs() -> None:
    """`temperature=0.0` (decoding determinístico, menos alucinación), `response_format="json"`
    y `include=["logprobs"]` (requisito de la API para que devuelva logprobs con este modelo) se
    mandan siempre — ver docstring del módulo para el motivo."""
    client, spy = _client_with("ok")
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert spy.calls[0]["temperature"] == stt.TEMPERATURE
    assert spy.calls[0]["response_format"] == "json"
    assert spy.calls[0]["include"] == ["logprobs"]


def test_transcribe_logs_average_logprob_when_present(caplog: Any) -> None:
    """Cuando la respuesta trae `logprobs`, se loguea el promedio a INFO — evidencia real para
    poder calibrar un filtro de confianza más adelante (ver docstring de `transcribe`)."""
    client, _spy = _client_with("ok", logprobs=[_FakeLogprob(-0.1), _FakeLogprob(-0.3)])
    audio = np.zeros(4, dtype=np.int16)

    with caplog.at_level("INFO", logger="jarvis.audio.stt"):
        transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert any("confianza promedio" in record.message for record in caplog.records)


def test_transcribe_does_not_crash_when_logprobs_is_none() -> None:
    """`logprobs=None` (default del stub, igual que la respuesta real cuando el modelo no los
    devuelve) no debe romper `transcribe()` — nunca hay `logprobs` que loguear en ese caso."""
    client, _spy = _client_with("ok", logprobs=None)
    audio = np.zeros(4, dtype=np.int16)

    result = transcribe(audio, client=client, sample_rate=SAMPLE_RATE)

    assert result == "ok"


def test_transcribe_packages_audio_as_valid_wav_with_correct_format() -> None:
    """El audio se empaqueta como un WAV válido (mono, 16-bit, con el sample_rate pedido) antes
    de enviarlo, y el archivo tiene un `.name` terminado en `.wav` (el SDK de OpenAI lo usa para
    inferir el content-type)."""
    client, spy = _client_with("ok")
    audio = np.array([100, -100, 32767, -32768, 0], dtype=np.int16)
    sample_rate = 22_050

    transcribe(audio, client=client, sample_rate=sample_rate)

    sent_file = spy.calls[0]["file"]
    assert sent_file.name.endswith(".wav")

    sent_file.seek(0)
    with wave.open(sent_file, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == len(audio)
        frames = wf.readframes(wf.getnframes())
        assert np.frombuffer(frames, dtype=np.int16).tolist() == audio.tolist()
