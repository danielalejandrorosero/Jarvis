# 0011 — Eliminar el fallback local de TTS

## Contexto

ADR-0004 estableció "JARVIS nunca queda mudo": un fallback local obligatorio (SAPI vía
`pyttsx3`, envuelto en `FallbackTTSClient`) que se activaba si el TTS primario (hoy la API de
OpenAI, `gpt-4o-mini-tts`) fallaba por cualquier motivo.

Ese fallback se activó hoy en producción, confirmado en `data/jarvis-error.log`: la cuenta de
OpenAI se quedó sin créditos y JARVIS degradó a la voz robótica de SAPI. El usuario, al
enterarse, pidió explícitamente sacarlo: prefiere perder un turno puntual (silencio, no voz)
antes que escuchar esa voz bajo ninguna circunstancia. Él se encarga de mantener la cuenta con
crédito.

## Decisión

Eliminar `SapiTTSClient` y `FallbackTTSClient` de `jarvis.audio.tts`. `load_default_tts_client()`
devuelve un `OpenAITTSClient` directo. TTS depende exclusivamente de la API de OpenAI, sin red de
seguridad local. Se saca la dependencia `pyttsx3` del repo.

`LockingTTSClient` (serialización de `.speak()` con un lock, no relacionado con fallback de
proveedor) no se toca.

## Por qué

- Pedido explícito y sin ambigüedad del usuario, con el trade-off entendido de su parte: un fallo
  o agotamiento de crédito de la API ahora se escucha como silencio en ese turno puntual, no como
  una voz distinta.
- El proceso de JARVIS no se cae por esto: el `except Exception` de nivel de turno que ya existe
  en `run()` (`jarvis.audio.pipeline`) sigue protegiendo el loop principal; solo se pierde el
  turno en el que `speak()` falló, el siguiente turno funciona normal.

## Consecuencias

- Un fallo de la API de OpenAI para TTS ya no se enmascara con una voz robótica — se pierde el
  turno y queda logueado, visible en vez de oculto.
- Se saca `pyttsx3` de `pyproject.toml` (y su override de `mypy`); `comtypes`, que antes se
  justificaba parcialmente como transitiva de `pyttsx3` en el comentario de `pycaw`
  (`jarvis.tools.volume_control`), ahora se documenta como lo que siempre fue en la práctica: una
  dependencia declarada por el propio `pycaw`.
- ADR-0004 no se borra ni se reescribe — queda como registro histórico de una decisión tomada y
  luego revertida (este ADR es la reversión formal, igual que ADR-0010 lo fue para el overlay).
