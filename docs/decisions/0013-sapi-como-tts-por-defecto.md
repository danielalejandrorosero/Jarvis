# 0013 — SAPI como TTS por defecto, reabre parcialmente ADR-0011

## Contexto

ADR-0011 sacó el fallback local de TTS (`SapiTTSClient`) y dejó `OpenAITTSClient` como único
camino: si la API de OpenAI fallaba, el turno se perdía en silencio en vez de degradar a una voz
robótica. La decisión tenía sentido en el momento (el fallback se había activado por agotamiento
real de crédito y el usuario prefería silencio antes que esa voz), pero en uso real el problema
se repitió: la cuenta de OpenAI volvió a quedarse sin crédito más de una vez, y cada vez JARVIS
quedó completamente mudo — sin fallback, sin nada.

El usuario, en vivo, resolvió la tensión de raíz: "para qué quiero que me hablen bonito... el
importante es el transcriptor" — no le importa la calidad de la voz de salida, lo que le importa
es que el STT (transcripción de lo que él dice) sea bueno, y eso ya está resuelto aparte
(Speechmatics, ADR-0012, sin depender de crédito de OpenAI).

## Decisión

`load_default_tts_client()` devuelve `SapiTTSClient()` (voz local de Windows vía `pyttsx3`) en
vez de `OpenAITTSClient()`. `OpenAITTSClient` se mantiene definida en el módulo — la interfaz
`TTSClient` sigue siendo swappable a propósito, por si en el futuro se quiere volver a la voz
neural — pero no es el default. `pyttsx3` vuelve a `pyproject.toml`.

**Lo que ADR-0011 estableció y NO se reabre acá**: seguir sin ningún fallback en cascada
(`FallbackTTSClient` no vuelve). Es un solo `TTSClient` a la vez, sin degradación silenciosa a
mitad de turno — solo cambia CUÁL es ese único cliente por defecto. Si `SapiTTSClient.speak()`
fallara (más improbable que una API remota: es local, sin red), la excepción se sigue propagando
igual que con `OpenAITTSClient` hoy, capturada por el mismo `except Exception` de nivel de turno
en `run()`.

## Por qué

- **SAPI es local, gratis, sin cuenta ni crédito externo** — elimina por completo la clase de
  falla que motivó tanto ADR-0011 como este mismo ADR: JARVIS ya no puede quedarse mudo por un
  problema de facturación de un proveedor externo, porque no depende de ninguno para hablar.
- **La calidad de voz nunca fue el problema real que el usuario quería resolver** — lo dijo
  explícitamente. La inversión de esta sesión en TTS (streaming, instrucciones de acento
  colombiano, comparación de proveedores) resolvía un problema que, en la práctica, le importaba
  mucho menos que la disponibilidad y que el STT funcione bien.
- **Mantener `OpenAITTSClient` en el código, no borrarla**: es la misma interfaz `TTSClient`, sin
  costo de mantenimiento real por dejarla ahí, y evita tener que reescribirla desde cero si el
  usuario cambia de opinión más adelante.

## Consecuencias

- JARVIS vuelve a sonar con la voz robótica de Windows (SAPI), no la voz neural entrenada. Trade-off
  aceptado explícitamente por el usuario, no un accidente.
- El riesgo de thread-safety de SAPI/COM bajo llamadas concurrentes (documentado ya en
  `LockingTTSClient` antes de ADR-0011, y ahora vuelve a aplicar) sigue mitigado por el mismo lock
  global — sin cambios ahí.
- `pyttsx3` vuelve a ser una dependencia declarada.
