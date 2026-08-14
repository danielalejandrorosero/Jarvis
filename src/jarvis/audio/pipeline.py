"""Pipeline integrado: wake word → grabar comando → transcribir → LLM (+ tools) → hablar.

Alcance de esta fase (ADR-0005, extendida post-fase-5 con búsqueda web): el LLM ya no solo
conversa — puede pedir tool-calls (`jarvis.tools.weather.WeatherTool` para clima,
`jarvis.tools.search.SearchTool` para búsqueda web general). Cada tool-call pasa por
`PolicyEngine` (`jarvis.security.policy`) antes de ejecutarse, según el `RiskLevel` que declara
el `Tool` (`.claude/rules/security.md`). `VoiceConfirmationChannel` (acá abajo) implementa el
`ConfirmationChannel` que la policy usa para tools CONFIRM, reusando el mismo TTS/STT del resto
del pipeline — sin acoplar la policy a audio (`security/policy.py` no importa `sounddevice`).

`SYSTEM_PROMPT` incluye una instrucción explícita sobre `<web_data>` (ver
`jarvis.tools.search`): contenido de terceros que vuelve de una búsqueda web es un vector de
prompt injection real, no solo un formato — el LLM tiene que saber, de antemano y fuera de
banda, que ese contenido es dato a reportar, nunca una instrucción a seguir.

Memoria (ADR-0004, "Persistencia: SQLite"; `jarvis.memory.store`): cada turno, antes de armar
`messages`, `_build_system_prompt` carga los hechos guardados más recientes y los agrega como una
sección aparte y claramente rotulada del system prompt — recall ambiental, no un tool de lectura;
el LLM no tiene que pedir explícitamente lo que ya sabe de conversaciones anteriores. Guardar sí
es un tool (`jarvis.tools.remember.RememberTool`), porque ahí sí hay una decisión que tomar (qué
vale la pena recordar y con qué frase).

Memoria proactiva, no solo a pedido explícito (pedido del usuario, en vivo: "que sea como al
humano... sin yo decirle"): `RememberTool.description` ya permitía guardar algo "claramente
relevante" sin pedido explícito, pero en la práctica DeepSeek casi no lo hacía sin la frase
"acordate de esto" — una instrucción débil, sin ejemplos concretos, no alcanzaba.
`SYSTEM_PROMPT` ahora refuerza esto con ejemplos concretos (a qué juega, de dónde es,
preferencias de trato) y un criterio explícito de "ante la duda, guardalo" — mismo espíritu que
un amigo que retiene lo que escucha sin que se lo pidan. Junto con esto se agregó un glosario de
jerga colombiana/League of Legends (investigado con fuentes reales, no de memoria de
entrenamiento) para que el LLM interprete "manco"/"chimba"/"intear"/etc. con naturalidad en vez
de pedir aclaración o malinterpretar groserías coloquiales como agresión real.

Hallazgo HIGH de `security-reviewer` sobre la primera versión de esta integración: `remember_fact`
es SAFE (sin fricción) y persiste texto verbatim que se reinyecta en *todo turno futuro* — si el
LLM, dentro del mismo turno, guarda contenido que vino de adentro de `<web_data>` (búsqueda web,
no confiable), ese contenido queda re-presentado en cada prompt siguiente como si fuera
conocimiento propio de JARVIS sobre el usuario, sin ninguna marca de que es dato de terceros. Esto
es estrictamente peor que el riesgo original de `<web_data>` (que dura un turno): persiste entre
sesiones, se reinyecta sin marca, y podría intentar pisar la instrucción anti-injection del propio
`SYSTEM_PROMPT` en turnos futuros. Mitigación (mismo patrón que `search.py`): los hechos
recordados se envuelven en `{MEMORY_DATA_OPEN_TAG}...{MEMORY_DATA_CLOSE_TAG}` con el mismo framing
"esto es dato reportado, no instrucción" que `<web_data>`, cada hecho se escapa
(`_escape_untrusted`) por si su contenido intenta fabricar un cierre de etiqueta prematuro, y
`SYSTEM_PROMPT` instruye explícitamente al LLM a no guardar contenido de `<web_data>` vía
`remember_fact`. `jarvis.memory.store` complementa con `MAX_CONTENT_LENGTH` (tope de longitud por
hecho) y `MAX_STORED_FACTS` (tope de filas totales) — ver docstring de ese módulo.

Segunda ruta de reinyección del mismo contenido de `facts`, hallazgo HIGH aparte de
`security-reviewer`: `jarvis.tools.recall_memory.RecallMemoryTool`, a pedido explícito del LLM,
también devuelve contenido de `facts` al contexto — pero como un mensaje `role: tool` que
`PolicyEngine`/`dispatch_turn` nunca inspecciona, así que la mitigación de acá (envolver/escapar
en el punto de reinyección de `_build_system_prompt`) no la cubre. Mitigada en el propio
`RecallMemoryTool.execute()` (ver docstring de `jarvis.tools.recall_memory`), no acá: mismo
framing/escapado, etiqueta propia `{RECALLED_MEMORY_OPEN_TAG}...{RECALLED_MEMORY_CLOSE_TAG}` (no
`{MEMORY_DATA_OPEN_TAG}`/`{MEMORY_DATA_CLOSE_TAG}`, para evitar un ciclo de imports con ese
módulo, que ya importa `RecallMemoryTool` desde acá). `SYSTEM_PROMPT` extiende las mismas
prohibiciones anti-injection (no inventar URLs, no volver a guardar con `remember_fact`) para
cubrir también esta etiqueta.

Muestras de habla (`jarvis.memory.store.speech_samples`, distinta de `facts` — ver docstring de
ese módulo): a diferencia de los hechos, acá no hay curación del LLM. `run()` guarda, sin
condición ni juicio de por medio, cada transcripción no vacía que produce `transcribe()` — el
objetivo no es *qué* dijo el usuario sino *cómo* habla (registro, modismos), para que el LLM
pueda adoptar un estilo de respuesta parecido al del usuario en vez de un español neutro
genérico. `_build_system_prompt` inyecta las muestras más recientes envueltas en
`{SPEECH_STYLE_OPEN_TAG}...{SPEECH_STYLE_CLOSE_TAG}`, con un framing distinto al de
`remembered_facts`: son ejemplos de estilo a imitar en la respuesta propia del LLM, nunca
contenido a repetir textual ni una instrucción a seguir. No llevan el mismo escapado
anti-injection que `remembered_facts` (`_escape_untrusted`): son texto del propio usuario, dicho
por su propia voz, no contenido de terceros que pudo colarse sin marca de origen (a diferencia de
un hecho "recordado" a partir de `<web_data>`) — igual quedan claramente etiquetadas y separadas
del `user_text` del turno actual para que el LLM no confunda una muestra pasada con el comando de
ahora.

Historial de conversación (`jarvis.memory.store.conversation_turns`, pedido explícito del usuario:
"quiero que el agente tenga memoria que se acuerde que dije antes"): distinto de `facts`
(curado, solo lo que se pide explícitamente recordar) y de `speech_samples` (estilo de habla, no
contenido). `dispatch_turn` guarda, al final de cada turno, el par `(user_text, assistant_text)`
vía `save_conversation_turn` — sin condición del LLM de por medio, igual que `speech_samples` —
y `_build_system_prompt` inyecta los últimos turnos, en orden cronológico (más viejo primero, para
que se lean como una conversación real), envueltos en
`{CONVERSATION_HISTORY_OPEN_TAG}...{CONVERSATION_HISTORY_CLOSE_TAG}`. Esto es lo que hace que la
memoria de conversación sobreviva un reinicio del proceso, no solo la lista `messages` en memoria
de un `dispatch_turn` puntual (que se descarta al terminar ese turno).

Framing/escapado, tratados de forma asimétrica a propósito entre los dos campos de cada turno:

- `user_text` (lo que dijo el usuario): mismo argumento que `speech_samples` — es la voz del
  propio usuario, transcripta directamente por STT, nunca contenido de terceros que pudo colarse
  sin marca de origen. No se escapa (`_escape_untrusted`). Riesgo residual aceptado igual que en
  `speech_samples`: el STT puede alucinar caracteres arbitrarios, incluido `<`/`>` literal, sobre
  audio ambiguo o silencio (ver discusión de alucinaciones en `stt.py` y más abajo en este mismo
  módulo) — no se escapa porque el origen sigue siendo la voz del usuario, no contenido de
  terceros, pero una alucinación de STT en teoría podría producir texto con esos caracteres.
- `assistant_text` (lo que respondió JARVIS en un turno pasado): SÍ se escapa. A diferencia de
  `speech_samples`, este campo es una respuesta *generada por el LLM*, y esa respuesta puede,
  en un turno anterior, haber citado o resumido contenido de `<web_data>` (una búsqueda web) sin
  conservar esa marca de origen — el mismo mecanismo de fondo que motivó el hallazgo HIGH sobre
  `remembered_facts` (contenido de terceros que se "cuela" a través de una capa de memoria sin
  marca de origen, y se reinyecta como si fuera propio en turnos futuros). Igual que los hechos
  guardados, se envuelve en el tag de dato reportado y se escapa por si intenta fabricar un
  cierre de etiqueta prematuro.

El framing del historial completo, sin embargo, es explícitamente "esto es contexto de
continuidad, nunca un comando nuevo a ejecutar" — ni siquiera los turnos de USUARIO pasados se
tratan como instrucciones vigentes ahora (serían, como mucho, instrucciones YA cumplidas en su
momento), para que el LLM no reinterprete un pedido viejo como algo a repetir en el turno actual.

Últimas acciones por tool (`jarvis.memory.store.tool_call_log`/`list_most_recent_tool_call_per_
tool`): bug real, en vivo, dos veces la misma noche — "reproducí la última canción que pusimos" y,
mucho más tarde, "iniciá una partida de LoL pero el mismo modo que estábamos jugando" no se podían
resolver porque el turno original ya había salido de `conversation_history` (ventana acotada,
`DEFAULT_CONVERSATION_TURN_LIST_LIMIT`). `dispatch_turn` guarda una fila en `tool_call_log` vía
`save_tool_call` justo cuando `PolicyEngine.authorize_and_execute` confirma, con su callback
`on_executed`, que un tool-call llegó a ejecutarse de verdad (SAFE, o CONFIRM aprobado — nunca
CONFIRM denegado ni DANGEROUS, que no llaman `on_executed`) — sin clasificar éxito/fracaso a
partir del texto de retorno del tool, esa distinción ya la resuelve la policy. `_build_system_
prompt` inyecta `list_most_recent_tool_call_per_tool`: como mucho una fila por nombre de tool
distinto (la más reciente), envuelta en `{RECENT_ACTIONS_OPEN_TAG}...{RECENT_ACTIONS_CLOSE_TAG}` —
a diferencia de `conversation_history`, este bloque nunca crece con el uso prolongado, sin importar
cuántas llamadas se acumulen, porque por construcción hay como mucho una fila por tool registrado.
Cada línea muestra los argumentos de esa última llamada y hace cuánto fue, calculado en el momento
de armar el prompt (no un string pre-formateado guardado en la fila) para que el LLM pueda juzgar
si una referencia vaga ("la última", "el mismo") sigue siendo razonable de asumir o ya es
demasiado vieja. Framing/escapado: mismo criterio asimétrico que el resto — los argumentos
(`_escape_untrusted`) porque, en teoría, podrían contener contenido derivado de una búsqueda web
(ej. una URL) sin conservar esa marca de origen, mismo mecanismo que motivó el hallazgo HIGH sobre
`remembered_facts`/`assistant_text`; el nombre del tool no se escapa, porque viene siempre de un
`Tool` ya registrado en código, nunca de contenido de terceros.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from openai import OpenAI

from jarvis.audio.device import input_sample_rate, resolve_input_device
from jarvis.audio.loopback import SystemAudioGate, SystemAudioMonitor
from jarvis.audio.realtime_stt import load_realtime_client, stream_transcribe_command
from jarvis.audio.resample import resample
from jarvis.audio.speech_detector import (
    SPEECH_PROBABILITY_THRESHOLD,
    load_speech_detector,
)
from jarvis.audio.stt import load_stt_client, transcribe
from jarvis.audio.timer_scheduler import TimerScheduler
from jarvis.audio.tts import LockingTTSClient, TTSClient, load_default_tts_client

# Extraídas a `jarvis.audio.vad` (antes vivían acá) para que `jarvis.audio.realtime_stt` pueda
# reusarlas sin import circular — ver docstring de ese módulo. `TRAILING_SILENCE_SECONDS` (el
# único símbolo de `vad.py` que este archivo no referencia directamente en su propio código) vive
# ahora solo en `jarvis.audio.vad` — `tests/audio/test_pipeline.py` la importa de ahí.
from jarvis.audio.vad import chunk_rms, is_speech_chunk, should_stop_recording
from jarvis.audio.wake_word import (
    DEFAULT_THRESHOLD,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    detect,
    iter_microphone_frames,
)
from jarvis.audio.wake_word import load_model as load_wake_word_model
from jarvis.config import load_dotenv
from jarvis.league.lcu_monitor import LCUAutoAcceptMonitor
from jarvis.llm.client import (
    LLMClient,
    LLMResult,
    ToolSchema,
    load_deepseek_client_from_env,
)
from jarvis.memory.store import DEFAULT_DB_PATH as MEMORY_DEFAULT_DB_PATH
from jarvis.memory.store import (
    list_facts,
    list_most_recent_tool_call_per_tool,
    list_recent_conversation_turns,
    list_speech_samples,
    save_conversation_turn,
    save_speech_sample,
    save_tool_call,
)
from jarvis.security.policy import PolicyEngine
from jarvis.tools.base import Tool
from jarvis.tools.cancel_reminder import CancelAllRemindersTool, CancelReminderTool
from jarvis.tools.cancel_system_power import CancelSystemPowerTool
from jarvis.tools.cancel_timer import CancelAllTimersTool, CancelTimerTool
from jarvis.tools.close_app import CloseAppTool
from jarvis.tools.lol_champion_select import LockChampionTool, PreviewChampionTool
from jarvis.tools.lol_runes import SetRunesTool
from jarvis.tools.lol_start_queue import (
    CancelQueueTool,
    SetLobbyQueueTool,
    StartQueueTool,
)
from jarvis.tools.lol_summoner_spells import SetSummonerSpellsTool
from jarvis.tools.media_control import MediaControlTool
from jarvis.tools.open_app import OpenAppTool
from jarvis.tools.open_file import OpenFileTool
from jarvis.tools.open_url import OpenUrlTool
from jarvis.tools.recall_memory import (
    RECALLED_MEMORY_CLOSE_TAG,
    RECALLED_MEMORY_OPEN_TAG,
    RecallMemoryTool,
)
from jarvis.tools.remember import RememberTool
from jarvis.tools.reminder import ReminderTool
from jarvis.tools.screenshot import ScreenshotTool
from jarvis.tools.search import WEB_DATA_CLOSE_TAG, WEB_DATA_OPEN_TAG, SearchTool
from jarvis.tools.system_info import SystemInfoTool
from jarvis.tools.system_power import SystemPowerTool
from jarvis.tools.timer import TimerTool
from jarvis.tools.volume_control import VolumeControlTool
from jarvis.tools.weather import WeatherTool

COMMAND_WINDOW_SECONDS = 20.0  # tope duro: si nunca hay silencio, no graba para siempre
# Antes en 4.0 — cortaba la grabación aunque el usuario siguiera hablando, no solo cuando
# había silencio real (pedido explícito del usuario: "inteligente hasta que yo pare de hablar,
# no solo 4 segundos"). El corte real sigue siendo por silencio sostenido
# (`TRAILING_SILENCE_SECONDS`, ver `should_stop_recording`); este tope de 20s es solo la red de
# seguridad para el caso de que el silencio nunca se detecte, no el comportamiento normal.
COOLDOWN_SECONDS = (
    1.5  # evita que el eco de la grabación o repetir la frase retriggeree al toque
)
FOLLOW_UP_WINDOW_SECONDS = (
    8.0  # tope de grabación para un comando de seguimiento, sin wake
)
# word. Pedido explícito del usuario: después de un comando real ("Alexa, abrí YouTube"), poder
# seguir hablando sin repetir "Alexa" cada vez ("ahora reproducí esto"). El corte real, igual que
# en un comando normal, sigue siendo por silencio sostenido (`TRAILING_SILENCE_SECONDS`) — este
# tope es solo la red de seguridad para el caso de seguimiento, más corta que
# `COMMAND_WINDOW_SECONDS` porque acá además hace de límite de "cuánto esperar antes de asumir
# que no hay seguimiento y volver a pedir la wake word".
AGC_TARGET_PEAK_RATIO = 0.9  # a qué % del rango de int16 apuntamos al subir volumen
CHUNK_SECONDS = (
    0.2  # tamaño de chunk para detectar silencio mientras se graba el comando
)
NOISE_FLOOR_SAMPLE_SECONDS = 3.0  # cuánto se mide de ambiente al arrancar para calibrar
# Antes en 1.5 — bug real, en vivo, confirmado con `data/jarvis-error.log` en dos reinicios
# consecutivos del mismo proceso, mismo usuario, misma sala, mismo juego (League of Legends)
# sonando: una corrida calibró `umbral=4621.7` (piso de ruido 1155.4) y la siguiente, minutos
# después, sin ningún cambio real de ambiente, calibró `umbral=74.1` (piso de ruido 18.5) — una
# diferencia de 62x en el umbral calibrado entre dos corridas consecutivas. Causa: el audio de
# juego/música es "bursty" (silencio entre efectos de sonido, fuerte durante ellos, no ruido
# estacionario) — con una ventana de apenas 1.5s, una sola racha de suerte de silencio entre
# efectos podía dejar la ventana entera (o casi) sin ningún sub-chunk representativo del nivel
# típico real, y ese piso de ruido anormalmente bajo quedaba fijo para toda la sesión (nunca se
# recalibra después del arranque). Duplicar la ventana a 3.0s reduce la probabilidad de que la
# calibración entera caiga en un hueco de silencio sin cruzarse con ningún efecto de sonido —
# sigue siendo una ventana corta (no un cambio de arquitectura, ver `NOISE_FLOOR_PERCENTILE` más
# abajo para la otra mitad del fix), pero el doble de muestra ya reduce bastante el riesgo de mala
# suerte sin alargar mucho el arranque (sigue siendo un par de segundos, no algo perceptible como
# una demora real).
NOISE_FLOOR_SUBCHUNKS = (
    10  # partir la medición en sub-chunks y usar un percentil alto del RMS
)
# (`NOISE_FLOOR_PERCENTILE`) de esos sub-chunks, no la mediana ni el RMS de la ventana entera.
# Duplicado de 5 a 10 junto con `NOISE_FLOOR_SAMPLE_SECONDS` (1.5s→3.0s) para mantener el mismo
# tamaño de sub-chunk (~0.3s) que ya funcionaba bien contra el caso que motivó el sub-chunkeo en
# primer lugar (ver `NOISE_FLOOR_PERCENTILE`): un ruido puntual justo al arrancar la calibración
# (el usuario todavía terminando de hablar, un clic, etc.) infla el RMS de toda la ventana y el
# umbral de silencio queda desproporcionado (llegó a calibrar en 17752, casi la mitad del rango de
# int16) — sub-chunks de ~0.3s siguen acotando cuánto puede pesar un solo pico aislado sobre el
# estadístico final, con el doble de resolución para el percentil de abajo.
NOISE_FLOOR_PERCENTILE = 75.0  # percentil (no la mediana) del RMS por sub-chunk usado
# para calibrar. Antes se usaba `np.median` (percentil 50): correcto contra UN sub-chunk aislado
# ruidoso (un clic, ver arriba), pero es exactamente el estadístico equivocado contra ruido
# "bursty" con más de un sub-chunk quieto que ruidoso en la ventana — la mediana termina
# reportando "el momento típico" (a menudo un hueco de silencio entre efectos), no "el nivel típico
# cuando el ambiente realmente está sonando", que es lo que hace falta para no mezclar audio de
# juego con habla real el resto de la sesión. Percentil 75 (no 50) sigue ignorando un outlier
# aislado hacia arriba (con `NOISE_FLOOR_SUBCHUNKS=10`, un solo sub-chunk ruidoso entre 9 quietos
# queda muy por debajo del percentil 75 — mismo caso cubierto arriba), pero ante una mezcla real de
# sub-chunks quietos/ruidosos (el caso "bursty" que motiva este cambio) el resultado se inclina
# hacia los momentos más fuertes de la ventana en vez de ignorarlos como si fueran outliers. No se
# eligió un percentil más alto (90/95) a propósito: correría el riesgo inverso, que un pico real
# aislado (no representativo del nivel típico "fuerte") termine dominando el piso calibrado —
# 75 es el punto medio razonado entre "ignorar por completo los momentos fuertes" (mediana) y
# "que un solo pico los defina" (percentil muy alto/máximo), sin datos en vivo de sesiones bursty
# reales para afinarlo más — revisar si en uso real todavía se ven picos aislados de calibración
# desproporcionados.
NOISE_FLOOR_MULTIPLIER = (
    4.0  # el umbral de silencio/voz se calibra a piso_de_ruido * esto
)
MIN_SILENCE_RMS_THRESHOLD = (
    150.0  # piso absoluto — nunca calibrar más sensible que esto
)
# Antes en 40.0 — demostrado insuficiente en vivo (ver `NOISE_FLOOR_SAMPLE_SECONDS`): con un piso
# de ruido mal medido de 18.5, el viejo mínimo de 40 ni siquiera llegaba a activarse (18.5 * 4.0 =
# 74.0 > 40), así que el umbral calibrado terminó en 74.1 — muy por debajo de lo que hace falta
# para no confundir audio de juego/música con habla real (la misma sesión, con una medición de
# suerte, había calibrado en 4621.7 minutos antes). 150.0 es una decisión razonada, no verificada
# en vivo contra el escenario exacto "sala en silencio, sin juego ni música" de este usuario (no se
# pudo forzar ese escenario en el momento de este fix): en esta misma sesión, el ambiente sin habla
# real medía RMS ~0.3-20 (silencio digital real, ver `chunk rms=0.3`/`rms=18.5` en
# `jarvis-error.log`) y la voz real del usuario, ya con el mic array del laptop (lejos de la boca,
# no un headset), midió mayormente RMS 300-9000 con picos frecuentes en el rango de miles — 150.0
# queda muy por encima del piso de silencio digital real observado (margen de ~7x-500x) y bastante
# por debajo del rango típico de habla real observado, así que no debería recortar habla suave
# genuina, pero es sustancialmente más alto que el 40.0 anterior, que la propia matemática de este
# incidente mostró que no alcanza como piso de seguridad. Marcar para revisión si en uso real
# (sala realmente silenciosa, sin juego/música) este piso resulta ser sistemáticamente demasiado
# alto o demasiado bajo — no se pudo verificar en vivo contra ese escenario específico esta noche.
# Resta espectral (ver `reduce_background_noise`): tamaño de ventana/hop en samples a 16kHz
# (64ms/16ms, 75% overlap — estándar para voz) y cuánto perfil de ruido restar de cada frame.
NOISE_REDUCTION_FRAME_SAMPLES = 1024
NOISE_REDUCTION_HOP_SAMPLES = 256
NOISE_REDUCTION_OVERSUBTRACTION = (
    2.0  # cuánto de más restar sobre el perfil medido — Boll 1979
)
NOISE_REDUCTION_SPECTRAL_FLOOR = (
    0.05  # nunca bajar una frecuencia de este % de su magnitud
)
# original — restar el 100% del ruido estimado genera "musical noise" (artefactos tipo silbido),
# el piso evita eso a costa de dejar pasar algo de ruido residual.
PRE_ROLL_SECONDS = (
    0.5  # cuánto audio previo a la wake word se guarda y se pega al comando
)
# Confirmado en vivo: al detectarse la wake word se cierra el stream de escucha y se abre uno
# nuevo para grabar el comando — ese cambio de stream tarda una fracción de segundo, y si se
# habla pegado a "Hey Jarvis" sin pausa, esos primeros milisegundos del comando se pierden
# (transcripciones truncadas tipo "Xim, ma, s, i, m" en vez de la frase completa). Guardar un
# colchón de audio previo evita depender de que el usuario pause justo ahí.
PRE_ROLL_FRAMES = max(1, round(PRE_ROLL_SECONDS * SAMPLE_RATE / FRAME_SAMPLES))
TOOL_CALL_ACK_PHRASE = (
    "Dale, dejame revisar eso."  # se dice antes de ejecutar un tool (clima,
)
# búsqueda web) porque la vuelta tarda unos segundos — sin este acuse quedaba en silencio y se
# sentía como que JARVIS se había colgado (pedido explícito del usuario, confirmado en vivo).
# `SYSTEM_PROMPT` y los `_*_FRAMING_HEADER` de acá abajo se cargan desde archivos `.md` en
# `jarvis/audio/prompts/` en vez de vivir como literales de Python (investigado con fuentes
# reales: es el patrón que usan proyectos de producción como Aider u OpenHands para un system
# prompt fijo de un solo desarrollador — mantenibilidad real, sin necesitar un motor de templates
# como Jinja2/LangChain, que sería sobre-ingeniería para un solo prompt sin variantes). Carga con
# `Path.read_text` (stdlib) a nivel de módulo, no lazy — mismo momento de carga y mismo valor
# final que antes; la única diferencia observable es de dónde sale el texto. Los placeholders que
# antes se interpolaban vía f-string (`{WEB_DATA_OPEN_TAG}`, `{MEMORY_DATA_OPEN_TAG}`, etc.) están
# escritos como su valor literal fijo directamente en los `.md` (ej. `<web_data>`), porque esas
# constantes son strings fijos sin ninguna lógica dinámica — no hay interpolación real que
# preservar sobre un archivo de texto plano. Toda la inyección dinámica de datos en tiempo de
# ejecución (fecha/hora, hechos recordados, historial, resultados de búsqueda) sigue viviendo en
# Python, sin tocar (`_build_system_prompt`, más abajo).
_PROMPTS_DIR = Path(__file__).parent / "prompts"
# `WEB_DATA_OPEN_TAG`/`WEB_DATA_CLOSE_TAG`/`RECALLED_MEMORY_OPEN_TAG`/`RECALLED_MEMORY_CLOSE_TAG`
# (importados arriba) ya no se interpolan en código acá — el valor literal de esos tags quedó
# escrito directo en `prompts/system_prompt.md` (ver comentario sobre `_PROMPTS_DIR`). Siguen
# importados solo como re-export intencional: tests existentes los referencian vía
# `pipeline.WEB_DATA_OPEN_TAG`/`pipeline.RECALLED_MEMORY_OPEN_TAG` para verificar que la etiqueta
# real aparece en `SYSTEM_PROMPT` sin tener que importar de `jarvis.tools.search`/
# `jarvis.tools.recall_memory` por separado. Esta línea es la única referencia real que les queda
# en este módulo — sin ella, `ruff` (F401) los marcaría como imports sin usar.
_REEXPORTED_THIRD_PARTY_DATA_TAGS = (
    WEB_DATA_OPEN_TAG,
    WEB_DATA_CLOSE_TAG,
    RECALLED_MEMORY_OPEN_TAG,
    RECALLED_MEMORY_CLOSE_TAG,
)
MEMORY_DATA_OPEN_TAG = "<remembered_facts>"
MEMORY_DATA_CLOSE_TAG = "</remembered_facts>"
# Framing de los hechos recordados en el system prompt (`_build_system_prompt`) — mismo principio
# que `_WEB_DATA_FRAMING_HEADER` de `jarvis.tools.search`: datos reportados, no instrucciones.
# Existe porque un hecho guardado con `remember_fact` puede, en la práctica, originarse en
# contenido de `<web_data>` que el LLM decidió "recordar" dentro del mismo turno — para cuando se
# reinyecta en un turno futuro ya no conserva esa marca de origen, así que el framing tiene que
# aplicarse siempre en el punto de reinyección, no solo confiar en que nunca se guarde nada
# no confiable (hallazgo HIGH de `security-reviewer`, ver docstring del módulo).
_MEMORY_FRAMING_HEADER = (
    (_PROMPTS_DIR / "memory_framing.md").read_text(encoding="utf-8").strip()
)
SPEECH_STYLE_OPEN_TAG = "<speech_style_examples>"
SPEECH_STYLE_CLOSE_TAG = "</speech_style_examples>"
# Framing de las muestras de habla en el system prompt (`_build_system_prompt`) — distinto al de
# `_MEMORY_FRAMING_HEADER` a propósito: no son datos a considerar, son ejemplos de REGISTRO a
# imitar en la respuesta propia del LLM. La instrucción explícita de "no las repitas literalmente"
# existe porque, sin ella, es fácil que el modelo confunda "acá tenés cómo habla el usuario" con
# "el usuario dijo esto ahora" y responda citando o reaccionando a una frase vieja fuera de
# contexto en vez de simplemente adoptar su tono.
_SPEECH_STYLE_FRAMING_HEADER = (
    (_PROMPTS_DIR / "speech_style_framing.md").read_text(encoding="utf-8").strip()
)
CONVERSATION_HISTORY_OPEN_TAG = "<conversation_history>"
CONVERSATION_HISTORY_CLOSE_TAG = "</conversation_history>"
# Framing del historial de turnos pasados en el system prompt (`_build_system_prompt`) — pedido
# explícito del usuario ("que se acuerde que dije antes"). Distinto de `_MEMORY_FRAMING_HEADER`
# (hechos curados) y de `_SPEECH_STYLE_FRAMING_HEADER` (ejemplos de estilo): esto es contexto de
# CONTINUIDAD conversacional, ni una orden a ejecutar ahora ni un ejemplo de tono a imitar. La
# instrucción "nunca como un comando nuevo" cubre tanto los turnos de usuario pasados (que son, a
# lo sumo, pedidos ya cumplidos en su momento, no algo a repetir en el turno actual) como los de
# JARVIS (ver docstring del módulo: una respuesta pasada puede haber citado contenido de
# `<web_data>` sin conservar esa marca de origen, mismo mecanismo que motivó el hallazgo HIGH
# sobre `remembered_facts`).
_CONVERSATION_HISTORY_FRAMING_HEADER = (
    (_PROMPTS_DIR / "conversation_history_framing.md")
    .read_text(encoding="utf-8")
    .strip()
)
RECENT_ACTIONS_OPEN_TAG = "<recent_actions>"
RECENT_ACTIONS_CLOSE_TAG = "</recent_actions>"
# Framing de las últimas acciones ejecutadas, por tool, en el system prompt
# (`_build_system_prompt`) — ver docstring del módulo para el bug real que motiva esto ("la última
# canción", "el mismo modo de juego"). Igual que `_MEMORY_FRAMING_HEADER`/
# `_CONVERSATION_HISTORY_FRAMING_HEADER`: dato reportado, nunca una instrucción a seguir, aunque
# el contenido parezca decir qué hacer.
_RECENT_ACTIONS_FRAMING_HEADER = (
    (_PROMPTS_DIR / "recent_actions_framing.md").read_text(encoding="utf-8").strip()
)
SYSTEM_PROMPT = (_PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")
MAX_TOOL_CALLS_PER_TURN = (
    5  # tope duro: corta un turno si el LLM insiste en pedir tools
)
# Antes en 3 — confirmado en vivo que se quedaba corto: "reproducí tal canción" necesita
# search_web (encontrar el video) + open_url (abrirlo) + una llamada más para la respuesta
# final = 3 llamadas al LLM justo en el límite, así que cualquier mínima ineficiencia (el modelo
# reintentando, pidiendo un tool de más) agotaba el tope y devolvía "No pude completar la
# solicitud" en vez de terminar el flujo. 5 da margen real sin volverse un loop sin fin.
_AFFIRMATIVE_WORDS = frozenset(
    {
        "si",
        "sí",
        "dale",
        "confirmo",
        "confirmado",
        "afirmativo",
        "ok",
        "okay",
        "correcto",
        "adelante",
        "hazlo",
        "hacelo",
        # Palabras agregadas para frases naturales completas tipo "dale, dale ya" o "sí, va,
        # confirmado" — capa 2 sigue exigiendo que TODA la frase esté compuesta solo por estas
        # palabras (nada cambia en la capa 1, el veto por negación, que sigue siendo la garantía
        # real de seguridad de ADR-0004). Ninguna de estas es ambigua por sí sola hacia una
        # negación — a diferencia de "bueno" (dejada afuera a propósito, ver test existente:
        # puede ser una duda/hesitación, no una afirmación clara).
        "ya",
        "va",
        "claro",
        "porfa",
        "favor",
        "por",
    }
)
# Cualquiera de estas palabras en la respuesta deniega, sin importar qué otra palabra afirmativa
# aparezca junto a ella — ver `_is_affirmative`. Existe para el caso "no, dale un momento,
# dejame pensar": contiene "dale" (afirmativa) pero es semánticamente una negativa/ambigüedad.
_NEGATIVE_WORDS = frozenset(
    {
        "no",
        "nunca",
        "jamás",
        "negativo",
        "cancelá",
        "cancela",
        "cancelo",
        "cancelar",
        "pará",
        "espera",
        "esperá",
        "todavía",
        "todavia",
    }
)


def tee_frames(
    frames: Iterator[np.ndarray], buffer: deque[np.ndarray]
) -> Iterator[np.ndarray]:
    """Devolver cada frame tal cual, guardando además una copia en `buffer` (deque de tamaño
    fijo) — así se puede recuperar el audio justo antes de una detección sin tocar `detect()`.
    """
    for frame in frames:
        buffer.append(frame)
        yield frame


def measure_noise_floor(
    *, device: int | None, sample_seconds: float = NOISE_FLOOR_SAMPLE_SECONDS
) -> tuple[float, np.ndarray]:
    """Medir el RMS del ambiente al arrancar, para calibrar los umbrales de silencio/voz contra
    las condiciones reales del momento (distancia al mic, ruido de fondo) en vez de un número
    fijo para siempre. Asume que en esos primeros segundos nadie está hablando todavía — es el
    enfoque de "umbral adaptativo" que usan los sistemas de producción, frente al umbral
    estático que veníamos usando (confirmado en vivo: la señal real varía muchísimo según
    dónde esté el usuario, no solo según su voz).

    El estadístico usado sobre los sub-chunks es `NOISE_FLOOR_PERCENTILE` (percentil 75, no la
    mediana) — ver el comentario de esa constante para el bug real que motiva el cambio: ante
    ruido "bursty" (audio de juego/música, silencio entre efectos y fuerte durante ellos), la
    mediana de una ventana corta puede caer entera en un hueco de silencio y calibrar un umbral
    de voz muchísimo más sensible de lo que el ambiente real amerita, sin ningún cambio real de
    condiciones entre una corrida y la siguiente (confirmado en vivo, 62x de diferencia entre dos
    reinicios consecutivos del mismo proceso).

    Devuelve `(rms, muestra_resampleada_a_16khz)` — antes solo devolvía el RMS y descartaba el
    audio. Ese mismo audio de "ambiente sin hablar" es exactamente el perfil de ruido que
    necesita `reduce_background_noise` para restarlo del audio de un comando real: sin esto
    habría que grabar una muestra de ruido aparte, cuando ya la estábamos grabando y tirando.
    """
    resolved_device = resolve_input_device(device)
    device_sr = input_sample_rate(resolved_device)
    samples = int(sample_seconds * device_sr)
    audio: np.ndarray = sd.rec(
        samples, samplerate=device_sr, channels=1, dtype="int16", device=resolved_device
    )
    sd.wait()
    resampled = resample(
        np.asarray(audio.reshape(-1)), orig_sr=device_sr, target_sr=SAMPLE_RATE
    )
    subchunks = np.array_split(resampled, NOISE_FLOOR_SUBCHUNKS)
    subchunk_rms_values = [chunk_rms(chunk) for chunk in subchunks]
    return (
        float(np.percentile(subchunk_rms_values, NOISE_FLOOR_PERCENTILE)),
        resampled,
    )


def calibrate_thresholds(noise_floor: float) -> float:
    """Calcular el umbral de silencio/voz a partir del piso de ruido medido: un múltiplo de
    ese piso, con un mínimo absoluto (`MIN_SILENCE_RMS_THRESHOLD`) para no volverse tan
    sensible que cualquier cosa cuente como "habla" si el ambiente está anormalmente callado.
    """
    return max(MIN_SILENCE_RMS_THRESHOLD, noise_floor * NOISE_FLOOR_MULTIPLIER)


def normalize_gain(audio: np.ndarray, *, min_peak: float) -> np.ndarray:
    """AGC simple: si el audio vino bajito pero con señal real, lo sube a ~90% del rango de
    int16 antes de mandarlo a transcribir. Nunca baja volumen ya fuerte (evita clipping doble),
    y no toca audio por debajo de `min_peak` — eso es piso de ruido calibrado, no habla;
    amplificarlo solo fabricaría "señal" falsa para Whisper, que ya sabemos que alucina sobre
    silencio (fase 2).
    """
    peak = int(np.abs(audio).max())
    max_int16 = 32767
    target = int(AGC_TARGET_PEAK_RATIO * max_int16)
    if peak < min_peak or peak >= target:
        return audio
    gain = target / peak
    boosted = audio.astype(np.float64) * gain
    return np.clip(boosted, -max_int16, max_int16).astype(np.int16)


def _spectral_noise_profile(
    noise_sample: np.ndarray, *, frame_samples: int, hop_samples: int
) -> np.ndarray:
    """Perfil de magnitud espectral del ruido de fondo: la mediana (no el promedio — mismo
    motivo que `measure_noise_floor` usa mediana para el RMS) de la magnitud FFT sobre ventanas
    superpuestas de `noise_sample`."""
    window = np.hanning(frame_samples)
    magnitudes = [
        np.abs(np.fft.rfft(noise_sample[start : start + frame_samples] * window))
        for start in range(0, len(noise_sample) - frame_samples + 1, hop_samples)
    ]
    if not magnitudes:
        return np.zeros(frame_samples // 2 + 1)
    return np.median(magnitudes, axis=0)


def reduce_background_noise(
    audio: np.ndarray,
    *,
    noise_sample: np.ndarray,
    frame_samples: int = NOISE_REDUCTION_FRAME_SAMPLES,
    hop_samples: int = NOISE_REDUCTION_HOP_SAMPLES,
) -> np.ndarray:
    """Restar el perfil espectral de `noise_sample` (el "silencio" medido por
    `measure_noise_floor` al arrancar) del audio grabado, vía resta espectral clásica
    (Boll, 1979) — atenúa ruido de fondo estacionario (ventilador, ruido de sala, el zumbido
    propio del mic) sin tocar la voz, que domina en las frecuencias donde el ruido es débil.

    Por qué esto y no `normalize_gain`: `normalize_gain` solo sube el volumen general (un AGC de
    pico) — si el ruido de fondo y la voz suben juntos, el *contraste* entre ambos (relación
    señal/ruido) no mejora en nada, que es lo que de verdad determina si Whisper entiende bien.
    Confirmado con los devices reales de esta máquina (`sounddevice.query_devices()`): el mic
    array del laptop graba a 48kHz vía WASAPI (mejor "calidad" nominal que los auriculares
    Bluetooth, que caen a 8kHz por el perfil Hands-Free), pero está lejos de la boca — el ruido
    de sala compite mucho más con la voz que en un mic de auriculares pegado a la boca. Esta
    función ataca ese problema real (relación señal/ruido), no el nivel.

    Sin dependencia nueva: `numpy.fft` (numpy ya es dependencia dura) alcanza para esto — mismo
    criterio que ya usa `resample.py` para no sumar `scipy`. Se evaluó `noisereduce` (la librería
    estándar para esto) y se descartó: arrastra `matplotlib` como dependencia transitiva dura
    (~10MB + varias más), bloat injustificado para un asistente sin interfaz gráfica
    (`.claude/rules/python.md`: "sin dependencias nuevas sin justificar la necesidad").
    """
    if len(audio) < frame_samples:
        return audio
    noise_profile = _spectral_noise_profile(
        noise_sample, frame_samples=frame_samples, hop_samples=hop_samples
    )
    window = np.hanning(frame_samples)
    audio_f64 = audio.astype(np.float64)
    output = np.zeros(len(audio), dtype=np.float64)
    window_sum = np.zeros(len(audio), dtype=np.float64)
    for start in range(0, len(audio) - frame_samples + 1, hop_samples):
        frame = audio_f64[start : start + frame_samples] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        cleaned_magnitude = np.maximum(
            magnitude - NOISE_REDUCTION_OVERSUBTRACTION * noise_profile,
            NOISE_REDUCTION_SPECTRAL_FLOOR * magnitude,
        )
        cleaned_frame = np.fft.irfft(
            cleaned_magnitude * np.exp(1j * phase), n=frame_samples
        )
        output[start : start + frame_samples] += cleaned_frame
        window_sum[start : start + frame_samples] += window
    # Overlap-add: normalizar por la suma de ventanas donde hubo al menos un frame procesado (no
    # `window**2` — la ventana ya se aplicó una sola vez, del lado del análisis, así que
    # `cleaned_frame` ya trae un factor de `window` adentro; dividir por `window**2` sobre-
    # normaliza cerca de los bordes de cada frame, donde la ventana es chica pero no cero, y
    # dispara valores absurdos — confirmado con un test real que clipeaba a ±32767 en vez de
    # atenuar). El resto (la cola que no llegó a completar un frame entero) queda con el audio
    # original en vez de silencio — sin este `where`, esos samples finales quedarían en 0.0
    # (mudos).
    covered = window_sum > 1e-8
    reconstructed = np.where(
        covered, output / np.where(covered, window_sum, 1.0), audio_f64
    )
    max_int16 = 32767
    return np.clip(reconstructed, -max_int16, max_int16).astype(np.int16)


def record_command(
    *,
    device: int | None,
    silence_threshold: float,
    max_duration: float = COMMAND_WINDOW_SECONDS,
    pre_roll: np.ndarray | None = None,
    system_audio: SystemAudioGate | None = None,
    noise_sample: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Grabar el comando dicho después de la wake word, cortando solo tras un silencio
    sostenido en vez de esperar siempre `max_duration` completos.

    Devuelve `(audio, speech_detected)` — `speech_detected` es `True` solo si algún chunk
    cruzó `silence_threshold` durante la grabación. Antes esto no se exponía, y el llamador
    mandaba el audio a transcribir sin importar si hubo habla real o no. Confirmado en vivo,
    revisando `data/jarvis.log`: eso disparaba una cascada de transcripciones alucinadas sobre
    silencio/ruido ambiente puro — el modelo de transcripción no confiablemente devuelve vacío
    ante silencio, a veces inventa texto en idiomas al azar (chino, árabe, japonés, tibetano,
    todo visto en una sola sesión). Con `speech_detected`, el llamador puede saltear la llamada
    a `transcribe()` directamente cuando no hubo habla real, cortando la alucinación de raíz en
    vez de intentar filtrarla después de generada.

    `silence_threshold` viene de `calibrate_thresholds()` — calibrado contra el ambiente real
    de esta sesión, no un número fijo. Medido en vivo: la mayoría de los comandos duran 1-2s,
    no los 4s fijos que se grababan antes.

    `pre_roll`, si se pasa, se pega al principio del audio grabado — es el colchón de audio de
    justo antes de la wake word (ver `PRE_ROLL_FRAMES`), para no perder el arranque del comando
    si se habla pegado a la wake word sin pausa.

    `system_audio`, si se pasa (`SystemAudioGate`, `jarvis.audio.loopback`), gatea qué cuenta
    como habla: mientras el sistema suena fuerte, ningún chunk cuenta como voz sin importar qué
    diga el resto de las señales — así el audio de un juego o música de fondo filtrándose al mic
    no cuenta como "el usuario está hablando" (no reemplaza cancelación de eco real, ver
    docstring de `loopback.py`). Sin `system_audio` (default `None`), idéntico al comportamiento
    de antes de este parámetro.

    La decisión de "¿es voz?" ya no es solo RMS — usa `jarvis.audio.speech_detector` (Silero VAD)
    si el modelo carga bien, degradando a solo-RMS si no (ver `vad.is_speech_chunk`). Se construye
    un detector NUEVO por cada llamada a `record_command` (no uno compartido entre comandos): el
    modelo tiene estado interno entre ventanas de audio y no expone ningún reset, así que la
    forma segura de que el contexto de un comando no se filtre al siguiente es una instancia
    fresca por sesión.

    `noise_sample`, si se pasa, se usa para restar el perfil de ruido de fondo del audio grabado
    antes del AGC (`reduce_background_noise` — ver su docstring: es lo que de verdad mejora el
    reconocimiento con un mic lejano/ruidoso como el array del laptop, a diferencia de
    `normalize_gain`, que solo sube volumen). Se salta si no hubo habla real (`speech_started`
    sigue `False`) — no tiene sentido limpiar ruido puro. Sin `noise_sample` (default `None`),
    idéntico al comportamiento de antes de este parámetro.
    """
    resolved_device = resolve_input_device(device)
    device_sr = input_sample_rate(resolved_device)
    chunk_samples = int(CHUNK_SECONDS * device_sr)
    chunks: list[np.ndarray] = []
    speech_started = False
    silence_run = 0.0
    elapsed = 0.0
    detector = load_speech_detector()
    with sd.InputStream(
        samplerate=device_sr,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        device=resolved_device,
    ) as stream:
        while True:
            chunk, _overflowed = stream.read(chunk_samples)
            chunk = resample(
                chunk.reshape(-1), orig_sr=device_sr, target_sr=SAMPLE_RATE
            )
            chunks.append(chunk)
            elapsed += CHUNK_SECONDS
            system_is_loud = system_audio is not None and system_audio.is_loud()
            speech_probability = (
                detector.speech_probability(chunk) if detector is not None else None
            )
            if is_speech_chunk(
                chunk_rms(chunk),
                min_rms_floor=silence_threshold,
                speech_probability=speech_probability,
                speech_probability_threshold=SPEECH_PROBABILITY_THRESHOLD,
                system_is_loud=system_is_loud,
            ):
                speech_started = True
                silence_run = 0.0
            elif speech_started:
                silence_run += CHUNK_SECONDS
            if should_stop_recording(
                speech_started=speech_started,
                silence_run_seconds=silence_run,
                elapsed_seconds=elapsed,
                max_seconds=max_duration,
            ):
                break
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    if pre_roll is not None and len(pre_roll) > 0:
        audio = np.concatenate([pre_roll, audio])
    if noise_sample is not None and speech_started and len(audio) > 0:
        audio = reduce_background_noise(audio, noise_sample=noise_sample)
    return normalize_gain(audio, min_peak=silence_threshold), speech_started


# Tope de largo para una respuesta hablada real de este asistente — "corto y directo" es una
# instrucción explícita de SYSTEM_PROMPT, ninguna respuesta legítima llega ni cerca de esto.
# Red de seguridad de CÓDIGO, no solo de prompt: bug real, en vivo — el LLM (`deepseek-chat`, sin
# el campo `reasoning_content` separado que sí tiene `deepseek-reasoner`, ver
# `jarvis.llm.client._log_reasoning_content`) a veces ignora la instrucción explícita de "nunca
# antepongas tu razonamiento interno" y lo mete adentro de `message.content` igual, sobre todo en
# casos ambiguos (ej. "¿'Node' es una deformación de qué canción de Billie Eilish?") — varios
# párrafos de deliberación en primera persona ("Hmm, considerando que...", "Debo preguntar para
# aclarar...") se leyeron en voz alta tal cual antes de este fix.
MAX_SPOKEN_RESPONSE_CHARS = 300

_LEAKED_REASONING_FALLBACK = "No te entendí bien, Daniel. ¿Podés repetirlo?"


def _sanitize_final_response(text: str) -> str:
    """Si `text` (la respuesta final del LLM, sin tool-call) parece razonamiento interno
    filtrado en vez de la frase corta que pide `SYSTEM_PROMPT`, se descarta y se reemplaza por
    una pregunta de aclaración genérica — nunca se lee en voz alta un párrafo de deliberación
    completo. Heurística deliberadamente simple (solo largo, ver `MAX_SPOKEN_RESPONSE_CHARS`):
    enumerar frases de "esto suena a razonamiento" es una lista sin fin; una respuesta hablada
    real de este asistente nunca es tan larga, sin importar de qué frase esté hecha.
    """
    if len(text) > MAX_SPOKEN_RESPONSE_CHARS:
        print(
            f"(respuesta del LLM sospechosamente larga, {len(text)} caracteres — probable "
            f"razonamiento interno filtrado, se descarta: {text!r})",
            file=sys.stderr,
        )
        return _LEAKED_REASONING_FALLBACK
    return text


def _is_affirmative(text: str) -> bool:
    """Contrato de ADR-0004 ("silencio o timeout ⇒ denegar por defecto"): denegar es el default
    en cualquier caso ambiguo, no solo en silencio puro.

    Hallazgo de `security-reviewer` sobre la primera versión de esta función: hacer
    "¿alguna palabra afirmativa aparece en la transcripción?" da falsos positivos sobre
    negaciones como "no, dale un momento, dejame pensar" (contiene "dale"). Dos capas, ambas
    tienen que dar `True` para aprobar:

    1. Si aparece cualquier palabra de `_NEGATIVE_WORDS` en la respuesta, se deniega sin mirar
       nada más — una negación en la misma frase que una afirmación sigue siendo una negación.
    2. La respuesta completa (normalizada y tokenizada) tiene que consistir *enteramente* en
       palabras de `_AFFIRMATIVE_WORDS` — no alcanza con que una esté presente en medio de una
       oración más larga. Esto es intencionalmente estricto: cualquier respuesta ambigua, larga
       o no reconocida deniega en vez de aprobar.
    """
    normalized = text.strip().lower()
    if not normalized:
        return False
    words = [word.strip(".,!¡¿?;:") for word in normalized.split()]
    words = [word for word in words if word]
    if not words:
        return False
    if any(word in _NEGATIVE_WORDS for word in words):
        return False
    return all(word in _AFFIRMATIVE_WORDS for word in words)


# Pedido explícito del usuario: poder decirle a JARVIS que "se vaya"/"descanse" y que dejе de
# responder hasta que le digas que "vuelva". JARVIS nunca deja de escuchar el micrófono de
# verdad (si lo hiciera, no habría forma de que "Jarvis, volvé" lo reactive) — mientras está
# "dormido" simplemente ignora cualquier comando que no sea la frase para despertarlo, ver `run()`.
_SLEEP_WORDS = frozenset(
    {
        "andate",
        "vete",
        "retirate",
        "retírate",
        "descansa",
        "descansá",
        "descansar",  # confirmado en vivo: "ya puede descansar" no matcheaba, solo el imperativo
        "descanso",
        "dormite",
        "dormi",
        "dormí",
        "dormir",
        "duerme",
        "silencio",
        # Bug real, en vivo: "Alexa, desconéctate" no matcheaba ninguna de las anteriores, caía
        # a `dispatch_turn` sin ninguna tool de "apagar/desconectar" — el LLM se inventaba un
        # ritual de confirmación ("decime adiós") que además no llevaba a ningún lado ni siquiera
        # diciendo "adiós", porque esa palabra tampoco estaba reconocida acá. Se agregan como
        # sinónimos del mismo modo dormir ya existente (sin confirmación, mismo criterio que
        # "descansá"): pedir que JARVIS deje de escuchar es exactamente lo mismo pedido con otras
        # palabras, no una acción distinta que necesite pasar por el LLM ni por PolicyEngine.
        "desconecta",
        "desconéctate",
        "desconectate",
        "apaga",
        "apágate",
        "apagate",
        "callate",
        "cállate",
        # "adiós"/"chau" quedan afuera a propósito: son despedidas comunes en conversación
        # normal (ej. despidiéndose de otra persona, no de JARVIS) — `_contains_any_word` matchea
        # con que la palabra aparezca en cualquier parte de la frase, así que agregarlas
        # dispararía el modo dormir en falsos positivos mucho más seguido que los imperativos de
        # arriba, que son inequívocamente un comando dirigido a JARVIS.
    }
)
_WAKE_WORDS = frozenset(
    {
        "volve",
        "volvé",
        "vuelve",
        "volver",
        "regresa",
        "regresá",
        "regresar",
        "desperta",
        "despertá",
        "despierta",
        "despertar",
        # Bug real, en vivo: dormido, el usuario dijo "Alexa, cierra Discord" — la wake word
        # acústica (`jarvis.audio.wake_word.WAKEWORD_NAMES`: "alexa"/"hey_jarvis"/"hey_mycroft")
        # ya había disparado la detección exterior y grabado el comando, pero acá adentro
        # ninguna de las palabras de arriba matcheaba, así que JARVIS se quedó dormido en
        # silencio, sin ejecutar "cerrá Discord" NI avisar que seguía dormido. Decir el nombre
        # de JARVIS es, para cualquier usuario, una forma obvia de "quiero tu atención" — tan
        # válida como "volvé"/"despertá" para despertarlo (aunque, igual que con esas, el resto
        # del pedido en la misma frase no se ejecuta automáticamente; el usuario tiene que
        # repetirlo ya despierto, mismo comportamiento que ya existía para "volvé").
        "alexa",
        "jarvis",
        "mycroft",
    }
)


LOL_AUTO_ACCEPT_ENV_VAR = "JARVIS_LOL_AUTO_ACCEPT"
# Valores que cuentan como "apagado" — comparación case-insensitive contra el valor crudo de la
# variable de entorno, ver `_lol_auto_accept_enabled`.
_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def _lol_auto_accept_enabled() -> bool:
    """Opt-out de `LCUAutoAcceptMonitor` (`jarvis.league.lcu_monitor`) vía la variable de entorno
    `JARVIS_LOL_AUTO_ACCEPT`, leída una vez por corrida de `run()`.

    A diferencia de `SystemAudioMonitor` (servicio de fondo de solo lectura, sin forma de
    "deshacer" nada), este servicio toma una acción real y difícil de deshacer: aceptar un
    ready-check compromete al usuario a la partida — declinarla después arriesga una penalización
    de LeaverBuster/dodge en el juego real (hallazgo LOW de `security-reviewer`, revisión previa
    al commit de esta feature). Habilitado por default (`True` si la variable no está seteada):
    el usuario pidió explícitamente esta feature y ya está funcionando, así que apagarla requiere
    una acción explícita, no al revés. Seteando `JARVIS_LOL_AUTO_ACCEPT=false` (o `0`/`no`/`off`,
    case-insensitive) en el entorno o en `.env` (cargado por `jarvis.config.load_dotenv`, ya
    invocado al principio de `run()`) la desactiva sin tocar código.
    """
    raw_value = os.environ.get(LOL_AUTO_ACCEPT_ENV_VAR)
    if raw_value is None:
        return True
    return raw_value.strip().lower() not in _FALSY_ENV_VALUES


def _contains_any_word(text: str, trigger_words: frozenset[str]) -> bool:
    """`True` si alguna palabra de `text` (normalizada: minúsculas, sin puntuación) está en
    `trigger_words` — a diferencia de `_is_affirmative`, acá alcanza con que aparezca en
    cualquier parte de la frase (ej. "Jarvis, andate a descansar"), no que sea la frase entera:
    esto no es un gate de seguridad tipo CONFIRM, es solo un modo de "no me molestes"."""
    normalized = text.strip().lower()
    words = [word.strip(".,!¡¿?;:") for word in normalized.split()]
    return any(word in trigger_words for word in words if word)


class VoiceConfirmationChannel:
    """Implementa `ConfirmationChannel` (`jarvis.security.policy`, ADR-0005) reusando el TTS/STT
    que ya vive en este módulo: pregunta por voz con `tts.speak`, graba la respuesta con el
    mismo `record_command` que graba comandos (mismo tope duro de duración — ese es el
    "timeout" del contrato), y la transcribe con el mismo cliente STT. No introduce un canal de
    confirmación nuevo.
    """

    def __init__(
        self,
        *,
        tts: TTSClient,
        stt_client: OpenAI,
        device: int | None,
        silence_threshold: float,
        system_audio: SystemAudioGate | None = None,
        noise_sample: np.ndarray | None = None,
    ) -> None:
        self._tts = tts
        self._stt_client = stt_client
        self._device = device
        self._silence_threshold = silence_threshold
        self._system_audio = system_audio
        self._noise_sample = noise_sample

    async def ask(self, prompt: str) -> bool:
        # `tts.speak`/`record_command`/`transcribe` son bloqueantes (I/O de audio real) — se
        # corren en un thread aparte para no bloquear el loop de asyncio que orquesta el
        # tool-call (`.claude/rules/python.md`: no mezclar código bloqueante en rutas async sin
        # `asyncio.to_thread`).
        await asyncio.to_thread(self._tts.speak, prompt)
        audio, speech_detected = await asyncio.to_thread(
            record_command,
            device=self._device,
            silence_threshold=self._silence_threshold,
            system_audio=self._system_audio,
            noise_sample=self._noise_sample,
        )
        if not speech_detected:
            # Mismo fix que en run(): sin habla real detectada, no tiene sentido transcribir
            # (y evita que una alucinación del STT sobre silencio se interprete como un "sí").
            # Coincide además con el contrato de ADR-0004: silencio ⇒ denegar por defecto.
            print(
                "(confirmación) Silencio real, denegando por defecto.", file=sys.stderr
            )
            return False
        text = await asyncio.to_thread(transcribe, audio, client=self._stt_client)
        print(f"(confirmación) Dijiste: {text!r}", file=sys.stderr)
        return _is_affirmative(text)


def _assistant_message_for_tool_call(result: LLMResult) -> dict[str, Any]:
    """Mensaje `role: assistant` a agregar al historial cuando el LLM pidió un tool-call: el
    LLM necesita ver su propio tool-call antes del mensaje `role: tool` que lo responde, para
    que la siguiente llamada a `complete()` tenga contexto completo.
    """
    assert result.tool_call is not None
    return {
        "role": "assistant",
        "content": result.text or None,
        "tool_calls": [
            {
                "id": result.tool_call.id,
                "type": "function",
                "function": {
                    "name": result.tool_call.name,
                    "arguments": json.dumps(result.tool_call.arguments),
                },
            }
        ],
    }


def _escape_untrusted(text: str) -> str:
    """Neutralizar `<`/`>` antes de insertar `text` en el prompt.

    Duplicado deliberado de `jarvis.tools.search._escape_untrusted` (mismo cuerpo, dos líneas) en
    vez de importado: esa función es un detalle interno de `search.py` (prefijo `_`), no un
    contrato compartido entre módulos. Acá cumple el mismo rol: un hecho guardado con
    `remember_fact` podría, en teoría, contener un `{MEMORY_DATA_CLOSE_TAG}` literal (copiado sin
    querer de contenido web, o adversarial a propósito) y "cerrar" el wrapper de datos recordados
    antes de tiempo — escapar antes de armar el bloque lo evita (hallazgo HIGH de
    `security-reviewer`, ver docstring del módulo).
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")


# Rangos Unicode de scripts no latinos que no pueden aparecer en una transcripción real en
# español — ver `_contains_non_latin_script` para el motivo. No exhaustivo (faltan armenio,
# georgiano, etc.), cubre los scripts vistos en vivo en alucinaciones reales de
# `gpt-4o-transcribe` esta noche (árabe) más los candidatos más probables del mismo patrón.
_NON_LATIN_SCRIPT_RANGES = (
    (0x0370, 0x03FF),  # griego
    (0x0400, 0x04FF),  # cirílico
    (0x0590, 0x05FF),  # hebreo
    (0x0600, 0x06FF),  # árabe
    (0x0750, 0x077F),  # árabe suplementario
    (0x0900, 0x097F),  # devanagari
    (0x0E00, 0x0E7F),  # thai
    (
        0x3000,
        0x303F,
    ),  # puntuación/símbolos CJK (ej. "。", el ideographic full stop) — gap
    # real encontrado en vivo revisando `data/jarvis.db`: una alucinación de un solo carácter de
    # este bloque quedó guardada en `conversation_turns` sin que el filtro original (que solo
    # cubría rangos de "letras") la agarrara, porque este bloque es puntuación, no un script de
    # letras — pero es igual de imposible en una transcripción real en español.
    (0x3040, 0x30FF),  # hiragana/katakana
    (0x4E00, 0x9FFF),  # ideogramas CJK
    (0xAC00, 0xD7A3),  # hangul
)


def _contains_non_latin_script(text: str) -> bool:
    """`True` si `text` contiene algún carácter de un script no latino (árabe, cirílico, CJK,
    etc.) — señal inequívoca de alucinación de `jarvis.audio.stt.transcribe()` sobre audio poco
    claro (ruido de fondo, voces superpuestas que sí cruzan el umbral de silencio pero no son el
    usuario), no una transcripción real: `LANGUAGE = "es"` (`jarvis.audio.stt`) fija el idioma
    esperado, y una transcripción legítima en español nunca usa estos scripts — a diferencia de
    palabras sueltas o gramaticalmente rotas en alfabeto latino (ambiguas, esas si pueden ser
    habla real mal transcripta y no se filtran acá). Bug real, en vivo, mismo mic ya calibrado:
    `Dijiste: 'أفهمت؟'` sobre audio de fondo, sin que el usuario dijera nada en árabe — mismo
    espíritu que la mitigación ya existente para silencio puro (`record_command`/`speech_detected`
    en `run()`), un filtro más contra la cascada de alucinación de idioma, esta vez sobre audio
    que sí tenía señal real (solo que no era el usuario)."""
    return any(
        any(start <= ord(ch) <= end for start, end in _NON_LATIN_SCRIPT_RANGES)
        for ch in text
    )


def _format_time_ago(created_at: str, *, now: datetime) -> str:
    """Traducir un `created_at` ISO 8601 (UTC, como los guarda `jarvis.memory.store.save_tool_call`)
    a una frase corta tipo "hace 34 min"/"hace 2h 14min", calculada contra `now` en el momento de
    armar el prompt — no un string pre-formateado guardado en la fila, que quedaría congelado
    desde el instante en que se guardó y le mentiría al LLM sobre cuán reciente es la acción a
    medida que pasa el tiempo real.

    Sobre un `created_at` malformado (no debería poder existir — `save_tool_call` siempre guarda
    `datetime.now(UTC).isoformat()`) devuelve una frase neutra en vez de lanzar, mismo criterio
    defensivo que `list_due_reminders` sobre un `due_at` corrupto.
    """
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return "hace un tiempo"
    elapsed_minutes = max(0, int((now - parsed).total_seconds() // 60))
    if elapsed_minutes < 1:
        return "hace instantes"
    if elapsed_minutes < 60:
        return f"hace {elapsed_minutes} min"
    hours, minutes = divmod(elapsed_minutes, 60)
    if minutes:
        return f"hace {hours}h {minutes}min"
    return f"hace {hours}h"


_SPANISH_WEEKDAYS = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)


def _current_time_line(*, now: datetime | None = None) -> str:
    """Línea de una frase con la fecha/hora local actual, para que el LLM pueda hacer aritmética
    de fecha/hora al armar `seconds_from_now` para `ReminderTool` (`jarvis.tools.reminder`) — sin
    esto, el prompt nunca le dice al modelo "qué hora es ahora", y "recordame a las 5" no tiene
    forma de resolverse a un número de segundos correcto.

    Hora LOCAL del sistema (`datetime.now().astimezone()`, con offset de zona horaria explícito),
    no UTC: el usuario dice "a las 5" pensando en su propio reloj de pared, no en UTC — usar la
    zona horaria configurada en el sistema operativo (la real de esta máquina) es lo correcto acá,
    a diferencia de `jarvis.memory.store`, que persiste todo en UTC internamente (una decisión de
    almacenamiento, ortogonal a qué hora se le muestra al LLM).

    `now` inyectable (no siempre `datetime.now()` hardcodeado adentro) para que
    `tests/audio/test_pipeline.py` pueda verificar el formato exacto sin depender del reloj real
    de la máquina que corre la suite.
    """
    resolved_now = now if now is not None else datetime.now().astimezone()
    weekday = _SPANISH_WEEKDAYS[resolved_now.weekday()]
    return f"Fecha y hora actual: {weekday} {resolved_now.strftime('%Y-%m-%d %H:%M (UTC%z)')}"


def _build_system_prompt(*, memory_db_path: str | Path) -> str:
    """Armar el system prompt de este turno: `SYSTEM_PROMPT` fijo, más la fecha/hora actual
    (`_current_time_line`, siempre presente, no condicional — necesaria para que `ReminderTool`
    calcule bien `seconds_from_now` en cualquier turno, no solo cuando hay memoria guardada), más,
    si corresponde, dos secciones aparte cargadas de `jarvis.memory.store` — hechos guardados y
    muestras de habla — cada una con su propio framing, agregadas en ese orden cuando están
    presentes.

    Hechos (`{MEMORY_DATA_OPEN_TAG}...{MEMORY_DATA_CLOSE_TAG}`): mismo patrón de mitigación que
    `<web_data>` (`jarvis.tools.search`), no solo el mismo espíritu — framing explícito
    (`_MEMORY_FRAMING_HEADER`) más escapado por hecho (`_escape_untrusted`) — un hecho guardado
    puede haberse originado en contenido de terceros (una búsqueda web que el LLM decidió
    "recordar") sin conservar esa marca de origen, así que la mitigación tiene que vivir acá, en
    el punto de reinyección, no solo confiar en que `remember_fact`/`SYSTEM_PROMPT` eviten que eso
    pase en primer lugar (hallazgo HIGH de `security-reviewer`).

    Muestras de habla (`{SPEECH_STYLE_OPEN_TAG}...{SPEECH_STYLE_CLOSE_TAG}`): las más recientes
    (`list_speech_samples`, tope chico a propósito — ver `jarvis.memory.store`), con
    `_SPEECH_STYLE_FRAMING_HEADER` explicando que son ejemplos de registro a imitar, no contenido
    a repetir ni el comando actual. Sin el mismo escapado anti-injection que los hechos: son
    palabras del propio usuario, no contenido de terceros que pudo colarse sin marca de origen
    (ver docstring del módulo).

    Historial de conversación (`{CONVERSATION_HISTORY_OPEN_TAG}...{CONVERSATION_HISTORY_CLOSE_TAG}`):
    los últimos turnos (`list_recent_conversation_turns`, tope chico — ver `jarvis.memory.store`),
    invertidos acá a orden cronológico (más viejo primero, `list_recent_conversation_turns`
    devuelve más reciente primero) para que se lean como una conversación real, no una lista sin
    orden. Cada turno se presenta como "Usuario: ...\nAlexa: ...". Escapado asimétrico entre
    campos (ver docstring del módulo): `user_text` no se escapa (voz directa del usuario, mismo
    criterio que `speech_samples`); `assistant_text` sí (`_escape_untrusted`) — una respuesta
    pasada de JARVIS puede haber citado contenido de `<web_data>` sin conservar esa marca de
    origen, mismo mecanismo que motivó el hallazgo HIGH sobre `remembered_facts`.

    Últimas acciones por tool (`{RECENT_ACTIONS_OPEN_TAG}...{RECENT_ACTIONS_CLOSE_TAG}`):
    `list_most_recent_tool_call_per_tool` — como mucho una fila por nombre de tool distinto, la
    más reciente (ver docstring del módulo). Cada línea: `tool_name → argumentos | hace X`, con el
    "hace X" calculado acá, contra la hora actual, vía `_format_time_ago` (no un string guardado).
    Mismo escapado asimétrico que el resto: los argumentos se escapan (`_escape_untrusted`), el
    nombre del tool no (viene de un `Tool` ya registrado en código, nunca de contenido de
    terceros).

    Sin hechos, muestras, turnos o acciones guardadas (primer uso, o listas vacías), no agrega esas
    secciones — pero la línea de fecha/hora siempre está, a diferencia de esas cuatro.
    """
    prompt = f"{SYSTEM_PROMPT}\n\n{_current_time_line()}"
    facts = list_facts(db_path=memory_db_path)
    if facts:
        facts_block = "\n".join(f"- {_escape_untrusted(fact)}" for fact in facts)
        memory_section = (
            f"{_MEMORY_FRAMING_HEADER}\n{MEMORY_DATA_OPEN_TAG}\n{facts_block}\n"
            f"{MEMORY_DATA_CLOSE_TAG}"
        )
        prompt = f"{prompt}\n\n{memory_section}"
    speech_samples = list_speech_samples(db_path=memory_db_path)
    if speech_samples:
        samples_block = "\n".join(f"- {sample}" for sample in speech_samples)
        style_section = (
            f"{_SPEECH_STYLE_FRAMING_HEADER}\n{SPEECH_STYLE_OPEN_TAG}\n{samples_block}\n"
            f"{SPEECH_STYLE_CLOSE_TAG}"
        )
        prompt = f"{prompt}\n\n{style_section}"
    recent_turns = list(
        reversed(list_recent_conversation_turns(db_path=memory_db_path))
    )
    if recent_turns:
        turns_block = "\n".join(
            f"Usuario: {turn.user_text}\nAlexa: {_escape_untrusted(turn.assistant_text)}"
            for turn in recent_turns
        )
        history_section = (
            f"{_CONVERSATION_HISTORY_FRAMING_HEADER}\n{CONVERSATION_HISTORY_OPEN_TAG}\n"
            f"{turns_block}\n{CONVERSATION_HISTORY_CLOSE_TAG}"
        )
        prompt = f"{prompt}\n\n{history_section}"
    recent_actions = list_most_recent_tool_call_per_tool(db_path=memory_db_path)
    if recent_actions:
        now = datetime.now(UTC)
        actions_block = "\n".join(
            f"{entry.tool_name} → {_escape_untrusted(entry.arguments_json)} | "
            f"{_format_time_ago(entry.created_at, now=now)}"
            for entry in recent_actions
        )
        actions_section = (
            f"{_RECENT_ACTIONS_FRAMING_HEADER}\n{RECENT_ACTIONS_OPEN_TAG}\n{actions_block}\n"
            f"{RECENT_ACTIONS_CLOSE_TAG}"
        )
        prompt = f"{prompt}\n\n{actions_section}"
    return prompt


def dispatch_turn(
    user_text: str,
    *,
    llm: LLMClient,
    tools: dict[str, Tool],
    tool_schemas: list[ToolSchema],
    policy: PolicyEngine,
    tts: TTSClient | None = None,
    memory_db_path: str | Path = MEMORY_DEFAULT_DB_PATH,
) -> str:
    """Un turno completo del planner bespoke de ADR-0005.

    Le pide al LLM una respuesta con los tools disponibles; si el LLM pide un tool-call, lo
    busca en `tools`, lo autoriza (o deniega) vía `PolicyEngine.authorize_and_execute` —el único
    punto de paso hacia `Tool.execute`, nunca se llama directo desde acá— y le devuelve el
    resultado al LLM como mensaje `role: tool` antes de pedir la respuesta final. Sin
    tool-call, se comporta igual que un `complete()` de una sola pasada.

    `MAX_TOOL_CALLS_PER_TURN` es un tope duro: corta el turno si el LLM sigue pidiendo tools más
    allá de lo razonable, en vez de loopear indefinidamente.

    Si se pasa `tts`, JARVIS dice una frase corta de acuse ("dejame revisar eso") apenas se
    detecta el PRIMER tool-call del turno, antes de ejecutarlo — un tool real (clima, búsqueda
    web) tarda unos segundos en volver, y sin esto quedaba en silencio todo ese tiempo, lo que se
    sentía como que se había colgado. `tts=None` (el default) preserva el comportamiento
    silencioso para quien llame a `dispatch_turn` sin audio (tests, u otros usos futuros no
    interactivos).

    Se dice como máximo UNA vez por turno, no una vez por tool-call encadenado (bug encontrado
    revisando `data/jarvis.log`/quejas en vivo del usuario: "dice muchas veces lo repetido y se
    demora en ejecutar cosas"). `SYSTEM_PROMPT` instruye explícitamente al LLM a encadenar
    `search_web` + `open_url` para pedidos tipo "poné tal canción en YouTube" — eso son 2-3
    iteraciones de este loop en un solo turno, y antes de este fix cada una volvía a decir la
    frase de acuse entera (con su propio round-trip de red al TTS y reproducción, ver
    `OpenAITTSClient.speak` en `jarvis.audio.tts` — no es gratis, es bloqueante), así que el usuario
    escuchaba "Dale, dejame revisar eso." repetido 2-3 veces seguidas Y esa repetición sumaba
    varios segundos reales de latencia extra encima de las llamadas al LLM/tools en sí. Una sola
    vez alcanza para el propósito original (avisar que no se colgó); repetirla no aporta nada
    nuevo y sí cuesta tiempo y suena mal.

    `memory_db_path` (default `jarvis.memory.store.DEFAULT_DB_PATH`) es la DB de la que se cargan
    los hechos recordados para este turno (`_build_system_prompt`) — parametrizable para tests
    (una DB en `tmp_path` en vez de la real) sin tocar el módulo de memoria. Al final del turno
    (con la respuesta final ya resuelta, antes de devolverla) se guarda el par
    `(user_text, respuesta final)` vía `save_conversation_turn` en esa misma DB — tanto en el
    camino sin tool-call como en el que agota `MAX_TOOL_CALLS_PER_TURN`, así el historial de
    conversación (ver docstring del módulo) refleja de verdad lo que el usuario escuchó, no solo
    el camino feliz.

    Cada tool-call que llega a `PolicyEngine.authorize_and_execute` se pasa con `on_executed`
    (`jarvis.memory.store.save_tool_call`, misma `memory_db_path`) — ese callback solo se dispara
    si el tool efectivamente ejecuta (SAFE, o CONFIRM aprobado; nunca CONFIRM denegado ni
    DANGEROUS, ver docstring de `PolicyEngine`), así que `tool_call_log` termina reflejando
    exactamente "qué llegó a correr de verdad", sin que este loop tenga que inspeccionar el texto
    de retorno para averiguarlo (ver docstring del módulo, sección "Últimas acciones por tool").
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(memory_db_path=memory_db_path),
        },
        {"role": "user", "content": user_text},
    ]
    acked = False
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        result = llm.complete(messages, tools=tool_schemas)
        if result.tool_call is None:
            final_text = _sanitize_final_response(result.text)
            save_conversation_turn(user_text, final_text, db_path=memory_db_path)
            return final_text
        tool_call = result.tool_call
        messages.append(_assistant_message_for_tool_call(result))
        if tool_call.arguments_error is None and tts is not None and not acked:
            tts.speak(TOOL_CALL_ACK_PHRASE)
            acked = True
        if tool_call.arguments_error is not None:
            # Argumentos malformados (JSON inválido, o JSON válido pero no un objeto) — nunca
            # se llega a `PolicyEngine`/`Tool.execute` con esto; se le devuelve el error al LLM
            # como mensaje `role: tool` para que pueda reintentar con argumentos válidos, en vez
            # de dejar propagar una excepción de parseo que tumbaría todo el proceso.
            tool_result = (
                f"No pude ejecutar '{tool_call.name}': {tool_call.arguments_error}"
            )
        else:
            tool = tools.get(tool_call.name)
            if tool is None:
                tool_result = f"Herramienta desconocida: {tool_call.name!r}."
            else:
                tool_name = tool_call.name
                tool_arguments = tool_call.arguments

                # Función anidada en vez de lambda con parámetros-default: ata explícitamente
                # `tool_name`/`tool_arguments` de esta iteración por argumento con anotación de
                # tipo (una lambda con defaults sin anotar deja a mypy sin poder inferir su tipo,
                # `misc`), evitando además la captura tardía de variables de loop que marcaría
                # ruff B023 — `authorize_and_execute` llama a `on_executed` de forma síncrona
                # dentro de esta misma iteración, pero ni el linter ni el type-checker pueden
                # saberlo mirando solo la firma.
                def _log_tool_call(
                    tn: str = tool_name, ta: dict[str, Any] = tool_arguments
                ) -> None:
                    save_tool_call(tn, ta, db_path=memory_db_path)

                tool_result = asyncio.run(
                    policy.authorize_and_execute(
                        tool, tool_arguments, on_executed=_log_tool_call
                    )
                )
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
        )
    fallback_reply = "No pude completar la solicitud con las herramientas disponibles."
    save_conversation_turn(user_text, fallback_reply, db_path=memory_db_path)
    return fallback_reply


def _process_command_text(
    text: str,
    *,
    sleeping: bool,
    llm: LLMClient,
    tools: dict[str, Tool],
    tool_schemas: list[ToolSchema],
    policy: PolicyEngine,
    tts: TTSClient,
    memory_db_path: str | Path = MEMORY_DEFAULT_DB_PATH,
) -> tuple[bool, bool]:
    """Núcleo de "qué hacer con un comando de voz ya resuelto a texto no vacío": dormir/despertar,
    o despachar vía `dispatch_turn`/`PolicyEngine`. Extraído de `run()` como su propia función
    para poder testearlo sin mockear `sd.InputStream`/hardware real.

    `memory_db_path` (default `jarvis.memory.store.DEFAULT_DB_PATH`, igual que `dispatch_turn`):
    parametrizable para tests (una DB en `tmp_path`) sin tocar `data/jarvis.db` real — `run()`
    nunca lo pasa explícitamente, así que su comportamiento en producción es idéntico a antes de
    agregar este parámetro.

    Devuelve `(sleeping, awaiting_wake_word)` actualizados — mismas variables que `run()` ya
    mutaba inline antes de esta extracción, ahora explícitas como valores de retorno en vez de
    mutación de closure.
    """
    if sleeping:
        if _contains_any_word(text, _WAKE_WORDS):
            sleeping = False
            wake_reply = "Volví. ¿En qué te ayudo?"
            print(f"JARVIS: {wake_reply}")
            tts.speak(wake_reply)
        else:
            print("(dormido, ignorando)", file=sys.stderr)
        # Dormir/despertar no abre ventana de seguimiento — mismo comportamiento que antes de
        # esta extracción.
        return sleeping, True

    if _contains_any_word(text, _SLEEP_WORDS):
        sleeping = True
        sleep_reply = 'Listo, descanso. Decime "Alexa, volvé" cuando me necesites.'
        print(f"JARVIS: {sleep_reply}")
        tts.speak(sleep_reply)
        return sleeping, True

    reply = dispatch_turn(
        text,
        llm=llm,
        tools=tools,
        tool_schemas=tool_schemas,
        policy=policy,
        tts=tts,
        memory_db_path=memory_db_path,
    )
    print(f"JARVIS: {reply}")
    if reply.strip():
        tts.speak(reply)
    # `awaiting_wake_word=False` ("comando real -> abrir ventana de seguimiento") es el mismo
    # comportamiento que antes de esta extracción.
    return sleeping, False


def run(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: int | None = None,
    duration: float | None = None,
) -> None:
    """Escuchar la wake word; al detectarla, grabar un comando y transcribirlo.

    Corre hasta Ctrl+C, o hasta que pasen `duration` segundos totales si se especifica.
    """
    # Crítico, no cosmético: corriendo sin consola (arranque automático vía pythonw.exe +
    # Tarea Programada), stdout/stderr redirigidos a archivo usan el codepage ANSI de Windows
    # (cp1252) por default, no UTF-8. Confirmado en vivo que eso tumbaba el proceso ENTERO —
    # `print(f"Dijiste: {text!r}")` con una tilde en el texto transcripto lanzaba
    # `UnicodeEncodeError` sin capturar, matando el loop de escucha completo (JARVIS dejaba de
    # responder hasta el próximo reinicio manual). `errors="replace"` como red de seguridad
    # adicional: aunque aparezca algún carácter que ni UTF-8 pueda manejar por algún motivo,
    # se reemplaza en vez de crashear.
    for stream in (sys.stdout, sys.stderr):
        # `.reconfigure()` es de `io.TextIOWrapper`, no del tipo estático `TextIO` que expone
        # el stub de `sys` — existe en runtime (stdout/stderr son `TextIOWrapper` de verdad),
        # mypy no lo sabe.
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    # Sin esto, cualquier `logging.getLogger(__name__).info/.debug(...)` de este codebase
    # (`jarvis.league.lcu_monitor`, ahora `jarvis.audio.stt`) es un no-op silencioso: sin ningún
    # handler configurado en ningún punto del proceso, Python usa el "handler de último recurso"
    # (`logging.lastResort`), que solo emite WARNING o más grave — `logger.exception` (ERROR) ya
    # se veía en `jarvis-error.log`, pero INFO/DEBUG nunca llegaban a ningún lado pese a que el
    # código los loguea como si fueran a aparecer. A stderr (mismo stream que ya reconfiguramos
    # arriba y que el lanzador de la Tarea Programada redirige a `jarvis-error.log`) — no a
    # stdout, que es donde van los `print()` de la transcripción de la conversación
    # ("Dijiste:"/"JARVIS:") en `jarvis.log`, para no mezclar ambos tipos de contenido.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # `basicConfig(level=INFO)` sube el nivel de TODOS los loggers del proceso, no solo los de
    # `jarvis.*` — sin esto, cada llamada HTTP de `httpx`/`httpcore` (LLM, STT, TTS, clima,
    # búsqueda) loguearía una línea "HTTP Request: ..." por request, inundando
    # `jarvis-error.log` con ruido de red en vez de la información de diagnóstico real que se
    # busca. Solo estos tres — más silencioso solo si aparece más ruido en vivo.
    for _noisy_logger_name in ("httpx", "httpcore", "openai"):
        logging.getLogger(_noisy_logger_name).setLevel(logging.WARNING)
    load_dotenv()
    wake_model = load_wake_word_model()
    stt_client = load_stt_client()
    # Credenciales para la captura de comandos generales en streaming (`jarvis.audio.realtime_stt`,
    # Realtime API de Speechmatics, ADR-0012) — `stt_client` (sync, OpenAI) se mantiene igual,
    # sigue siendo el que usa `VoiceConfirmationChannel` (camino batch, ver docstring de
    # `realtime_stt.py` para por qué las confirmaciones no migran).
    realtime_client = load_realtime_client()
    llm: LLMClient = load_deepseek_client_from_env()
    # Envuelto en `LockingTTSClient` (`jarvis.audio.tts`): `TimerScheduler` corre en su propio
    # thread de fondo y puede llamar `tts.speak()` en cualquier momento, incluso mientras el loop
    # principal está a mitad de decir otra cosa — algo que ningún otro llamador de `speak()` hacía
    # hasta ahora (ver docstring de `LockingTTSClient` para el razonamiento completo). Se envuelve
    # UNA vez acá, así que todo el resto de este módulo (`VoiceConfirmationChannel`, `dispatch_turn`,
    # los `tts.speak()` directos más abajo) queda serializado contra `TimerScheduler` sin tener que
    # tocar ningún otro call site.
    tts: TTSClient = LockingTTSClient(inner=load_default_tts_client())
    # Un solo thread de fondo para toda la corrida (no uno por iteración del loop de escucha) —
    # ver `loopback.py`: mide RMS de lo que reproduce el sistema para gatear falsos triggers de
    # wake word y contaminación del audio del comando por sonido del propio PC.
    #
    # `mic_device=device` es lo único que `SystemAudioMonitor` necesita de acá para decidir por su
    # cuenta, y de forma dinámica, si el gate debe aplicar (`device.is_combined_headset` — no
    # tiene sentido con un headset combinado, donde no hay filtración acústica real del audio de
    # salida hacia el mic del mismo headset). Antes esa decisión se calculaba UNA sola vez acá, al
    # arrancar (`is_combined_headset(...)` + `enabled=not headset_in_use`), y quedaba fija para
    # toda la corrida — si el usuario cambiaba de auriculares a parlantes (o al revés) a mitad de
    # sesión, o si el headset se desconectaba, el gate no se reajustaba hasta un reinicio manual
    # del proceso. Ahora `SystemAudioMonitor` re-resuelve el mic y la salida y recalcula
    # `is_combined_headset` en cada intento de (re)arranque de su thread de fondo (ver
    # `RETRY_SECONDS` en `loopback.py`), así que sigue vigente sin que `pipeline.py` tenga que
    # comunicarle nada más allá de qué mic está usando.
    system_audio = SystemAudioMonitor(mic_device=device)
    system_audio.start()
    # Servicio de fondo independiente, sin relación con audio: acepta automáticamente las colas
    # de matchmaking de League of Legends vía la LCU API mientras JARVIS está corriendo, sin
    # necesidad de ningún comando de voz (ver docstring de `jarvis.league.lcu_monitor` — mismo
    # lifecycle start()/stop() y misma resiliencia que `system_audio`, corre siempre, se queda
    # inactivo en silencio si League no está abierto). Opt-out vía `JARVIS_LOL_AUTO_ACCEPT`, ver
    # `_lol_auto_accept_enabled` — el objeto se construye siempre (así el `.stop()` del `finally`
    # de más abajo es incondicional e idempotente sin importar la variable de entorno), pero solo
    # se arranca si el opt-out no está activo.
    league_auto_accept = LCUAutoAcceptMonitor()
    if _lol_auto_accept_enabled():
        league_auto_accept.start()
    # Tercer servicio de fondo con el mismo lifecycle start()/stop(): sondea timers en memoria y
    # recordatorios persistidos (`jarvis.memory.store.reminders`) y los anuncia por `tts.speak()`
    # de forma proactiva, sin que haya ningún turno de voz en curso — ver docstring de
    # `jarvis.audio.timer_scheduler.TimerScheduler` para el porqué esto no puede vivir dentro de
    # `dispatch_turn()`. `TimerTool`/`ReminderTool` (más abajo, en `tools`) son los dos únicos
    # puntos de entrada que registran algo acá.
    timer_scheduler = TimerScheduler(tts=tts)
    timer_scheduler.start()

    tools: dict[str, Tool] = {
        tool.name: tool
        for tool in (
            WeatherTool(),
            SearchTool(),
            RememberTool(embedding_client=stt_client),
            RecallMemoryTool(embedding_client=stt_client),
            OpenAppTool(),
            OpenFileTool(),
            OpenUrlTool(),
            CloseAppTool(),
            TimerTool(scheduler=timer_scheduler),
            ReminderTool(),
            CancelTimerTool(scheduler=timer_scheduler),
            CancelReminderTool(),
            CancelAllTimersTool(scheduler=timer_scheduler),
            CancelAllRemindersTool(),
            MediaControlTool(),
            VolumeControlTool(),
            SystemInfoTool(),
            SystemPowerTool(),
            CancelSystemPowerTool(),
            ScreenshotTool(),
            SetRunesTool(),
            SetSummonerSpellsTool(),
            PreviewChampionTool(),
            LockChampionTool(),
            StartQueueTool(),
            SetLobbyQueueTool(),
            CancelQueueTool(),
        )
    }
    tool_schemas = [
        ToolSchema(
            name=tool.name, description=tool.description, parameters=tool.parameters
        )
        for tool in tools.values()
    ]

    print("Calibrando piso de ruido (no hables todavía)...", file=sys.stderr)
    noise_floor, noise_sample = measure_noise_floor(device=device)
    silence_threshold = calibrate_thresholds(noise_floor)
    print(
        f"Piso de ruido: {noise_floor:.1f} — umbral de voz calibrado: {silence_threshold:.1f}",
        file=sys.stderr,
    )
    confirmation = VoiceConfirmationChannel(
        tts=tts,
        stt_client=stt_client,
        device=device,
        silence_threshold=silence_threshold,
        system_audio=system_audio,
        noise_sample=noise_sample,
    )
    policy = PolicyEngine(confirmation)

    deadline = time.monotonic() + duration if duration is not None else None
    sleeping = False
    print(
        "Escuchando... decí 'Alexa', 'Hey Jarvis' o 'Hey Mycroft' "
        f"(Ctrl+C para salir, umbral={threshold})",
        file=sys.stderr,
    )
    # Corre sin ventana visible (arranque automático, ver scripts/start_jarvis.ps1) — este es el
    # único aviso de que arrancó bien y quedó escuchando (pedido explícito del usuario: "al
    # iniciar quiero que se presente para saber si está en buen estado").
    greeting = "Alexa activa y funcionando correctamente."
    try:
        tts.speak(greeting)
    except Exception as exc:  # noqa: BLE001 — frontera de recuperación explícita: desde que se
        # sacó el fallback local de TTS (ADR-0011), un fallo real de la API (ej. créditos de
        # OpenAI agotados — bug real, en vivo: mató el proceso completo apenas arrancaba, ni
        # siquiera llegaba a abrir el loop principal) ya no degrada solo, así que este saludo
        # necesita su PROPIA red de seguridad — no está cubierto por el `except Exception` del
        # loop principal más abajo, que arranca recién dentro del `while`. Perder el saludo de
        # arranque es aceptable (el usuario no se entera de que JARVIS está vivo por voz esa vez,
        # pero sí queda el log); que JARVIS no arranque en absoluto por eso, no.
        print(f"No se pudo anunciar el arranque por voz: {exc!r}", file=sys.stderr)
    try:
        while deadline is None or time.monotonic() < deadline:
            remaining = (deadline - time.monotonic()) if deadline is not None else None
            frames = iter_microphone_frames(device=device, duration=remaining)
            pre_roll_buffer: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
            # Detector de voz nuevo por ciclo de escucha (misma vida útil que `iter_microphone_
            # frames`/`pre_roll_buffer` de esta misma iteración) — mismo criterio que
            # `record_command`/`stream_transcribe_command`: `SileroVAD` tiene estado interno tipo
            # RNN sin reset expuesto, así que una instancia fresca por sesión es la forma segura de
            # que el contexto de un ciclo no se filtre al siguiente (ver
            # `jarvis.audio.speech_detector.ChunkSpeechDetector`). Cierra el hueco real
            # confirmado en vivo (ver docstring de `detect()` en `wake_word.py`): con el gate de
            # `system_audio` desactivado por headset combinado, audio de fondo (música/video)
            # bleeding al mic disparaba la wake word solo, repetidamente, con score=1.00.
            wake_speech_detector = load_speech_detector()
            try:
                hit = next(
                    detect(
                        tee_frames(frames, pre_roll_buffer),
                        model=wake_model,
                        threshold=threshold,
                        system_audio=system_audio,
                        speech_detector=wake_speech_detector,
                    ),
                    None,
                )
            except (sd.PortAudioError, OSError) as exc:
                # Mismo principio que el except del turno más abajo (ver su comentario sobre el
                # incidente en vivo del UnicodeEncodeError que tumbó el proceso entero) — acá es
                # la MISMA clase de incidente pero en la otra mitad del loop: confirmado en vivo
                # (`AUDCLNT_E_DEVICE_INVALIDATED`, PortAudioError -9999) que un dispositivo de
                # entrada desconectado/invalidado a mitad de la ESPERA de la wake word (antes de
                # detectar nada, no durante el procesamiento de un turno ya en curso) propagaba
                # sin capturar hasta `main()` y mataba el proceso completo — el except de más
                # abajo nunca corría porque el error pasa antes de siquiera entrar al loop
                # interno. `sd.PortAudioError` (falla reportada por PortAudio/WASAPI, ej. device
                # invalidado o desconectado en caliente) y `OSError` (mismo tipo de falla puede
                # surgir como error de sistema según el modo exacto de falla) son las dos formas
                # razonables en que esto se manifiesta — no `Exception` genérico como en el except
                # de más abajo, porque acá sí sabemos qué esperar (falla de hardware/stream de
                # audio, no cualquier error de LLM/tool/TTS).
                #
                # Recuperación: NO reintentar sobre el mismo stream (`frames`) — un device
                # invalidado casi siempre necesita un stream nuevo, no un `read()` de nuevo sobre
                # el mismo handle. `continue` vuelve al principio del `while`, donde
                # `iter_microphone_frames()` se vuelve a llamar de cero más arriba, y
                # `resolve_input_device()` (`jarvis.audio.device`, sin ningún caché — vuelve a
                # consultar `sd.query_devices()`/`sd.default.device` en cada llamada) re-resuelve
                # el device desde cero en la siguiente iteración: un stream realmente nuevo, no un
                # reintento ciego sobre el mismo objeto roto. Un `COOLDOWN_SECONDS` fijo antes de
                # reintentar evita spinnear el CPU en loop apretado si el device está
                # permanentemente desconectado (mismo estilo que el resto de este módulo — nada de
                # backoff exponencial ni jitter, sería sobre-ingeniería para este caso).
                print(f"Error en el stream de escucha: {exc!r}", file=sys.stderr)
                frames.close()
                time.sleep(COOLDOWN_SECONDS)
                continue
            frames.close()  # cierra el stream de escucha antes de abrir el de grabación
            if hit is None:
                break  # se acabó la ventana de duration sin detectar nada más
            print(
                f"Wake word detectada (score={hit.score:.2f}). Escuchando comando...",
                file=sys.stderr,
            )
            # Después de un comando real (no dormir/despertar), se escucha una ventana corta de
            # seguimiento sin volver a pedir la wake word — pedido explícito del usuario: "Alexa,
            # abrí YouTube" y después, sin decir "Alexa" de nuevo, "ahora reproducí esto". Si no
            # hay nada en esa ventana, se vuelve a exigir la wake word normalmente.
            awaiting_wake_word = False
            first_listen = True

            while not awaiting_wake_word:
                try:
                    pre_roll = (
                        np.concatenate(list(pre_roll_buffer))
                        if first_listen and pre_roll_buffer
                        else None
                    )
                    max_duration = (
                        COMMAND_WINDOW_SECONDS
                        if first_listen
                        else FOLLOW_UP_WINDOW_SECONDS
                    )
                    first_listen = False
                    # Captura + transcripción en streaming (`jarvis.audio.realtime_stt`,
                    # Realtime API de Speechmatics, ADR-0012) en vez del combo
                    # `record_command()` + `transcribe()` batch de antes — mismo contrato de
                    # `speech_detected` (ver docstring de `stream_transcribe_command`). `run()` es
                    # sync (igual que el resto de este loop); `asyncio.run()` acá es el mismo
                    # patrón que ya usa `dispatch_turn` para invocar
                    # `policy.authorize_and_execute` desde código sync.
                    text, speech_detected = asyncio.run(
                        stream_transcribe_command(
                            client=realtime_client,
                            device=device,
                            silence_threshold=silence_threshold,
                            max_duration=max_duration,
                            pre_roll=pre_roll,
                            system_audio=system_audio,
                        )
                    )
                    if speech_detected and _contains_non_latin_script(text):
                        # Alucinación confirmada en vivo (ver docstring de
                        # `_contains_non_latin_script`) — se descarta acá, antes del print y de
                        # `save_speech_sample` más abajo, para que ni contamine el log ni la
                        # memoria de estilo de habla con texto que el usuario nunca dijo.
                        text = ""
                    print(f"Dijiste: {text!r}")
                    if text.strip():
                        # Log automático, sin juicio del LLM (a diferencia de `remember_fact`):
                        # toda transcripción real se guarda como muestra de estilo de habla, sin
                        # importar si después resulta ser un comando para dormir/despertar o algo
                        # que dispatch_turn no puede resolver — aprender CÓMO habla el usuario es
                        # ortogonal a QUÉ pidió en este turno puntual (ver docstring del módulo).
                        save_speech_sample(text, db_path=MEMORY_DEFAULT_DB_PATH)
                    if not text.strip():
                        # Silencio real (sin habla, o la ventana de seguimiento se agotó) — se
                        # acaba la ventana de seguimiento y se vuelve a pedir la wake word.
                        print(
                            "(nada que responder, no se detectó habla real)",
                            file=sys.stderr,
                        )
                        awaiting_wake_word = True
                    else:
                        sleeping, awaiting_wake_word = _process_command_text(
                            text,
                            sleeping=sleeping,
                            llm=llm,
                            tools=tools,
                            tool_schemas=tool_schemas,
                            policy=policy,
                            tts=tts,
                        )
                except Exception as exc:  # noqa: BLE001 — última línea de defensa del turno: un
                    # error inesperado en STT/LLM/tool/TTS de un turno puntual no tiene que
                    # tumbar el proceso entero (pedido implícito por el incidente en vivo: un
                    # UnicodeEncodeError en un print() mató el loop completo y JARVIS dejó de
                    # responder hasta el próximo reinicio manual — este turno se pierde, pero el
                    # siguiente "Hey Jarvis"/"Alexa" sigue funcionando en vez de silencio total).
                    print(f"Error procesando el turno: {exc!r}", file=sys.stderr)
                    awaiting_wake_word = True
                time.sleep(COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        print("Detenido.", file=sys.stderr)
    finally:
        system_audio.stop()
        league_auto_accept.stop()
        timer_scheduler.stop()
    print("Fin de la escucha.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline JARVIS: wake word + transcripción"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Índice del dispositivo de entrada (ver sounddevice.query_devices())",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Segundos totales de escucha (default: infinito, Ctrl+C)",
    )
    args = parser.parse_args()
    run(threshold=args.threshold, device=args.device, duration=args.duration)


if __name__ == "__main__":
    main()
