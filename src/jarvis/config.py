"""Carga de `.env` a variables de entorno.

Sin dependencia externa (`python-dotenv`): el formato que usamos (`.env.example`) es
`CLAVE=valor` simple, sin comillas ni interpolación, así que un parser mínimo alcanza.

Regla dura de este módulo: nunca loguear, imprimir, ni incluir en excepciones el *valor* de una
variable cargada — solo nombres de variables (`.claude/rules/security.md`, "Higiene de
secretos").
"""

from __future__ import annotations

from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Leer `path` y setear cada `CLAVE=valor` en `os.environ`, si no está ya seteada.

    No sobreescribe variables de entorno ya presentes (mismo comportamiento que
    `python-dotenv` por default) — así una var exportada manualmente en la sesión de shell
    siempre gana sobre el archivo.
    """
    import os

    env_path = Path(path)
    if not env_path.is_file():
        return
    # utf-8-sig, no utf-8: PowerShell (`Set-Content -Encoding utf8` en Windows PowerShell 5.1)
    # antepone un BOM. Con "utf-8" simple, el BOM queda pegado al nombre de la primera variable
    # del archivo y esa variable nunca matchea — confirmado, no hipotético (el .env real de este
    # proyecto lo tiene). utf-8-sig lo descarta si está, y no rompe nada si no está.
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()
