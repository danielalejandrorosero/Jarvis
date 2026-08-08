"""Tests para `load_dotenv()` (carga de `.env` a `os.environ`).

Usan archivos temporales reales (`tmp_path`), nunca el `.env` del proyecto. `os.environ` se
aisla en cada test vía `monkeypatch` para no filtrar estado entre tests ni hacia el proceso
real de pytest.

Regla dura (`.claude/rules/security.md`, "Higiene de secretos"): estos tests nunca imprimen ni
incluyen en asserts el *valor* real de una API key; los valores usados acá son placeholders
inventados para el test, no secretos reales.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis.config import load_dotenv


@pytest.fixture(autouse=True)
def _isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reemplaza `os.environ` por un dict vacío para cada test: `load_dotenv()` nunca ve ni
    modifica el entorno real del proceso de test."""
    monkeypatch.setattr(os, "environ", {})


def test_load_dotenv_sets_vars_from_well_formed_file(tmp_path: Path) -> None:
    """Caso aceptado: un archivo de 2 líneas bien formado setea ambas vars en os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_load_dotenv_strips_utf8_bom_from_first_key(tmp_path: Path) -> None:
    """Regresión: un archivo con BOM UTF-8 al inicio (lo que produce
    `Set-Content -Encoding utf8` en Windows PowerShell 5.1) no debe dejar el BOM pegado al
    nombre de la primera variable. Con "utf-8" simple esto rompía el match silenciosamente."""
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xef\xbb\xbfKEY=value")
    keys_before = set(os.environ.keys())

    load_dotenv(env_file)

    assert os.environ["KEY"] == "value"
    # Se agrega exactamente una clave, "KEY": si el BOM hubiera quedado pegado (p. ej.
    # "﻿KEY"), esta comparación lo detectaría como una clave distinta a "KEY".
    assert set(os.environ.keys()) - keys_before == {"KEY"}


def test_load_dotenv_skips_blank_comment_and_malformed_lines(tmp_path: Path) -> None:
    """Caso rechazado: líneas en blanco, comentarios (`#`) y líneas sin `=` se ignoran sin
    lanzar excepción y sin agregar claves espurias."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n"
        "# esto es un comentario\n"
        "sin_signo_igual\n"
        "REAL=value\n"
        "   \n",
        encoding="utf-8",
    )
    keys_before = set(os.environ.keys())

    load_dotenv(env_file)

    assert os.environ["REAL"] == "value"
    assert set(os.environ.keys()) - keys_before == {"REAL"}


def test_load_dotenv_does_not_override_already_set_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Una variable ya presente en os.environ (p. ej. exportada manualmente) siempre gana sobre
    el valor del archivo."""
    monkeypatch.setenv("FOO", "original")
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=from_file\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["FOO"] == "original"


def test_load_dotenv_missing_file_is_a_silent_noop(tmp_path: Path) -> None:
    """Si el archivo no existe, no lanza excepción y no modifica os.environ."""
    missing = tmp_path / "does_not_exist.env"
    environ_before = dict(os.environ)

    load_dotenv(missing)

    assert os.environ == environ_before
