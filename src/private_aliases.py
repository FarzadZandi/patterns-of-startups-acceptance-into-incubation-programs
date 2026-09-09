"""Load confidential startup-name aliases from an ignored local file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable


DEFAULT_ALIAS_PATH = Path("labels/startup_aliases.json")


def expand_private_aliases(
    aliases: set[str],
    normalize: Callable[[object], str],
) -> set[str]:
    """Expand aliases using a local JSON mapping that is excluded from Git."""
    path = Path(os.environ.get("CF_STARTUP_ALIASES", DEFAULT_ALIAS_PATH))
    if not path.is_file():
        return aliases

    with path.open(encoding="utf-8") as handle:
        private_aliases = json.load(handle)
    if not isinstance(private_aliases, dict):
        raise ValueError(f"Private alias file must contain a JSON object: {path}")

    normalized = {
        normalize(source): normalize(target)
        for source, target in private_aliases.items()
    }
    aliases.update(
        normalized[alias]
        for alias in list(aliases)
        if alias in normalized and normalized[alias]
    )
    return aliases
