#!/usr/bin/env bash
# Markdown is authoritative; regenerate the TeX preview on every call.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$repo_root/scripts/render_tex.py" "$@"
