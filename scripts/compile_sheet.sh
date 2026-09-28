#!/usr/bin/env bash
# Compile an existing TeX preview. Use build_attendance.py to rebuild from Markdown.
# Usage: bash scripts/compile_sheet.sh attendance_exercises/1_automatic_differentiation.tex
set -euo pipefail

if [ "$#" -ne 1 ] || [[ "$1" != *.tex ]] || [ ! -f "$1" ]; then
    echo "Usage: $0 <existing-sheet.tex>" >&2
    exit 1
fi

tex_file="$(realpath "$1")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sheet_dir="$(basename "$(dirname "$tex_file")")"
output_dir="$repo_root/build/$sheet_dir"
cd "$repo_root"
mkdir -p "$output_dir"
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -synctex=1 \
    -outdir="$output_dir" "$tex_file"
cp "$output_dir/$(basename "${tex_file%.tex}").pdf" "${tex_file%.tex}.pdf"
echo "Compiled ${tex_file%.tex}.pdf"
