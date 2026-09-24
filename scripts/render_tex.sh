#!/usr/bin/env bash
# Render one attendance_exercises/attendance_solutions Markdown sheet to LaTeX
# via pandoc + templates/course-sheet.latex, so that .md stays the single
# source of truth and .tex is always regenerated from it (in CI and locally),
# instead of being hand-maintained and drifting out of sync.
#
# Usage: scripts/render_tex.sh <path/to/sheet.md>
# Writes <path/to/sheet.tex> next to it.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path/to/sheet.md>" >&2
    exit 1
fi

md_file="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
template="$repo_root/templates/course-sheet.latex"
out_file="${md_file%.md}.tex"

if [[ "$md_file" == *attendance_solutions/* ]]; then
    solutions=true
else
    solutions=false
fi

# Some sheets use ATX ("# Title") headings, others Setext ("Title\n====").
# Normalize to ATX first so title extraction/removal below is uniform.
normalized_md="$(mktemp)"
trap 'rm -f "$normalized_md" "$body_md"' EXIT
pandoc "$md_file" --from=gfm --to=gfm --markdown-headings=atx -o "$normalized_md"

title="$(grep -m1 '^# ' "$normalized_md" | sed 's/^# *//')"
if [ -z "$title" ]; then
    echo "No top-level '# Title' heading found in $md_file" >&2
    exit 1
fi

# Drop only the first line that is exactly the title heading; any text
# before it (some sheets open with a short intro sentence) stays as body.
body_md="$(mktemp)"
awk -v heading="# $title" '!done && $0 == heading { done=1; next } { print }' "$normalized_md" > "$body_md"

pandoc "$body_md" \
    --from=gfm \
    --to=latex \
    --no-highlight \
    --template="$template" \
    -M title="$title" \
    -M solutions="$solutions" \
    -o "$out_file"

echo "Rendered $md_file -> $out_file"
