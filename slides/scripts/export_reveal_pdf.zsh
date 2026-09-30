#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

DECK="${DECK:-deck}"
PORT="${PORT:-4187}"
BASE_URL="${1:-http://127.0.0.1:${PORT}/${DECK}.html?export=1}"
OUT_DIR="${2:-qa/pdf-slides}"
PDF_OUT="${3:-${DECK}.pdf}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VENV="${ROOT}/.venv-pdf"
ANNOTATE="${ROOT}/scripts/annotate_pdf_links.py"
HTML="${ROOT}/${DECK}.html"

if [[ ! -f "$HTML" ]]; then
  print -u2 "missing $HTML — run quarto render first (make pdf does this)"
  exit 1
fi

if [[ ! -x "$CHROME" ]]; then
  print -u2 "Google Chrome not found at $CHROME"
  exit 1
fi

mkdir -p "$OUT_DIR"
setopt NULL_GLOB
rm -f "$OUT_DIR"/slide-*.png
unsetopt NULL_GLOB

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "$VENV"
  "${VENV}/bin/pip" install -q -r "${ROOT}/scripts/requirements-pdf.txt"
fi

ids=("${(@f)$(perl -ne 'print "$1\n" if /<section id="([^"]+)"/' "$HTML")}")
if (( ${#ids} == 0 )); then
  print -u2 "no <section id=...> slides in ${DECK}.html"
  exit 1
fi

server_pid=""
if ! curl -sf "http://127.0.0.1:${PORT}/${DECK}.html" >/dev/null; then
  python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
  server_pid=$!
  trap '[[ -n ${server_pid} ]] && kill ${server_pid} 2>/dev/null || true' EXIT
  for _ in {1..50}; do
    if curl -sf "http://127.0.0.1:${PORT}/${DECK}.html" >/dev/null; then
      break
    fi
    sleep 0.1
  done
fi

i=1
for id in "${ids[@]}"; do
  printf -v number "%02d" "$i"
  url="${BASE_URL}#/${id}"
  "$CHROME" \
    --headless \
    --disable-gpu \
    --no-sandbox \
    --hide-scrollbars \
    --window-size=1920,1080 \
    --run-all-compositor-stages-before-draw \
    --virtual-time-budget=90000 \
    --screenshot="${OUT_DIR}/slide-${number}.png" \
    "$url" >/dev/null 2>&1
  (( i++ ))
done

# 1920×1080 at 144 dpi → 13.333" × 7.5" (16:9)
magick "${OUT_DIR}"/slide-*.png \
  -units PixelsPerInch -density 144 \
  -page 1920x1080 \
  "$PDF_OUT"

"${VENV}/bin/python" "${ANNOTATE}" \
  --html "$HTML" \
  --base-url "$BASE_URL" \
  --pdf "$PDF_OUT" \
  --width 1920 \
  --height 1080

print "wrote ${PDF_OUT} (${#ids} slides, 1920×1080, 16:9)"
