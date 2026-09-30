#!/usr/bin/env zsh
# Static posters for the widgets (shown in PDF export and before a widget loads).
# Each widget opens with ?demo, which pre-runs a representative scenario where it has one.
set -euo pipefail
cd "${0:A:h:h}/widgets"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for w in ${@:-*.html}; do
  name="${w:r}"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
    --window-size=1600,820 --virtual-time-budget=4000 \
    --screenshot="${PWD}/${name}-poster.png" "file://${PWD}/${w}?demo" 2>/dev/null
  print "${name}-poster.png"
done
