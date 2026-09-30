#!/usr/bin/env python3
"""Add clickable URI annotations to a slide PDF built from PNG screenshots."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from playwright.sync_api import sync_playwright

SLIDE_ID_RE = re.compile(r'<section id="([^"]+)"')
LINK_JS = """() => {
  const slide = document.querySelector('section.present')
    || document.querySelector('.slides > section');
  if (!slide) return [];
  return [...slide.querySelectorAll('a[href^="http"]')].map((e) => {
    const r = e.getBoundingClientRect();
    return { href: e.href, x: r.x, y: r.y, w: r.width, h: r.height };
  });
}"""


def slide_ids(html_path: Path) -> list[str]:
    text = html_path.read_text(encoding="utf-8")
    return SLIDE_ID_RE.findall(text)


def slide_url(base_url: str, slide_id: str) -> str:
    if slide_id == "title-slide":
        return base_url.split("#")[0]
    return f"{base_url.split('#')[0]}#/{slide_id}"


def html_rect_to_pdf(rect: tuple[float, float, float, float], page_height: float) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    return (x, page_height - y - h, x + w, page_height - y)


def collect_links(base_url: str, html_path: Path, viewport: tuple[int, int], wait_ms: int) -> list[list[dict]]:
    ids = slide_ids(html_path)
    per_slide: list[list[dict]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        for slide_id in ids:
            page.goto(slide_url(base_url, slide_id), wait_until="networkidle")
            page.wait_for_timeout(wait_ms)
            raw = page.evaluate(LINK_JS)
            links = [
                item
                for item in raw
                if item["w"] > 1 and item["h"] > 1 and item.get("href", "").startswith("http")
            ]
            per_slide.append(links)
        browser.close()
    return per_slide


def annotate(pdf_path: Path, links_per_page: list[list[dict]]) -> int:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append(reader)
    if len(links_per_page) != len(writer.pages):
        raise SystemExit(
            f"Slide/link count mismatch: {len(links_per_page)} slides vs {len(writer.pages)} PDF pages"
        )

    added = 0
    for page_index, links in enumerate(links_per_page):
        page_height = float(writer.pages[page_index].mediabox.height)
        for item in links:
            rect = html_rect_to_pdf((item["x"], item["y"], item["w"], item["h"]), page_height)
            writer.add_annotation(page_index, Link(rect=rect, url=item["href"]))
            added += 1

    tmp = pdf_path.with_suffix(".tmp.pdf")
    with tmp.open("wb") as fh:
        writer.write(fh)
    tmp.replace(pdf_path)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:4187/aman.html")
    parser.add_argument("--html", type=Path, default=Path("aman.html"))
    parser.add_argument("--pdf", type=Path, default=Path("aman.pdf"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--wait-ms", type=int, default=1500)
    args = parser.parse_args()

    if not args.html.is_file():
        raise SystemExit(f"Missing HTML: {args.html}")
    if not args.pdf.is_file():
        raise SystemExit(f"Missing PDF: {args.pdf}")

    links_per_page = collect_links(
        args.base_url,
        args.html,
        (args.width, args.height),
        args.wait_ms,
    )
    total_links = sum(len(x) for x in links_per_page)
    if total_links == 0:
        print("annotate_pdf_links: no http links found; PDF unchanged", file=sys.stderr)
        return

    added = annotate(args.pdf, links_per_page)
    print(f"annotate_pdf_links: added {added} link(s) across {len(links_per_page)} page(s)")


if __name__ == "__main__":
    main()
