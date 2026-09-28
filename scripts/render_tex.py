#!/usr/bin/env python3
"""Render one Markdown sheet with the shared class and editorial conventions."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def render(source, output=None):
    source = Path(source).resolve()
    if source.suffix != '.md' or not source.is_file():
        raise ValueError(f'Expected an existing Markdown file: {source}')
    doc = json.loads(subprocess.check_output([
        'pandoc', str(source), '--from=markdown-smart-implicit_figures', '--to=json']))
    meta = doc['meta']
    if 'title' not in meta:
        for i, block in enumerate(doc['blocks']):
            if block['t'] == 'Header' and block['c'][0] == 1:
                meta['title'] = {'t': 'MetaInlines', 'c': block['c'][2]}
                del doc['blocks'][i]
                # Legacy sheets used H2 for their top-level body headings.
                levels = [b['c'][0] for b in doc['blocks'] if b['t'] == 'Header']
                shift = min(levels, default=1) - 1
                for b in doc['blocks']:
                    if b['t'] == 'Header':
                        b['c'][0] -= shift
                break
        else:
            raise ValueError(f'Missing title in {source}')
    if 'published' not in meta:
        number = int(source.name.split('_')[0])
        day = {1:25, 2:25, 3:28, 4:28, 5:29, 6:29, 7:30}[number]
        meta['published'] = {'t': 'MetaString', 'c': f'{day} September 2026'}
    meta['solutions'] = {'t': 'MetaBool', 'c': source.parent.name == 'attendance_solutions'}
    rendered = subprocess.check_output([
        'pandoc', '--from=json', '--to=latex', '--no-highlight',
        '--template=' + str(ROOT / 'templates/course-sheet.latex'),
        '--lua-filter=' + str(ROOT / 'templates/course-sheet.lua'),
    ], input=json.dumps(doc).encode()).decode()
    output = Path(output) if output else source.with_suffix('.tex')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('% Generated from Markdown. Edit the .md source, then render again.\n' + rendered)
    print(f'Rendered {source.relative_to(ROOT)} -> {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
