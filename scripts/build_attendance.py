#!/usr/bin/env python3
"""Build fresh PDFs from attendance Markdown, locally and in GitHub Actions."""
import argparse
import os
from pathlib import Path
import subprocess
from render_tex import ROOT, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', type=Path,
                        help='Selected Markdown files; default: all active sheets')
    args = parser.parse_args()
    sources = args.sources or sorted(
        p for folder in ('attendance_exercises', 'attendance_solutions')
        for p in (ROOT / folder).glob('[0-9]*.md'))
    if not sources:
        parser.error('No attendance Markdown sources found')
    env = dict(os.environ)
    env['TEXINPUTS'] = str(ROOT) + ':' + env.get('TEXINPUTS', '')
    for source in sources:
        source = source.resolve()
        if source.parent.name not in ('attendance_exercises', 'attendance_solutions'):
            parser.error(f'Not an active attendance sheet: {source}')
        tex = render(source, ROOT / 'build/tex' / source.parent.name / source.with_suffix('.tex').name)
        output = ROOT / 'build' / source.parent.name
        output.mkdir(parents=True, exist_ok=True)
        log = output / (source.stem + '.build.log')
        with log.open('w') as stream:
            result = subprocess.run([
                'latexmk', '-norc', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                '-file-line-error', '-outdir=' + str(output), str(tex),
            ], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-10000:])
            raise SystemExit(f'Build failed: {source.name}; see {log}')
        print(f'PDF: {output / source.with_suffix(".pdf").name}')


if __name__ == '__main__':
    main()
