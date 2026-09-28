"""Regression checks for fresh Markdown builds and math-aware conversion."""
import re
from pathlib import Path
import tempfile
import unittest
from render_tex import ROOT, render


class RendererTest(unittest.TestCase):
    def test_markdown_edits_replace_existing_tex_and_preserve_math(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / '1_example.md'
            output = source.with_suffix('.tex')
            output.write_text('% Editable pilot sheet.\nSTALE CONTENT')
            source.write_text('''---
title: A test sheet
published: 25 September 2026
---
# Key concepts

1. First question.
2. Second question.

::: numbered
$$
f(x)
=
x^2
$$
:::

$$g(x)=0$$

| A | B |
|---|---|
| value | explanation |

![Diagram](../img/trees_forest.pdf){width="75%"}
''')
            render(source)
            tex = output.read_text()
            for required in (r'\documentclass{course_sheet}', r'\published{25 September 2026}',
                             r'\section{Key concepts}', r'\begin{enumerate}',
                             r'\begin{equation}', r'\begin{equation*}', r'\begin{tabular}', r'\includegraphics[width=0.75\linewidth]{img/trees_forest.pdf}'):
                self.assertIn(required, tex)
            for forbidden in ('STALE CONTENT', r'\begin{longtable}', r'\begin{minipage}', r'\begin{figure}', r'\caption'):
                self.assertNotIn(forbidden, tex)
            self.assertRegex(tex, r'f\(x\)\s*=\s*x\^2')
            self.assertNotRegex(tex, r'\\begin\{equation\*?\}\n\s*\n')
            source.write_text(source.read_text().replace('First question.', 'Changed question.'))
            render(source)
            self.assertIn('Changed question.', output.read_text())
            self.assertNotIn('First question.', output.read_text())

    def test_legacy_title_is_not_repeated_as_a_section(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / '1_legacy.md'
            source.write_text('# Legacy title\n\n## Part I\n\nText.\n')
            tex = render(source).read_text()
            self.assertIn(r'\title{\\Legacy title}', tex)
            self.assertIn(r'\section{Part I}', tex)
            self.assertNotIn(r'\section{Legacy title}', tex)


if __name__ == '__main__':
    unittest.main()
