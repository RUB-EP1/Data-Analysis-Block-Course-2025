# Data Analysis Block Course 2026

Exercise notebooks and attendance sheets for the data analysis block course at
TU Dortmund and Ruhr-University Bochum.

https://indico.global/event/14058/

## Attendance sheets

Edit the Markdown files in `attendance_exercises/` and `attendance_solutions/`.
Markdown is the source of truth for both content and layout choices. The
reviewed content of sheets 1–6, including the complete solutions, equations,
tables and diagrams, has been transferred from the edited LaTeX documents.
The filenames are retained, including `4_boosting_olution`. Sheet 6 covers Adam.

Build all active sheets with Pandoc, Python 3, latexmk and TeX Live:

```sh
python3 scripts/build_attendance.py
```

To build selected sheets:

```sh
python3 scripts/build_attendance.py attendance_exercises/1_automatic_differentiation.md attendance_solutions/1_automatic_differentiation_solution.md
```

PDFs and logs are written to `build/attendance_exercises/` and
`build/attendance_solutions/`; generated TeX lives in `build/tex/`. The GitHub
workflow uses exactly this command and uploads the PDFs and build sources/logs.
Every build starts by rendering the Markdown. Existing TeX previews cannot
hide Markdown changes. Only active Markdown sheets are compiled; archived
solutions and other TeX files are excluded.

All sheets use `course_sheet.cls`, `2026-SoSe.tex`, `course-info.tex`, and the
logos in `img/` for the shared course header. The publication date and title
are specified in each revised Markdown file's YAML header. Sheet 7 also builds
with this header; it has not undergone the content review applied to sheets 1–6.

For a TeX preview beside the Markdown (for example, to inspect it in VS Code):

```sh
bash scripts/render_tex.sh attendance_exercises/1_automatic_differentiation.md
bash scripts/compile_sheet.sh attendance_exercises/1_automatic_differentiation.tex
```

These TeX previews are generated files and are overwritten when rendered
again. Make lasting changes in Markdown. A preview beside its Markdown can
also be compiled from its own folder in VS Code using pdfLaTeX or latexmk.
The build helper keeps generated sources separate and compiles from the
repository root so that the shared class and images are found.

The editorial rules and Markdown notation are documented in
[`templates/attendance-style.md`](templates/attendance-style.md).
`templates/course-sheet.lua` implements the equation environments, simple
`tabular` tables and heading layout, while `templates/course-sheet.latex`
provides the document preamble and course header.

## Figures and numerical checks

Solution 2 contains the threshold scatter plot in `img/temperature_roc.pdf`.
Regenerate it from the Markdown patient table with
`python3 scripts/plot_temperature_roc.py` (Matplotlib).
The tree diagrams in solution 3 can be regenerated with
`python3 scripts/plot_decision_trees.py` (Matplotlib and NumPy).
Solution 5 keeps its TikZ diagrams in raw LaTeX blocks inside the Markdown.

Run `python3 scripts/check_attendance_calculations.py` for the finite numerical
checks of sheets 3–6 (Python standard library only).
For the content review, see [`docs/attendance-review-3-6.md`](docs/attendance-review-3-6.md).
The unrelated former solution to sheet 6 is preserved in
`attendance_solutions/archive/6_regularization_solution.md` and is not built.
