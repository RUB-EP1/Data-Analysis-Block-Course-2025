# Course lecture decks

The course uses eight Quarto Reveal decks. The planned deck names and migration
status live in `lectures.yml`.

## Directory convention

- One workspace per deck: `lectures/Lecture_*/`.
- One editable source in each workspace: `Lecture_*/Lecture_*.qmd`.
- Deck-specific media live in that workspace's `figures/` directory.
- Lecture-specific demonstrations and code live in `widgets/` and `notebooks/`.
- Generated HTML and PDFs sit beside the lecture source inside its workspace.
- Shared styling: `styles.css` and `lecture-styles.css`.
- Shared browser helpers: `scripts/`.
- Archived Keynote, PowerPoint, and PDF references: `previous_lectures/`.
- New-deck starter: `templates/Lecture_TEMPLATE.qmd`.

Do not create one `.qmd` file per slide. A lecture may use a separate include
only for a substantial reusable block, such as a long interactive section.

Lectures 1B, 2A, 2B, 3A, 3B, 4A and 4B are fully native Quarto. Lecture 2B includes three interactive
widgets (split search, regression boosting, BCE classification). Lecture 2A includes four interactive
widgets and executable Julia notebooks in its local workspace. Its obsolete full-slide SVG screenshots and
per-slide include files have been removed. Lecture 3A (neural networks, gradient descent,
optimizers) includes five widgets. Lecture 3B (convolutional networks) includes four widgets (two run a trained MNIST CNN in the browser)
and a marimo notebook that opens the pretrained AlexNet. Lecture 4A (transformers) includes five widgets and a marimo notebook
that opens GPT-2 and trains a tiny GPT. Lecture 4B (chats and agents) includes four widgets and
a marimo notebook with a complete agent loop. Lecture 1A still contains imported
reference slides; the combined legacy Lecture 3 is kept only as a reference for 3A and 3B.

## Build commands

Build every lecture source currently present:

```bash
make lectures
```

List the detected decks:

```bash
make lecture-list
```

Build or preview one lecture:

```bash
make html DECK=Lecture_1B
quarto preview lectures/Lecture_1B/Lecture_1B.qmd --port 4188 --host 127.0.0.1
```

Render and open one lecture in the default browser:

```bash
make open-html DECK=Lecture_3A
```

Decks with live Julia cells list their Pluto notebooks in `lectures/<DECK>/pluto.toml`.
Start their server (port 1243) before presenting; until it answers, those widgets
show the start command:

```bash
make start-julia-server DECK=Lecture_3A
```

Build the lecture PDFs from inside this `slides/` directory:

```bash
make pdf DECK=Lecture_1A
make pdf DECK=Lecture_1B
```

Or run the same builds from the repository root:

```bash
make -C slides pdf DECK=Lecture_1A
make -C slides pdf DECK=Lecture_1B
```

The generated files live inside each lecture workspace, for example
`lectures/Lecture_1B/Lecture_1B.pdf`. The PDF target first renders the Reveal.js HTML,
captures each slide at 1920x1080 with headless Chrome, and combines the
captures into a 16:9 PDF.

PDF export remains available with `make pdf DECK=Lecture_1B`.

The deck template uses a 1920×1080 Reveal.js canvas with native MathML and no
CDN requirement.

## Requirements

- [Quarto](https://quarto.org/)
- For PDF: Google Chrome, ImageMagick (`magick`), Python 3

## Adding or migrating a lecture

1. Create `lectures/<deck-id>/` and copy `templates/Lecture_TEMPLATE.qmd` into it using the deck's ID.
2. Create local `figures/`, `widgets/`, and `notebooks/` directories as needed.
3. Reconstruct slide text, equations, tables, and layouts as native Quarto.
4. Retain raster files only when they are genuine photographs, plots, or source illustrations.
5. Update the deck's status in `lectures.yml`.
6. Render the HTML and inspect every slide at 1920×1080 before committing.

The old source presentations stay under `previous_lectures/` as migration
references. They are not build inputs for a native deck.
