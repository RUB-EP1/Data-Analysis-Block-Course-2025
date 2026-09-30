# Lecture 4B: LLM chats and agents

`Lecture_4B.qmd` is the native Quarto source (40 slides). It follows Lecture 4A
(transformers) and covers:

- training: pre-training data (Common Crawl, FineWeb), the training stages,
  reinforcement learning (policy gradient, GRPO), where rewards come from, and the
  controversies (labelling labour, copyright, sycophancy)
- chatbots: the chat template, statelessness, the context window
- thinking: chain of thought, reasoning models, the DeepSeek-R1-Zero "aha moment",
  and "Wait" budget forcing (s1)
- agents: tools, what a harness is and its parts (validation, search, editing and
  harness insertions), then the agent loop
- safety (section): prompt injection, Claude Code permission modes and rules, the
  `/sandbox` OS sandbox, dev containers and cloud VMs
- flavours (section): model families with logos (Claude, GPT/Codex, Gemini, Grok,
  open weights) and interfaces (chat, CLI, desktop, IDE, cloud)
- scaling (section): METR time horizons, and four axes of scaling (size and data,
  RL compute, test-time compute, number of agents)
- backup: prefill/decode latency and cost, "dreaming" and recursive self-improvement

Product names and prices are as of mid-2026 and will date quickly.

## Present

From the repository root:

```sh
make -C slides html DECK=Lecture_4B
quarto preview slides/lectures/Lecture_4B/Lecture_4B.qmd --port 4196
make -C slides pdf DECK=Lecture_4B
```

## Marimo notebook: Build an agent

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/marimo edit Build_an_agent.py
```

A kinematics module with a sign error, five sandboxed tools, a harness that lints
after every edit, and the agent loop in ~15 lines. *Scripted* mode replays a fixed
trajectory offline (the harness catches an ambiguous edit and a syntax error).
*Live* mode calls `claude-opus-5` through the Anthropic SDK (adaptive thinking with
summaries, medium effort, server-side refusal fallbacks); it needs
`ANTHROPIC_API_KEY` or an `ant auth login` profile and costs a few cents per run.
The live path has been checked to fail gracefully without credentials but has not
been run against the API. Static export: `notebooks/exports/Build_an_agent.html`.

## Four interactive widgets

1. **chat.html** — a conversation serialized into one token stream; context fill and
   cumulative input tokens over many turns, with and without prompt caching.
2. **latency.html** (backup) — timeline and cost of one request (network, prefill, hidden
   thinking, answer) for four scenarios; Claude API list prices, mid-2026.
3. **wait.html** — constructed reasoning traces that can be stopped or extended with
   "Wait" (budget forcing); labelled as illustrative.
4. **agent.html** — replay of real, condensed steps from the Claude Code session that
   built the AlexNet notebook of Lecture 3B, including the two bugs found by checks.

Posters: `zsh scripts/make_posters.zsh`.

## Figures

- `scripts/make_data.py` (run with `../Lecture_4A/.venv/bin/python`): GPT-2 base-model
  answers, prefill vs decode timing on the local CPU (minimum of repeats; numbers
  depend on the machine), context-window history, and `figures/results.json`.
- `scripts/make_diagrams.py` (no dependencies): training stages, RL loop, agent loop, harness.
- `scripts/make_scaling.py` (Lecture 4A venv): METR time horizons from
  `figures/metr-horizons.csv` (Time Horizon 1.1, copied from metr.org/time-horizons,
  May 2026) and published pre-training token counts.
- `figures/logos/`: company logos from LobeHub icons (`@lobehub/icons-static-svg`, MIT).
