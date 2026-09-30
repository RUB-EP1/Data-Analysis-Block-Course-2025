import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build an agent

    **Lecture 4B companion.** An agent is a language model in a loop: the model asks for a tool,
    a program (the *harness*) runs it and appends the result, and the model decides the next step.
    Here we write the whole thing ourselves: a tiny project with a bug, five tools, a harness that
    checks every edit, and the loop.

    Install once, from the `Lecture_4B` folder:

    ```bash
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/marimo edit Build_an_agent.py
    ```

    **Two modes.** *Scripted* replays a fixed sequence of model replies and needs nothing else — use it
    to study the mechanics. *Live* calls Claude through the Anthropic API; it needs credentials
    (`ANTHROPIC_API_KEY`, or a profile from `ant auth login`) and costs a few cents per run.
    """)
    return


@app.cell
def _():
    import json
    import re
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    return Path, SimpleNamespace, json, re, shutil, subprocess, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A small project with a bug

    Relativistic kinematics in natural units: $E^2=m^2+p^2$, so the invariant mass is
    $m=\sqrt{E^2-p^2}$. The code has a sign error, and one test fails.
    """)
    return


@app.cell
def _(Path, mo, shutil):
    FILES = {
        "kinematics.py": '''"""Relativistic kinematics in natural units (c = 1); energies and momenta in GeV."""
import math


def energy(m, p):
    """Energy of a particle with mass m and momentum p."""
    return math.sqrt(m**2 + p**2)


def invariant_mass(E, p):
    """Invariant mass of a system with total energy E and total momentum p."""
    return math.sqrt(E**2 + p**2)
''',
        "test_kinematics.py": '''from kinematics import energy, invariant_mass


def test_energy():
    assert abs(energy(0.938, 0.0) - 0.938) < 1e-9


def test_invariant_mass():
    # a J/psi-like system: E = 3.2 GeV, p = 0.8 GeV, so m = sqrt(9.6) = 3.0984 GeV
    assert abs(invariant_mass(3.2, 0.8) - 3.0984) < 1e-3


if __name__ == "__main__":
    failed = 0
    for test in (test_energy, test_invariant_mass):
        try:
            test()
            print("PASSED", test.__name__)
        except AssertionError:
            failed += 1
            print("FAILED", test.__name__)
    raise SystemExit(failed)
''',
    }
    WORKSPACE = Path(mo.notebook_dir()) / "agent_workspace"

    def reset_workspace():
        """A fresh copy of the project for every run."""
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        WORKSPACE.mkdir()
        for name, content in FILES.items():
            (WORKSPACE / name).write_text(content)

    reset_workspace()
    mo.md(f"Project written to `{WORKSPACE}`:\n\n```python\n{FILES['kinematics.py']}```")
    return FILES, WORKSPACE, reset_workspace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Tools and harness

    Each tool is a Python function plus a JSON description that the model reads. The harness adds three things
    the model does not control:

    - **sandbox**: tools only see files inside the workspace;
    - **safe editing**: `str_replace` changes text only if `old_string` occurs **exactly once**;
    - **validation**: after every edit the harness compiles the file and appends the result — a linter the model cannot forget to run.
    """)
    return


@app.cell
def _(WORKSPACE, re, subprocess, sys):
    def inside(path):
        p = (WORKSPACE / path).resolve()
        if WORKSPACE.resolve() not in p.parents and p != WORKSPACE.resolve():
            raise ValueError(f"{path} is outside the workspace")
        return p

    def list_files():
        return "\n".join(sorted(p.name for p in WORKSPACE.iterdir() if p.is_file()))

    def read_file(path):
        lines = inside(path).read_text().splitlines()
        return "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(lines))

    def grep(pattern):
        hits = []
        for f in sorted(WORKSPACE.glob("*.py")):
            for i, line in enumerate(f.read_text().splitlines()):
                if re.search(pattern, line):
                    hits.append(f"{f.name}:{i + 1}: {line.strip()}")
        return "\n".join(hits) or "no matches"

    def lint(path):
        """Harness check: does the file still compile?"""
        r = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True)
        return "lint: OK" if r.returncode == 0 else "lint: FAILED\n" + r.stderr.strip().splitlines()[-1]

    def str_replace(path, old_string, new_string):
        p = inside(path)
        text = p.read_text()
        n = text.count(old_string)
        if n != 1:
            raise ValueError(f"old_string found {n} times in {path}; it must occur exactly once. Add more context.")
        p.write_text(text.replace(old_string, new_string))
        return f"Edited {path}.\n" + lint(p)

    def run_tests():
        r = subprocess.run([sys.executable, "test_kinematics.py"], cwd=WORKSPACE, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip() + f"\n(exit code {r.returncode})"

    TOOL_FUNCTIONS = {"list_files": list_files, "read_file": read_file, "grep": grep,
                      "str_replace": str_replace, "run_tests": run_tests}

    def obj(**props):
        return {"type": "object", "properties": props, "required": list(props)}

    TOOLS = [
        {"name": "list_files", "description": "List the files in the project.", "input_schema": obj()},
        {"name": "read_file", "description": "Read a file, with line numbers.", "input_schema": obj(path={"type": "string"})},
        {"name": "grep", "description": "Search all Python files for a regular expression.", "input_schema": obj(pattern={"type": "string"})},
        {"name": "str_replace", "description": "Replace old_string by new_string in a file. old_string must occur exactly once; include enough surrounding text to make it unique.",
         "input_schema": obj(path={"type": "string"}, old_string={"type": "string"}, new_string={"type": "string"})},
        {"name": "run_tests", "description": "Run the test suite.", "input_schema": obj()},
    ]

    def run_tool(call):
        """Execute one tool_use block and return the tool_result block for the model."""
        try:
            fn = TOOL_FUNCTIONS[call.name]
            out, err = fn(**call.input), False
        except Exception as e:                   # errors go back to the model, they do not crash the loop
            out, err = f"Error: {e}", True
        return {"type": "tool_result", "tool_use_id": call.id, "content": out, "is_error": err}

    return TOOLS, run_tool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Two models behind the same interface

    Both take the message history and return a reply whose `content` is a list of blocks:
    `thinking`, `text`, or `tool_use`. The scripted model ignores the history and replays a prepared
    trajectory — including two mistakes that the harness catches.
    """)
    return


@app.cell
def _(SimpleNamespace, TOOLS):
    SYSTEM = ("You are a coding agent working in a small Python project. Use the tools to inspect, edit and test "
              "the code. Make the smallest change that fixes the problem, run the tests to confirm, then reply with "
              "a two-sentence summary.")

    def block(type, **kw):
        return SimpleNamespace(type=type, **kw)

    def call(i, name, **inp):
        return block("tool_use", id=f"call_{i}", name=name, input=inp)

    SCRIPT = [
        [block("thinking", thinking="Start by running the tests to see what fails."), call(1, "run_tests")],
        [block("thinking", thinking="test_invariant_mass fails. Find the function."), call(2, "grep", pattern="def invariant_mass")],
        [call(3, "read_file", path="kinematics.py")],
        [block("thinking", thinking="m² = E² − p², but line 12 adds p². Replace the plus by a minus."),
         call(4, "str_replace", path="kinematics.py", old_string="+ p**2", new_string="- p**2")],
        [block("thinking", thinking="`+ p**2` also occurs in `energy()`. Use the whole expression."),
         call(5, "str_replace", path="kinematics.py", old_string="math.sqrt(E**2 + p**2)", new_string="math.sqrt(E**2 - p**2")],
        [block("thinking", thinking="The linter says the parenthesis is not closed. Fix my edit."),
         call(6, "str_replace", path="kinematics.py", old_string="math.sqrt(E**2 - p**2", new_string="math.sqrt(E**2 - p**2)")],
        [call(7, "run_tests")],
        [block("text", text="invariant_mass added p² instead of subtracting it; it now returns √(E² − p²). "
                            "Both tests pass.")],
    ]

    def scripted_model(messages):
        step = sum(1 for m in messages if m["role"] == "assistant")
        content = SCRIPT[min(step, len(SCRIPT) - 1)]
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=None)

    def live_model(messages):
        import anthropic
        client = anthropic.Anthropic()          # credentials from ANTHROPIC_API_KEY or `ant auth login`
        return client.beta.messages.create(
            model="claude-opus-5",
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                 # a declined request is retried on a fallback model
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "medium"},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

    return live_model, scripted_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The loop

    This is the whole agent. Everything the model “remembers” is the `messages` list, which is sent in full on every call.
    """)
    return


@app.cell
def _(run_tool):
    def agent_loop(task, model, max_steps=12, log=print):
        messages = [{"role": "user", "content": task}]
        for step in range(max_steps):
            reply = model(messages)                                   # one call to the language model
            messages.append({"role": "assistant", "content": reply.content})
            log(step, "model", reply)
            if reply.stop_reason in ("refusal", "max_tokens"):
                break
            calls = [b for b in reply.content if b.type == "tool_use"]
            if not calls:                                             # no tool requested: the model is done
                break
            results = [run_tool(c) for c in calls]                    # the harness acts
            messages.append({"role": "user", "content": results})     # results go back into the context
            log(step, "tools", results)
        return messages

    return (agent_loop,)


@app.cell
def _(mo):
    mode = mo.ui.dropdown(["scripted", "live (Claude API)"], value="scripted", label="Model")
    task = mo.ui.text_area(value="The tests in test_kinematics.py fail. Find and fix the bug.", full_width=True, rows=2, label="Task")
    max_steps = mo.ui.slider(2, 20, value=12, label="Max steps", show_value=True)
    go = mo.ui.run_button(label="Run the agent")
    mo.vstack([task, mo.hstack([mode, max_steps, go], justify="start", gap=2)])
    return go, max_steps, mode, task


@app.cell
def _(agent_loop, go, json, live_model, max_steps, mo, mode, reset_workspace, scripted_model, task):
    mo.stop(not go.value, mo.md("*Press “Run the agent”.*"))
    reset_workspace()
    _cards, _usage = [], {"in": 0, "out": 0}

    def _fmt(x):
        return x if isinstance(x, str) else json.dumps(x)

    def _log(step, who, payload):
        if who == "model":
            u = getattr(payload, "usage", None)
            if u is not None:
                _usage["in"] += u.input_tokens; _usage["out"] += u.output_tokens
            for b in payload.content:
                if b.type == "thinking" and b.thinking:
                    _cards.append(mo.md(f"**{step + 1} · thinking** — *{b.thinking}*"))
                elif b.type == "text":
                    _cards.append(mo.md(f"**{step + 1} · answer** — {b.text}"))
                elif b.type == "tool_use":
                    _cards.append(mo.md(f"**{step + 1} · tool call** `{b.name}({_fmt(b.input)})`"))
            if payload.stop_reason == "refusal":
                _cards.append(mo.md(f"**refused** — {getattr(payload, 'stop_details', '')}"))
        else:
            for r in payload:
                flag = "error" if r["is_error"] else "result"
                _cards.append(mo.md(f"**{step + 1} · {flag}**\n```\n{r['content']}\n```"))

    _model = scripted_model if mode.value == "scripted" else live_model
    try:
        transcript = agent_loop(task.value, _model, max_steps.value, _log)
        _summary = f"{len(transcript)} messages" + (f", {_usage['in']:,} input and {_usage['out']:,} output tokens billed" if _usage["in"] else "")
    except Exception as e:                       # e.g. no credentials in live mode
        transcript, _summary = [], (f"Stopped: {type(e).__name__}: {e}. Live mode needs credentials: "
                                    "set ANTHROPIC_API_KEY or run `ant auth login`, then restart the notebook.")
    mo.vstack(_cards + [mo.md(f"*{_summary}*")])
    return (transcript,)


@app.cell
def _(FILES, WORKSPACE, mo, transcript):
    import difflib
    _new = (WORKSPACE / "kinematics.py").read_text()
    _diff = "".join(difflib.unified_diff(FILES["kinematics.py"].splitlines(True), _new.splitlines(True), "before", "after"))
    mo.md(f"### What changed on disk\n\n```diff\n{_diff or '(nothing)'}\n```\n\n{len(transcript)} messages were sent back and forth.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. In scripted mode, find the two steps where the harness corrected the model. Which check caught which mistake?
    2. Remove the `lint(p)` call from `str_replace`. What happens at step 6 of the scripted run, and when would the model notice?
    3. Change `str_replace` to replace *all* occurrences. Why is that dangerous here?
    4. Live mode: give a vaguer task (“improve the code”). How many steps does the model take? What does it change?
    5. Add a tool `write_file`. Should the harness allow overwriting the test file? Add a rule that forbids it.
    6. The whole history is resent on every call. Using the token counts printed in live mode, estimate the cost of a 50-step session.
    """)
    return


if __name__ == "__main__":
    app.run()
