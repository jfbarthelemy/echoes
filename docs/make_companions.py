#!/usr/bin/env python3
"""Generate Jupyter notebooks and companion Python scripts from QMD source files.

For each chapter QMD the script produces:
  - CHAPTER.py    — plain Python script (code cells only, Quarto options kept)
  - CHAPTER.ipynb — cleaned Jupyter notebook (prose + code, Quarto syntax stripped,
                    custom LaTeX macros expanded inline)

Usage (from the echoes book root):
    python make_companions.py
"""

import json
import os
import re
import subprocess
import glob

# ---------------------------------------------------------------------------
# Quarto cell options to remove from notebook code cells
# ---------------------------------------------------------------------------
DROP_OPTS = {
    'code-fold', 'code-summary', 'warning', 'error', 'message',
    'include', 'echo', 'fig-width', 'fig-height', 'fig-cap',
    'fig-subcap', 'fig-align', 'fig-pos', 'fig-env',
    'label', 'out-width', 'tbl-cap', 'output',
}

# ---------------------------------------------------------------------------
# Custom LaTeX macro definitions for inline expansion in markdown cells.
# Macros are expanded at build time so notebooks work in every renderer
# (KaTeX, MathJax 2, MathJax 3) without any cross-cell state sharing.
# ---------------------------------------------------------------------------

# Zero-argument macros: command name (without \) -> expansion
MACROS_0 = {
    'C':        r'\mathbb{C}',
    'R':        r'\mathbb{R}',
    'Q':        r'\mathbb{Q}',
    'Z':        r'\mathbb{Z}',
    'N':        r'\mathbb{N}',
    'x':        r'\underline{x}',
    'n':        r'\underline{n}',
    'eps':      r'\boldsymbol{\varepsilon}',
    'E':        r'\boldsymbol{E}',
    'sig':      r'\boldsymbol{\sigma}',
    'Sig':      r'\boldsymbol{\Sigma}',
    'cod':      r'\underline{\mathcal{b}}',
    'sotimes':  r'\stackrel{s}{\otimes}',
    'sboxtimes': r'\stackrel{s}{\boxtimes}',
    'ud':       r'\,\mathrm{d}',
    'dcirc':    r'\overset{\circ}{:}',
    'arcosh':   r'\operatorname{arcosh}',
    'divz':     r'\operatorname{div}',
    'divu':     r'\underline{\operatorname{div}}',
    'hess':     r'\operatorname{hess}',
    'gradu':    r'\underline{\operatorname{grad}}',
    'graduu':   r'\boldsymbol{\operatorname{grad}}',
    'Mat':      r'\operatorname{Mat}',
    'tr':       r'\operatorname{tr}',
    'ISO':      r'\operatorname{ISO}',
    'mat':      r'\mathsf',       # takes following {arg} naturally
}

# One-argument macros: command name (without \) -> expansion template (#1 = arg)
MACROS_1 = {
    'uu':    r'\boldsymbol{#1}',
    'uuuu':  r'\mathbb{#1}',
    'uv':    r'\underline{#1}',
    've':    r'\underline{e}_{#1}',
    'trans': r'{}^{t}{#1}',
    'norm':  r'\lVert #1 \rVert',
    'volt':  r'{#1}^{-1\circ}',
    'jump':  r'\llbracket #1 \rrbracket',
}


def _match_brace(s: str, start: int) -> int:
    """Return index of the } closing the { at s[start]."""
    depth = 0
    i = start
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s) - 1


def _expand_math(text: str) -> str:
    """Expand all custom macros inside a single math expression (multiple passes)."""
    for _ in range(8):
        prev = text

        # One-argument macros (longest name first to avoid prefix conflicts)
        for name in sorted(MACROS_1, key=len, reverse=True):
            tmpl = MACROS_1[name]
            pat = re.compile(r'\\' + re.escape(name) + r'(?![a-zA-Z*@])\s*\{')
            while True:
                m = pat.search(text)
                if not m:
                    break
                brace_open = m.end() - 1
                brace_close = _match_brace(text, brace_open)
                arg = text[brace_open + 1:brace_close]
                replacement = tmpl.replace('#1', arg)
                text = text[:m.start()] + replacement + text[brace_close + 1:]

        # Zero-argument macros (longest name first)
        for name in sorted(MACROS_0, key=len, reverse=True):
            exp = MACROS_0[name]
            text = re.sub(
                r'\\' + re.escape(name) + r'(?![a-zA-Z*@])',
                lambda m, e=exp: e,   # lambda avoids re interpreting backslashes
                text,
            )

        if text == prev:
            break
    return text


def expand_macros_in_markdown(md: str) -> str:
    """Expand custom LaTeX macros inside all math regions of a markdown string."""
    # Protect code spans/blocks with placeholders so we don't touch them
    saved = []

    def save(m):
        saved.append(m.group(0))
        return f'\x00SAVED{len(saved) - 1}\x00'

    md = re.sub(r'```[\s\S]*?```', save, md)   # fenced code blocks
    md = re.sub(r'`[^`\n]+`', save, md)         # inline code spans

    # Expand macros inside display math  $$...$$
    md = re.sub(
        r'\$\$([\s\S]*?)\$\$',
        lambda m: '$$' + _expand_math(m.group(1)) + '$$',
        md,
    )

    # Expand macros inside inline math  $...$
    # Exclude cases like $$ (already handled) and lone $ in plain text
    md = re.sub(
        r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)',
        lambda m: '$' + _expand_math(m.group(1)) + '$',
        md,
    )

    # Restore saved code spans
    for i, code in enumerate(saved):
        md = md.replace(f'\x00SAVED{i}\x00', code)

    return md


# ---------------------------------------------------------------------------
# Markdown cleaner (Quarto-specific syntax)
# ---------------------------------------------------------------------------
def clean_markdown(text: str) -> str:
    """Strip Quarto-specific syntax from a markdown cell."""
    text = re.sub(r'^\s*:::\s*\{[^}]*\}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*:::\s*$',             '', text, flags=re.MULTILINE)
    text = re.sub(r'\{\{<[^>]+>\}\}\s*', '', text)
    text = re.sub(r'\s*\{[#.][^}]*\}', '', text)
    text = re.sub(r'@sec-[\w-]+', '[§]',   text)
    text = re.sub(r'@fig-[\w-]+', '[fig]', text)
    text = re.sub(r'@eq-[\w-]+',  '[eq]',  text)
    text = re.sub(r'@tbl-[\w-]+', '[tbl]', text)
    text = re.sub(r'@lst-[\w-]+', '[lst]', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Code cell cleaner
# ---------------------------------------------------------------------------
def clean_code_cell(source_lines: list) -> list:
    """Remove Quarto cell options; flag eval:false cells with a comment."""
    is_eval_false = False
    cleaned = []
    for line in source_lines:
        if line.startswith('#|'):
            key = line[2:].split(':')[0].strip()
            if key == 'eval' and 'false' in line:
                is_eval_false = True
            continue
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    if is_eval_false:
        note = ('# NOTE: This cell is not evaluated by default '
                '(long computation — run manually to reproduce).\n')
        cleaned.insert(0, note)
    return cleaned


# ---------------------------------------------------------------------------
# Notebook post-processor
# ---------------------------------------------------------------------------
def strip_yaml_frontmatter(text: str) -> str:
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            text = text[end + 4:].lstrip('\n')
    return text


def process_notebook(ipynb_path: str) -> None:
    """Post-process a quarto-converted notebook for clean Jupyter use."""
    with open(ipynb_path, encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    first_markdown = True

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if first_markdown:
                src = strip_yaml_frontmatter(src)
                first_markdown = False
            src = clean_markdown(src)
            src = expand_macros_in_markdown(src)
            if src.strip():
                cell = dict(cell)
                cell['source'] = [src]
                new_cells.append(cell)

        elif cell['cell_type'] == 'code':
            src = clean_code_cell(cell['source'])
            if src:
                cell = dict(cell)
                cell['source'] = src
                cell['outputs'] = []
                cell['execution_count'] = None
                new_cells.append(cell)

    nb['cells'] = new_cells
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    nb['metadata']['language_info'] = {
        'name': 'python',
        'version': '3.11.0',
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    qmds = sorted(glob.glob('**/*.qmd', recursive=True))
    skip = {'docs', '_extensions'}
    qmds = [q for q in qmds
            if os.sep in q and q.split(os.sep)[0] not in skip]

    ok = 0
    for qmd in qmds:
        ipynb = qmd.replace('.qmd', '.ipynb')

        # Step 1: QMD → raw ipynb
        r1 = subprocess.run(
            ['quarto', 'convert', qmd, '--output', ipynb],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            print(f'FAIL quarto convert : {qmd}')
            print(f'  {r1.stderr[:300]}')
            continue

        # Step 2: raw ipynb → .py  (before cleaning so #| opts survive as comments)
        r2 = subprocess.run(
            ['jupytext', '--to', 'script', ipynb],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            print(f'FAIL jupytext       : {ipynb}')

        # Step 3: post-process ipynb (strip Quarto syntax, expand macros)
        process_notebook(ipynb)

        print(f'OK  {qmd}')
        ok += 1

    print(f'\n{ok}/{len(qmds)} chapters processed.')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--notebook':
        # Single-notebook mode: expand macros in an existing .ipynb written by hand.
        #
        # Use this when you write a new notebook manually using the custom macro
        # shorthands (\sig, \uu{...}, \eps, \tr, etc.) instead of plain LaTeX.
        # Running this command expands all macros inline so the notebook renders
        # correctly in any environment (KaTeX in VS Code, MathJax 3 in JupyterLab,
        # nbconvert, …) without any renderer configuration.
        #
        # Usage:
        #   python make_companions.py --notebook path/to/my_notebook.ipynb
        #
        # The file is modified in place. Quarto-specific syntax (:::, {{< >}}, …)
        # is also stripped if present. The .py companion is NOT regenerated here
        # (it is only produced from a QMD source via the default mode).
        process_notebook(sys.argv[2])
        print(f'OK  {sys.argv[2]}')
    else:
        # Default mode: regenerate all .ipynb and .py companions from QMD sources.
        #
        # Usage (from the echoes book root):
        #   python make_companions.py
        main()
