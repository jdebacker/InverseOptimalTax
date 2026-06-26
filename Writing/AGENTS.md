# AGENTS.md

## Project Goal

This project is writing an economics paper that uses inverse optimal tax methods to infer the political weights that U.S. presidential candidates place on taxpayers across the income distribution. The goal is to produce a publishable paper for a peer-reviewed journal such as the *National Tax Journal*.

Treat the work as an academic manuscript, not just a LaTeX cleanup task. Edits should improve clarity, credibility, precision, and journal fit.

## Main Files

- `main.tex` is the primary draft of the paper. It currently contains the preamble inline; the old `\input{preamble.tex}` line is commented out.
- `references.bib` contains bibliography entries.
- `images/` contains figures used by the paper.
- `slides/` contains presentation materials and should generally be left alone unless the user asks.
- Build artifacts such as `main.aux`, `main.bbl`, `main.blg`, `main.fdb_latexmk`, `main.fls`, `main.out`, `main.pdf`, and `main.synctex.gz` may appear locally after compilation.

## Paper Content

The paper studies major-party U.S. presidential candidates in the 2012, 2016, 2020, and 2024 elections. It maps candidate tax proposals into marginal tax rate schedules using Tax-Calculator, combines those schedules with an estimated income distribution and ETI assumptions, and recovers implied political weights using inverse optimal tax formulas.

Current core sections:

- Introduction
- Inverse Optimal Tax Approach
- Calibration
- Tax Policies of Presidential Candidates
- Political Weights of Recent Presidential Candidates
- Robustness
- Inverting the inverse
- Conclusions
- Appendix

The candidate-policy section in the main text now uses a compact table. Full modeled policy lists are preserved in `\subsection{Candidate Policy Details}` in the appendix.

## Writing Priorities

- Target the tone and structure of an applied public finance journal article.
- Keep the interpretation careful: the recovered objects are political weights implied by candidate platforms under the model, not direct estimates of candidates' true social welfare functions.
- Avoid overclaiming. When a result depends on ETI assumptions, proxy platforms, or omitted tax bases, say so directly.
- Prefer quantitative statements over vague phrasing when results are available.
- Move detailed implementation notes, long policy lists, and mechanical derivations to tables or appendices when possible.
- Remove draft-like language before final circulation, except for any TODOs the user intentionally leaves.

## Terminology

- Use `taxpayers` or `tax units` for the unit of analysis, not `households` or generic `individuals`, unless discussing a cited paper or a legal/statutory term.
- Preserve statutory tax terms such as `individual income tax`, `individual AMT`, `individual provisions of the TCJA`, and `head-of-household`.
- Use `Democratic candidates`, not `Democrat candidates`.
- The prompt once mentioned 2026, but the paper covers 2012, 2016, 2020, and 2024.

## Known Substantive Caveats

- Obama 2012 and Trump 2020 rely on policy proxies rather than detailed campaign platforms. These should remain clearly labeled as proxies.
- The model focuses on federal marginal tax rates on labor income. Business tax provisions, wealth taxes, and proposals without observable tax bases in the data are excluded.
- Tip-income exemptions proposed in 2024 are omitted because tip income is not separately identified in the data.
- The right tail of the income distribution and income-varying ETI assumptions are important robustness issues.
- There is a TODO in the political weights/results section about expanding the discussion or adding a decomposition plot. The user intentionally left at least one TODO as a reminder.

## LaTeX and Verification

Useful checks:

```sh
git diff --check -- main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

A single `pdflatex` pass may show unresolved citation and reference warnings if BibTeX has not been run. That is not necessarily a source error.

Tables that begin rows with bracketed interval notation like `[\$0, \$37,500)` can be misread by LaTeX as optional arguments after `booktabs` rules. Wrap such interval entries in braces, for example `{[\$0, \$37,500)}`.

## Collaboration Notes

- Always check `git status --short` before editing. The working tree may contain user changes, generated files, and untracked research materials.
- Do not revert or delete unrelated changes.
- Use scoped edits. For writing tasks, preserve the economics and citations unless the user asks for substantive revision.
- If adding new citations or claims, verify that `references.bib` has the corresponding entry.
