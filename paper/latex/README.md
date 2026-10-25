# Paper draft (LaTeX)

ACL format (`acl.sty`, `acl_natbib.bst` from github.com/acl-org/acl-style-files, unmodified).

    # Debian/Ubuntu: texlive-latex-extra texlive-fonts-recommended texlive-pictures latexmk
    cd paper/latex && latexmk -pdf main.tex

The figure reads `figures/*.dat`, written by `uv run python scripts/paper/figure_tex.py`.
Every number in the tables was transcribed from `docs/results-paper.md`,
`docs/results-phase4.md` and `docs/results-phase7.md`, which the scripts named in each
table's source comment regenerate from committed runs. If a run in
`scripts/paper/runs.py` changes, update the table by hand.

`refs.bib` was generated on 2026-10-25 from arXiv metadata and ACL Anthology BibTeX
(author lists and titles as published). A venue is given only where arXiv or the
Anthology confirms it; the others are cited as arXiv preprints. Upgrade them if the
final versions have appeared.

Before submission:

- Choose the venue and check its page limit; the main text is 7 pages.
- Fill in the author block and switch `[review]` to `[final]` or `[preprint]` when the
  venue allows it (keep `[review]` for anonymous submission).
- Re-run the related-work search if more than a few weeks have passed (last run
  2026-10-25; see `paper/RELATED_WORK.md`).
