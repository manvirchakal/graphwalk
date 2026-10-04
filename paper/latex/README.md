# Paper draft (LaTeX)

ACL format (`acl.sty`, `acl_natbib.bst` from github.com/acl-org/acl-style-files, unmodified).

    # Debian/Ubuntu: texlive-latex-extra texlive-fonts-recommended texlive-pictures latexmk
    cd paper/latex && latexmk -pdf main.tex

The figure reads `figures/*.dat`, written by `uv run python scripts/paper/figure_tex.py`.
Every number in the tables was transcribed from `docs/results-paper.md`,
`docs/results-phase4.md` and `docs/results-phase7.md`, which the scripts named in each
table's source comment regenerate from committed runs. If a run in
`scripts/paper/runs.py` changes, update the table by hand.

Before submission:

- Fill in the author block and switch `[review]` to `[final]` or `[preprint]`.
- `refs.bib`: entries marked `VERIFY` were written from notes; check titles and author
  lists against the URLs. Re-run the related-work search (`paper/RELATED_WORK.md`).
- Check the venue's page limit; the main text is 7 pages.
