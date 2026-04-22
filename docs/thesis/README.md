# Thesis LaTeX Source

This directory contains the thesis source, generated PDF, bibliography, and the figures required by `main.tex`.

## Required TeX Environment

Use a TeX Live installation with XeLaTeX and BibTeX. The document depends on these LaTeX packages:

- `ctex`
- `geometry`
- `amsmath`, `amssymb`, `amsthm`, `mathrsfs`
- `xcolor`
- `hyperref`
- `caption`, `subcaption`, `bicaption`
- `graphicx`
- `float`
- `tikz`
- `longtable`, `booktabs`, `threeparttable`, `array`, `multirow`
- `algorithm2e`
- `listings`
- `titlesec`
- `gbt7714`

On Ubuntu/Debian, a practical system package setup is:

```bash
sudo apt-get install texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-science latexmk
```

## Build

From this directory:

```bash
latexmk main.tex
```

The generated PDF is written to `build/main.pdf`. To reproduce the checked-in root PDF manually, run:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## Files

- `main.tex`: thesis source
- `main.pdf`: checked-in PDF snapshot
- `reference.bib`: bibliography database
- `figures/`: figures referenced by `main.tex`
