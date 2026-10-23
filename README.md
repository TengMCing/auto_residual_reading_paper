# Automated Assessment of Residual Plots with Computer Vision Models

Weihao Li, Dianne Cook, Emi Tanaka, Susan VanderPlas, Klaus Ackermann

This repository contains the manuscript, code, data, and cached results for the
paper *Automated Assessment of Residual Plots with Computer Vision Models*.

The work trains a computer vision model to automate the lineup protocol for
visual inference. Instead of asking human judges to pick the real data plot from
a field of decoys, the model scores residual plots directly, predicting a
distance that quantifies how far a residual distribution departs from the
reference distribution under a classical normal linear regression model. The
model is less sensitive than conventional residual tests but more sensitive than
human visual inference, and it scales to large numbers of plots.

## Repository layout

The repository root contains this README and the [`jcgs/`](jcgs/) folder, which
holds all project files.

| Path | Description |
| --- | --- |
| `jcgs/paper.Rmd` | Manuscript source (R Markdown); knits to `paper.pdf` |
| `jcgs/paper.pdf` | Compiled manuscript |
| `jcgs/paper.tex`, `jcgs/paper.aux`, etc. | LaTeX build outputs |
| `jcgs/appendix.Rmd`, `jcgs/appendix.pdf` | Supplementary appendix (data generation, network layers, model training, violation index) |
| `jcgs/paper.R` | R code extracted from `paper.Rmd` (`knitr::purl`) |
| `jcgs/bibliography.bib` | References |
| `jcgs/data/` | Data used in the analyses (see below) |
| `jcgs/cached_data/` | Precomputed model outputs and diagnostics used by the manuscript |
| `jcgs/figures/` | Figure assets |
| `jcgs/paper_files/`, `jcgs/appendix_files/` | Figures generated when knitting |
| `jcgs/cover_letter/` | Cover letter source and PDF |
| `jcgs/revise_decision/` | Response-to-reviewers documents |
| `jcgs/diff1.*`, `jcgs/diff2.*` | Marked-up manuscript versions |
| `jcgs/paper-submission1.tex`, `jcgs/paper-submission2.tex` | Earlier submission versions |
| `jcgs/arxiv_version_2024_11_2/` | Archived arXiv version |
| `jcgs/interact.cls`, `jcgs/tfcad.bst` | LaTeX class and bibliography style |
| `jcgs/jcgs.Rproj` | RStudio project file |

## Data

| File | Description |
| --- | --- |
| `data/meta.csv` | Simulation metadata for the training/test images: design factors, effect size, and plot identifiers |
| `data/train_pred.csv`, `data/test_pred.csv` | Per-image model predictions (`filename`, `vss`, `res`, `plot_uid`) |
| `data/vss_32.rds` | Lineup-level model scores/p-values for the human-subject experiment (1,176 lineups) |
| `data/lineup_vss.rds` | Per-plot scores and ranks within each lineup (23,520 rows) |
| `data/actual_ss.rds` | True effect size for each lineup |
| `data/housing.csv` | Boston housing extract (`RM`, `LSTAT`, `PTRATIO`, `MEDV`) used in the Boston housing example |

`cached_data/` holds precomputed results (attention maps, checks for the
left-triangle, Boston housing, DatasauRus, and Figure 1 examples) so the
manuscript can be rebuilt without re-running training.

## Reproducing the manuscript

1. Open `jcgs/jcgs.Rproj` in RStudio.
2. Install the R packages used by the document. The setup chunk in `paper.Rmd`
   installs most of them automatically: `tidyverse`, `glue`, `remotes`,
   `reticulate`, `ggmosaic`, `here`, `knitr`, `yardstick`, `kableExtra`,
   `autovi`, `patchwork`, and `datasauRus`. The visual inference package
   `visage` is installed from GitHub (`TengMCing/visage`).
3. Set up Python for `reticulate`. The manuscript uses a conda environment named
   `auto_residual_reading_paper` with Python 3.11.9 and TensorFlow (the setup
   code is in the hidden chunk at the top of `paper.Rmd`):

   ```r
   reticulate::use_condaenv("auto_residual_reading_paper")
   ```

   The pretrained model is provided by `autovi::get_keras_model("vss_phn_32")`.
4. Knit `paper.Rmd` to produce `paper.pdf`, and `appendix.Rmd` to produce
   `appendix.pdf`.

Knitr cache directories (such as `paper_cache/`) are deliberately ignored by
git and are regenerated when the documents are knitted.

## Data availability

The data supporting this study are in this repository. The human-subject data
were originally collected and released under an MIT license by Li et al. (2024);
this study uses a secondary analysis of that de-identified, publicly available
data.
