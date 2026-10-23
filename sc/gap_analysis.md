# Statistics and Computing — Gap Analysis of the Current Manuscript Package

> **AI disclaimer:** This gap analysis was drafted with the assistance of an AI
> language model on 23 October 2026. It is based on the files in `jcgs/` and
> publicly available journal guidelines; it is not an official statement from
> the journal or publisher. Verify requirements on the official pages linked
> under Sources before submitting.

**Current package:** `jcgs/` — the manuscript prepared for the *Journal of
Computational and Graphical Statistics* (Taylor & Francis).
**Target journal:** *Statistics and Computing* (Springer).
**Reviewed:** 23 October 2026.

## 1. What is in the `jcgs/` package

### Manuscript and build

| File | What it is |
| --- | --- |
| `paper.Rmd` | Main manuscript source (R Markdown, 110 KB) |
| `paper.R` | R code extracted from `paper.Rmd` (`knitr::purl`) |
| `paper.tex` | Generated LaTeX using **`interact.cls`** (Taylor & Francis) |
| `paper.pdf` | Compiled manuscript, **37 pages**, double-spaced |
| `paper.bbl` | 65 cited references, author–year format (`tfcad.bst`) |
| `paper.aux`, `.log`, `.out` | LaTeX build files |
| `appendix.Rmd`, `appendix.tex`, `appendix.pdf` | Supplementary appendix, **17 pages**, also `interact.cls` |
| `bibliography.bib` | 220 entries in total; 65 are cited in the paper |
| `figures/`, `paper_files/`, `appendix_files/` | Figure assets and knitr-generated figures |
| `data/` | `meta.csv` (23 MB), `train_pred.csv` (8.8 MB), `test_pred.csv` (0.9 MB), `vss_32.rds`, `lineup_vss.rds`, `actual_ss.rds`, `housing.csv` |
| `cached_data/` | 18 precomputed RDS files used by the manuscript |

### JCGS-specific material (not part of an S&C submission)

| File/folder | What it is |
| --- | --- |
| `cover_letter/` | Cover letter addressed to **JCGS** |
| `revise_decision/` | Response-to-reviewers documents |
| `diff1.*`, `diff2.*`, `diff.tex` | Marked-up manuscript versions |
| `paper-submission1.tex`, `paper-submission2.tex` | Earlier submission versions |
| `paper_author_details.pdf`, `paper_with_author_details.pdf`, `paper_copy.pdf` | Duplicate/author-details PDFs |
| `arxiv_version_2024_11_2/`, `.zip` | Archived arXiv version |
| `interact.cls`, `tfcad.bst` | Taylor & Francis class and bibliography style |
| `jcgs.Rproj` | RStudio project |

## 2. Current manuscript form at a glance

| Aspect | Current state |
| --- | --- |
| Template | `rticles::tf_article` → `interact.cls` (Taylor & Francis) |
| Length | 37 pages, double-spaced; appendix 17 pages |
| Abstract | **177 words** |
| Keywords | **10**, stored as one semicolon-separated block |
| Section depth | 10 sections, 12 subsections, 3 subsubsections (3 levels) |
| Figures | **13** included: 11 vector PDFs, 2 PNGs at **72 dpi** |
| Tables | 4 |
| References | 65 cited, author–year, `tfcad.bst`; **no DOI strings** in `paper.bbl`; **5 arXiv preprints** |
| Title page | 5 authors, 4 affiliations (department, university, city, state, country); **no corresponding-author email, no ORCID** |
| Declarations | Acknowledgement (incl. ChatGPT), Data Availability Statement, Disclosure Statement; **no funding statement, no author contributions** |
| Supplementary | Separate `appendix.pdf`, cited as “Appendix” |
| Data/code | GitHub repository link in the Data Availability Statement |

## 3. Gap against *Statistics and Computing* requirements

Priority key: **Must** = required or likely to block/return the submission;
**Should** = strongly recommended; **Pass** = already satisfied;
**Verify** = depends on the submission form or editorial office.

| # | S&C requirement | Current state | Gap and action | Priority |
| --- | --- | --- | --- | --- |
| 1 | LaTeX via the **Springer Nature template** (`sn-jnl.cls`) with the **`iicol`** option | Taylor & Francis `interact.cls` | Switch the R Markdown output to `rticles::springer_article` and rebuild; set `classoptions: [iicol, sn-basic]` | **Must** |
| 2 | Editable source set (`.tex`, class, style files, figures) plus compiled PDF at submission | `.tex` uses `interact.cls`; source set is T&F | Produce and submit an `sn-jnl` source set; include `sn-jnl.cls`, the chosen `.bst`, figures, and the PDF | **Must** |
| 3 | Abstract 150–250 words | 177 words | No change needed | Pass |
| 4 | 4–6 keywords | 10 keywords | Cut to 4–6 and convert the block string to a YAML list | **Must** |
| 5 | Title page with full postal affiliations, corresponding-author email, ORCID where available | Department/university/city/state/country; no email or ORCID | Add a designated corresponding author with email; add ORCIDs; consider street/postcode | **Must** (email); Should (ORCID/postcode) |
| 6 | “Statements and Declarations” covering competing interests, funding, ethics approval/consent, data/material/code availability, author contributions | Separate Acknowledgement, Data Availability, and Disclosure sections; funding and author contributions missing | Consolidate and add the missing funding and author-contribution statements; keep ethics/consent information | **Must** |
| 7 | Data availability statement for all original research | Present | Move under the Statements and Declarations heading | Pass (relocate) |
| 8 | Author–year references | Author–year (`tfcad.bst`) | Content is compatible; restyle with `sn-basic` (author–year) | Pass (restyle) |
| 9 | Give DOIs as full links | No DOI strings in `paper.bbl` | Ensure DOI fields are in `bibliography.bib` and printed by the new style | **Must** (guideline says always when available) |
| 10 | Reference list only published or accepted works | 5 arXiv preprints (TensorFlow, Adam, VGG, Krishnan–Hofmann, Singh et al.) | Replace with published versions where they exist (TensorFlow → OSDI 2016; Adam → ICLR 2015; VGG → ICLR 2015); otherwise cite in text only | **Must** |
| 11 | Figures: EPS for vector art; TIFF for halftones; 1200 dpi line art, 300 dpi halftone, 600 dpi combination | 11 vector PDFs; 2 included PNGs at 72 dpi (`model-performance-1.png`, `over-under-1.png`) | Regenerate the two PNGs at ≥600 dpi (or as vector); convert PDFs to EPS or confirm PDF acceptance; resize for the two-column layout | **Must** (PNGs); Should (PDF→EPS) |
| 12 | Figure/table captions, permissions, accessibility | Captions present; all figures appear original | Check permissions for any reused assets; add alt-style descriptions and avoid colour-only encoding | Should |
| 13 | Supplementary files cited as “Online Resource” | “Appendix: … (appendix.pdf)” | Relabel and upload as an Online Resource with a caption | Should |
| 14 | Acknowledgements in a separate title-page section | Acknowledgement at the end of the paper | Move to the title page or follow the template's acknowledgement field | Should |
| 15 | Decimal headings, no more than three levels | 10/12/3 headings | No change needed | Pass |
| 16 | No manuscript length limit | 37 pages double-spaced | No change needed; length will shrink in `iicol` two-column layout | Pass |
| 17 | LLM policy: LLMs cannot be authors; use documented except AI-assisted copy editing | ChatGPT acknowledged for clarity/fluency | This is AI-assisted copy editing, which need not be declared; keep the acknowledgement for transparency | Pass |
| 18 | Review anonymisation | Full author details | Guidelines do not state single vs double anonymisation; check the submission form | Verify |
| 19 | Cover letter | Addressed to JCGS | Rewrite for S&C | **Must** |
| 20 | Submission package contents | Contains JCGS diffs, responses, prior submissions | Exclude all JCGS-specific files from the new submission | **Must** |
| 21 | APC / open access | — | Not a manuscript gap; note the 1 July 2026 APC rule and 2027 full-OA transition | Note |

## 4. Must-fix checklist

1. Change the R Markdown output to `rticles::springer_article` with
   `classoptions: [iicol, sn-basic]` and rebuild the PDF.
2. Rewrite the YAML author/affiliation structure for the Springer template
   (`authors:` with `firstname`, `lastname`, `affiliation`, `corresponding`,
   `email`, `orcid`; `affiliations:` with `number`, `info`, `address`).
3. Reduce keywords from 10 to 4–6 and store them as a YAML list.
4. Add the corresponding-author email and ORCIDs; consider adding postcodes to
   the affiliations.
5. Consolidate the end matter into a **Statements and Declarations** section:
   competing interests, funding, ethics/consent, data/code availability, author
   contributions.
6. Add DOIs to the references and replace the five arXiv preprints with
   published versions where available.
7. Regenerate the two 72-dpi PNGs at publication resolution or as vector art;
   check figure widths for the two-column layout.
8. Rewrite the cover letter and remove all JCGS-specific files from the package.

## 5. Recommended package for the S&C submission

Include:

- `paper.Rmd` and the generated `paper.tex` (sn-jnl source), `paper.pdf`
- `sn-jnl.cls` and the selected `sn-*.bst`
- `bibliography.bib`, `paper.bbl` if required
- All figure source files (EPS/PDF/TIFF at required resolution)
- `appendix.pdf` (as an Online Resource)
- Cover letter and the Statements and Declarations

Exclude:

- `diff1.*`, `diff2.*`, `diff.tex`
- `revise_decision/`
- `paper-submission1.tex`, `paper-submission2.tex`
- `paper_author_details.pdf`, `paper_with_author_details.pdf`,
  `paper_copy.pdf`
- `arxiv_version_2024_11_2/`, `arxiv_version_2024_11_2.zip`
- `interact.cls`, `tfcad.bst`
- The JCGS cover letter

## 6. Converting the rticles template

The current YAML uses `rticles::tf_article`. The S&C configuration is:

```yaml
output:
  rticles::springer_article:
    keep_tex: true
classoptions:
  - iicol
  - sn-basic
```

Notes:

- `sn-basic` is the author–year Springer style; do **not** add `Numbered`.
- The `iicol` option is the journal's requested double-column layout.
- The current `header-includes` loads `setspace`, `hyperref`, `inputenc`, and
  sets `\doublespacing`. Audit these when switching to `sn-jnl`: some packages
  are already loaded by the class, and duplicate `hyperref` or option clashes
  can break the build.
- `biblio-style` is not used by the current Springer template; reference style
  is selected through `classoptions`.
- The appendix is a separate document using `interact.cls`. It can be submitted
  as-is as supplementary material, but relabel it as an Online Resource.

## Sources

- S&C submission guidelines:
  <https://link.springer.com/journal/11222/submission-guidelines>
- S&C aims and scope: <https://link.springer.com/journal/11222/aims-and-scope>
- S&C how to publish / APC:
  <https://link.springer.com/journal/11222/how-to-publish-with-us>
- rticles Springer template (skeleton and `template.tex`):
  <https://github.com/rstudio/rticles/tree/main/inst/rmarkdown/templates/springer>
- rticles Elsevier template (for contrast):
  <https://github.com/rstudio/rticles/tree/main/inst/rmarkdown/templates/elsevier>
