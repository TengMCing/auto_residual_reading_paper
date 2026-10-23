# Computational Statistics & Data Analysis — Gap Analysis of the Current Manuscript Package

> **AI disclaimer:** This gap analysis was drafted with the assistance of an AI
> language model on 23 October 2026. It is based on the files in `jcgs/`, the
> journal's Guide for Authors, and publicly available journal pages; it is not
> an official statement from the journal or publisher. Verify requirements on
> the live Guide for Authors before submitting.

**Current package:** `jcgs/` — the manuscript prepared for the *Journal of
Computational and Graphical Statistics* (Taylor & Francis).
**Target journal:** *Computational Statistics & Data Analysis* (Elsevier).
**Reviewed:** 23 October 2026.

*Note:* the live CSDA Guide for Authors blocks automated access. The
journal-specific rules below are taken from the official guide as archived on
15 April 2024, supplemented by current 2026 journal pages. Re-verify before
submission.

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

### JCGS-specific material (not part of a CSDA submission)

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
| Abstract | **177 words**, self-contained |
| Keywords | **10**, stored as one semicolon-separated block |
| Section depth | 10 sections, 12 subsections, 3 subsubsections (numbered 1/1.1/1.1.1) |
| Figures | **13** included: 11 vector PDFs, 2 PNGs at **72 dpi** |
| Tables | 4 |
| References | 65 cited, author–year, `tfcad.bst`; no DOI strings; 5 arXiv preprints clearly marked |
| Title page | 5 authors, 4 affiliations (department, university, city, state, country); **no corresponding-author email** |
| Declarations | Acknowledgement (incl. ChatGPT), Data Availability Statement, Disclosure Statement; **no formal generative-AI declaration, no funding statement** |
| Supplementary | Separate `appendix.pdf`, described in a Supplementary Materials section |
| Data/code | GitHub repository link in the Data Availability Statement |
| Scope | Computational (CV model) and data-analysis components both present |

## 3. Gap against CSDA requirements

Priority key: **Must** = required or likely to block/return the submission;
**Should** = strongly recommended; **Pass** = already satisfied;
**Optional** = voluntary; **Verify** = depends on the submission form.

| # | CSDA requirement | Current state | Gap and action | Priority |
| --- | --- | --- | --- | --- |
| 1 | New submissions may be a single PDF/Word file in any format (“Your Paper Your Way”) | `paper.pdf` exists | Acceptable at initial submission; no template conversion needed yet | Pass (initial) |
| 2 | At revision, editable source in `elsarticle.cls` | `paper.tex` uses `interact.cls` | Convert to `rticles::elsevier_article` before revision/final acceptance | **Must** (revision) |
| 3 | Numbered sections 1, 1.1, 1.1.1 | 10/12/3 headings | No change needed | Pass |
| 4 | Abstract concise, factual, self-contained | 177 words | No change needed | Pass |
| 5 | Maximum 6 keywords | 10 keywords | Cut to ≤6 and store as a YAML list | **Must** |
| 6 | Highlights: 3–5 bullets, ≤85 characters each, separate editable file | None | Prepare a separate “Highlights” file (optional but encouraged) | Optional (recommended) |
| 7 | Graphical abstract | None | Optional; prepare if useful | Optional |
| 8 | Title page: designated corresponding author with email; full postal address incl. country | No corresponding-author email; addresses at institution level | Add email and corresponding-author flag; add street/postcode if available | **Must** (email); Should (address) |
| 9 | References: consistent style at submission; journal style applied at proof | Author–year, consistent | Pass; DOIs encouraged but not required | Pass (optional DOIs) |
| 10 | Preprints clearly marked | 5 arXiv references, marked “arXiv preprint” | Pass | Pass |
| 11 | Competing-interests statement in the manuscript **and** the separate Declaration of Interest form | Disclosure Statement present; form not prepared | Complete Elsevier's Declaration of Interest form at submission | **Must** (form) |
| 12 | Generative AI disclosure in a section before References, using Elsevier's wording | ChatGPT mentioned in the Acknowledgement | Convert to the formal “Declaration of Generative AI and AI-assisted technologies in the writing process” section | **Must** |
| 13 | Funding statement (or standard “no specific grant” sentence) | Missing | Add a funding statement | **Must** |
| 14 | Data statement | Present | Pass; keep and link to the repository | Pass |
| 15 | Author contributions / CRediT | Missing | Add if the submission system asks; check the current form | Verify |
| 16 | Artwork: EPS/PDF vector; TIFF/JPG at 300 dpi (halftone), 1000 dpi (line), 500 dpi (combination) | 11 vector PDFs (fine); 2 PNGs at 72 dpi | Convert/regenerate the two PNGs as TIFF at ≥500–1000 dpi, or supply vector versions | **Must** |
| 17 | Supplementary material with concise captions, published as received | `appendix.pdf` described in the manuscript | Pass; upload separately with a caption | Pass |
| 18 | Single-anonymised review | Author details present | No anonymisation needed | Pass |
| 19 | Computational reporting: RNGs, algorithms, computers, languages, major software | Seeds present; Python 3.11.9, TensorFlow, Keras, MASSIVE M3 mentioned | Add a concise reproducibility paragraph with software versions, R version, and hardware if not already explicit | Should |
| 20 | Scope: computational or data-analysis component required; illustrative application where appropriate | CV model, residual diagnostics, human-experiment analysis, examples | Pass | Pass |
| 21 | Simulation studies: thorough settings, limitations, guidance | Simulation design described; appendix has details | Confirm the main text states limitations and when the method applies | Should (check) |
| 22 | Cover letter | Addressed to JCGS | Rewrite for CSDA | **Must** |
| 23 | Submission package contents | Contains JCGS diffs, responses, prior submissions | Exclude all JCGS-specific files | **Must** |
| 24 | No page charges; open access APC USD 3,540 | — | Not a manuscript gap; decide subscription vs OA | Note |
| 25 | Proofs returned within two days | — | Process requirement after acceptance | Note |

## 4. Must-fix checklist

1. Reduce keywords from 10 to at most 6.
2. Add a designated corresponding author and email to the title page.
3. Prepare the formal **Declaration of Generative AI and AI-assisted
   technologies in the writing process** section before the references.
4. Add a funding statement (or Elsevier's “no specific grant” sentence).
5. Complete the separate Declaration of Interest form and keep the manuscript
   competing-interests statement.
6. Regenerate the two 72-dpi PNGs at publication resolution; keep the PDF
   figures as vector art.
7. Convert the manuscript to `rticles::elsevier_article` for revision, keeping
   an `elsarticle.cls` source set with figures and bibliography.
8. Rewrite the cover letter and remove all JCGS-specific files from the
   submission package.

## 5. Recommended package for the CSDA submission

**Initial submission (Your Paper Your Way):**

- `paper.pdf` (current PDF is acceptable, but replace the low-resolution figures)
- `appendix.pdf` as supplementary material
- Cover letter
- Highlights file (optional)
- Data/code link (GitHub repository)

**Revision/final package:**

- `paper.Rmd` and the generated `paper.tex` using `elsarticle.cls`
- `elsarticle.cls`, the selected `.bst`, `bibliography.bib`
- Figure source files (EPS/PDF/TIFF at required resolution)
- `appendix.pdf` with a caption
- Completed Declaration of Interest form

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

The current YAML uses `rticles::tf_article`. The CSDA configuration is:

```yaml
output:
  rticles::elsevier_article:
    keep_tex: true
    citation_package: natbib
journal: "Computational Statistics & Data Analysis"
classoption: preprint, 3p, authoryear
biblio-style: elsarticle-harv
```

Notes:

- Elsevier's YAML key is **`classoption` (singular)**, not Springer's
  `classoptions`.
- Rewrite the author/address schema for the Elsevier template:

  ```yaml
  author:
    - name: Weihao Li
      email: patrick.li@anu.edu.au
      affiliation: Monash University
      correspondingauthor: true
  address:
    - code: Monash University
      organization: Department of Econometrics and Business Statistics
      addressline: ...
      city: Clayton
      state: VIC
      postcode: ...
      country: Australia
  ```

- Store keywords as a YAML list of at most six items.
- The current `\citep`/`\citet` commands are natbib-compatible and should work
  with `citation_package: natbib`.
- Audit `header-includes` when switching to `elsarticle`: the current file loads
  `setspace`, `hyperref`, `inputenc`, and sets `\doublespacing`; remove
  duplicates or options already provided by the class.
- Under “Your Paper Your Way”, the initial PDF does not need to follow the
  `elsarticle` layout; the source conversion is only needed at revision.

## Sources

- CSDA Guide for Authors (live):
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/publish/guide-for-authors>
- CSDA Guide for Authors (archived 15 April 2024):
  <https://web.archive.org/web/20240415204941/https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/publish/guide-for-authors>
- CSDA aims and scope:
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/about/aims-and-scope>
- CSDA journal insights:
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/about/insights>
- rticles Elsevier template (skeleton and `template.tex`):
  <https://github.com/rstudio/rticles/tree/main/inst/rmarkdown/templates/elsevier>
