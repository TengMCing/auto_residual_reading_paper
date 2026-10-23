# Statistics and Computing — Submission Requirements

> **AI disclaimer:** This summary was drafted with the assistance of an AI
> language model from publicly available sources on 23 October 2026. It is not
> an official statement from the journal or publisher. Requirements, fees, and
> policies may contain errors or omissions and can change — verify them against
> the official pages linked under Sources before submitting.

*Source: the journal's live submission guidelines and journal page, checked
23 October 2026.*

## 1. Article types and length

- **Article types:** original research articles, review papers, discussion
  papers, and special issues. The journal also publishes book and software
  reviews; the guidelines additionally mention editorials and comments.
- **Length:** no word or page limit is stated.
- **Abstract:** 150–250 words, with no undefined abbreviations or unspecified
  references.
- **Keywords:** 4–6.
- **Title:** concise and informative; avoid abbreviations and formulae where
  possible.

## 2. Manuscript format and template

- **LaTeX is the expected format.** Use the **Springer Nature LaTeX template**
  with the **`iicol`** formatting option.
- Submit the **editable source** — all style files, figures, and the `.tex`
  file — together with a **compiled PDF**, at every submission and revision. An
  incomplete source set means the paper will not be considered for review.
- **Word files are also accepted**, but the journal does not provide a Word
  template; accepted papers are formatted in production.
- **Headings:** decimal numbering with no more than three levels.
- **Footnotes**, not endnotes. Footnotes must not consist only of a citation,
  contain bibliographic details, or include figures or tables.
- Define abbreviations at first use.

## 3. Title page and declarations

The title page should include:

- Author names and full affiliation addresses (institution, department, city,
  state, country).
- A working corresponding-author email.
- ORCID identifiers where available.
- Acknowledgements in a separate title-page section; funder names written in
  full.

A **Statements and Declarations** section is required before the reference list.
It should cover, as applicable:

- Competing interests
- Funding
- Ethics approval and consent
- Data, material, and/or code availability
- Author contributions

Author-contribution and competing-interest information must also be entered
through the submission interface — only the interface entries are used in the
published version. Submissions missing relevant declarations may be returned as
incomplete.

## 4. References

- Cite by **author and year**, in parentheses or narratively.
- The reference list includes only works that are published or accepted.
  Personal communications and unpublished work are mentioned only in the text.
- Alphabetise the list by first author.
- Give DOIs as full links and use ISSN-approved journal abbreviations.
- The guidelines provide formats for journal articles, DOI-only articles,
  books, book chapters, and online documents.

## 5. Figures, tables, and supplementary material

- Number tables and figures with Arabic numerals and cite them consecutively;
  provide explanatory captions.
- **Preferred figure formats:** EPS for vector art and TIFF for halftones;
  Office files are acceptable. Embed fonts in vector files. Name files, for
  example, `Fig1.eps`.
- **Minimum resolutions:** 1200 dpi for line art, 300 dpi for halftones, and
  600 dpi for combination art.
- Use RGB for colour figures. Online colour is free; print colour may incur a
  charge. Ensure information remains distinguishable in black and white if
  print will be monochrome.
- Use Helvetica or Arial lettering at roughly 8–12 pt in final size. Do not put
  titles or captions inside the artwork; place captions in the manuscript text.
- Obtain permission for reused figures and tables for both print and online
  formats.
- **Accessibility:** descriptive captions, patterns in addition to colour, and
  lettering contrast of at least 4.5:1.
- **Supplementary files:** standard formats, with article/author details,
  captions, and cited as “Online Resource”. Videos must be 16:9 or 4:3 and no
  larger than 2 GB; text/presentations should be PDF; spreadsheets CSV or XLSX.

## 6. Ethics, data, and AI policy

- The work must be original, not under consideration elsewhere, and free of
  plagiarism and salami slicing. Reused material requires permission.
- Every original research article needs a **data availability statement**;
  depositing data in a suitable repository is encouraged.
- **Competing interests:** disclose interests within the three years before the
  work began, plus older interests that could reasonably appear relevant.
  Disclose both financial and non-financial interests.
- **Large language models cannot be authors.** LLM use must be documented in
  the Methods section or an equivalent part, except AI-assisted copy editing
  (readability, grammar, spelling, punctuation, tone), which need not be
  declared. Humans remain accountable for the final text.
- Authorship changes are generally restricted during revision and are not
  allowed after acceptance.

## 7. Submission and review process

- Submit through **Editorial Manager** (`editorialmanager.com/stco`), uploading
  all manuscript files.
- Authors may suggest independent reviewers or request exclusions, supplying
  institutional email addresses or another means of identity verification.
  Suggestions are not guaranteed to be used.
- The journal aims for a **first decision in under three months**; the official
  median is 3 days, which mostly reflects desk screening.
- After acceptance, online publication (the official version of record) usually
  takes **3–5 weeks** if proofs are returned promptly.
- There is **no submission fee**. Open-access terms are in transition: the
  journal is hybrid now and becomes fully open access on **1 January 2027**;
  submissions received from **1 July 2026** are subject to an APC if accepted
  (€2,390 / $2,990 / £2,090). See the journal overview for details.
- The guidelines do not explicitly state whether review is single- or
  double-anonymous; check the current submission form and prepare the title page
  accordingly.

## 8. Practical checklist for this manuscript

- Produce a LaTeX version from the R Markdown source using the Springer Nature
  template with the `iicol` option; include every figure and style file plus the
  compiled PDF.
- Shorten the abstract to **150–250 words** and supply **4–6 keywords**; the
  current manuscript abstract is longer.
- Convert references to Springer's author–year format and include full DOI
  links.
- Add the **Statements and Declarations** section: competing interests, funding,
  data/code availability (the GitHub repository), and author contributions.
- Add a **Data Availability Statement** pointing to the repository and released
  data/code.
- The manuscript's acknowledgement of ChatGPT for clarity and fluency falls
  under AI-assisted copy editing, which Springer does not require to be
  declared; keep the acknowledgement for transparency.
- Budget for the APC if the paper is accepted after 1 July 2026.

## 9. R Markdown / R package support

- There is **no journal-specific R package** for *Statistics and Computing*.
  Springer distributes the LaTeX class (`sn-jnl.cls`) and style files, not an R
  package.
- **Online guidance:** as of October 2026, searches of the rticles GitHub issues
  (both the repository and GitHub-wide), Stack Overflow, the Posit Community
  forum, and GitHub repositories found **no discussion or example specifically
  about using rticles for this journal**. The configuration below is derived
  from the generic template and the journal's LaTeX requirements, not from a
  journal-specific community recipe.
- The **`rticles`** package on CRAN provides
  **`rticles::springer_article()`**, a generic Springer Nature R Markdown
  template built on `sn-jnl.cls`. This is the closest ready-made R workflow for
  the journal.
- The journal asks for the **`iicol`** formatting option. The template's
  skeleton describes this option as a double-column layout. Set it through the
  top-level `classoptions` field (the template default is `sn-basic`):

  ```yaml
  output: rticles::springer_article
  classoptions:
    - iicol
  ```

- **Documentation note:** `classoptions` and the supported class options
  (including `iicol`) are listed in the template's `skeleton.Rmd` comments,
  which ship with the package. The `?springer_article` help page documents only
  the function arguments, not the YAML fields.
- Create a new document with:

  ```r
  rticles::springer_article()
  ```

  or in RStudio via **File → New File → R Markdown → From Template → Springer
  Journal Article**.
- `springer_article()` writes a `.tex` file using `sn-jnl.cls` and bundles the
  class, bibliography, and style files, so the editable source set the journal
  requires is produced directly.
- Caveats:
  - The template is generic, not journal-specific; verify the generated PDF
    against the journal's current layout requirements.
  - Set the title, authors, affiliations, abstract (150–250 words), and
    keywords (4–6) in the YAML; the default template abstract is a placeholder.
  - Keep the `.tex`, `.cls`, `.bst`, and figure files from the rendered output
    for submission.

## Sources

- Submission guidelines:
  <https://link.springer.com/journal/11222/submission-guidelines>
- Journal home and article types: <https://link.springer.com/journal/11222>
- Aims and scope: <https://link.springer.com/journal/11222/aims-and-scope>
- How to publish / APC: <https://link.springer.com/journal/11222/how-to-publish-with-us>
- Springer Nature LaTeX template:
  <https://www.springernature.com/gp/authors/campaigns/latex-author-support>
- Springer Nature research data policy:
  <https://www.springernature.com/gp/authors/research-data-policy>
