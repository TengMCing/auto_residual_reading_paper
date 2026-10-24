# Computational Statistics & Data Analysis — Submission Requirements

> **AI disclaimer:** This summary was drafted with the assistance of an AI
> language model from publicly available sources, and re-verified on 25 October
> 2026 against the journal's live Guide for Authors (rendered in a browser; the
> site blocks non-browser clients). It is not an official statement from the
> journal or publisher. Requirements, fees, and policies may contain errors or
> omissions and can change — verify them against the official pages linked under
> Sources before submitting.
>
> **Note:** the live guide has been substantially rewritten since the 2024
> version (the last one archived by the Wayback Machine). The old
> "Your Paper Your Way" section, the six-keyword maximum, the prescribed
> "Declarations of interest: none" wording, the RunMyCode paragraph, and the
> no-page-charge sentence are **gone**. Anything based on the 2024 text is
> flagged below.

## 1. Journal profile and article types

- **Official journal** of the Network for Computational and Methodological
  Statistics (CMStatistics) and of the International Association of Statistical
  Computing (IASC).
- **Sections:** the aims text says "the journal consists of four refereed
  sections", then lists three: I) Computational Statistics, II) Statistical
  Methodology for Data Analysis, III) Special Applications.
- **Article types:** research papers, short communications, review papers, and
  book reviews.
- **Scope:** the live aims page no longer says papers "must include either a
  computational or data analysis component", and the old "purely theoretical
  papers … will be returned to the authors" sentence has been removed. The
  simulation-study expectations paragraph is also gone. Do not treat those as
  current rules.
- **Metrics (live page, 25 Oct 2026):** CiteScore 3.6, Impact Factor 1.7,
  acceptance rate 8%.
- **Timeline (live page):** 4 days from submission to first decision; 106 days
  from submission to decision after review; 278 days from submission to
  acceptance; 6 days from acceptance to online publication.

## 2. Manuscript format and files

- **File types:** `.doc`/`.docx` for Word, `.tex` for LaTeX. **"A PDF is not an
  acceptable source file."** A compiled PDF may accompany the source, but the
  editable source is what is required.
- **Word submissions:** single-column layout. "Double-column formatting is only
  permitted for LaTeX submissions."
- Remove strikethrough and underline formatting before submitting.
- **LaTeX:** "We encourage you use our LaTeX template when preparing a LaTeX
  submission. You will be asked to provide all relevant editable source files
  upon submission or revision." `elsarticle.cls` is no longer named in the live
  guide (it was "recommended" in 2024); the generic Elsevier LaTeX template is
  the safe choice and rticles::elsevier_article already emits an
  `elsarticle.cls` document.
- **Section numbering:** "Divide your manuscript into clearly defined and
  numbered sections. Number subsections 1.1 (then 1.1.1, 1.1.2, ...), then 1.2,
  etc." Headings go on a separate line; the abstract is not numbered.
- **Appendices:** lettered A, B, …; equations (A.1), tables Table A.1.

## 3. Title page

- Title: concise and informative.
- Author names with a **lowercase superscript letter** immediately after the
  name, matched to the affiliation addresses.
- "Full name and postal address of each affiliation, including the country
  name."
- "The email address of the corresponding author must be included on the title
  page and will be published in the article."
- Present/permanent addresses as a footnote using superscript Arabic numerals.
- Group authorship must be declared on the title page.

## 4. Abstract and keywords

- **Abstract:** "concise and factual … does not exceed 250 words" (upper bound
  only — the 2024 "maximum 250 words" is still the operative rule, but there is
  no stated minimum). It must state the purpose, principal results, and major
  conclusions, stand alone, avoid references where possible (if essential, cite
  author and year in the text), and define abbreviations at first mention.
- **Keywords:** "You are required to provide 1 to 7 keywords for indexing
  purposes. Keywords should be written in English. Try to avoid keywords
  consisting of multiple words (using "and" or "of")." Abbreviations only if
  firmly established. *The old "maximum of 6 keywords" and "American spelling"
  rules are no longer stated.*

## 5. Highlights

- "You are encouraged to provide article highlights at submission." (Optional,
  not mandatory.)
- Separate editable file "with the word "highlights" included in the file name";
  upload it under the "Highlights" file type.
- "3 to 5 bullet points, each a maximum of 85 characters, including spaces."
- Capture the core results only.

## 6. Tables, figures, and artwork

- **Tables:** "must be submitted as editable text, not as images"; cited in the
  text, numbered consecutively, caption above, notes below the body, "Avoid
  vertical rules and shading within table cells".
- **Figures** are supplied as separate files with logical names (`Figure_1`,
  `Figure_2`, …). Captions consist of a brief title (not embedded in the
  artwork) plus a description.
- **Artwork formats and minimum resolution** (live table):

  | Type | Formats | Minimum resolution |
  | --- | --- | --- |
  | Vector drawings | EPS or PDF (embed fonts or save text as graphics) | — |
  | Photographs (halftone) | TIFF, JPG, PNG | 300 dpi (≥1063 px for single column, ≥2244 px full page) |
  | Bitmapped line drawings | TIFF, JPG, PNG | 1000 dpi (≥3543 px / ≥7480 px) |
  | Charts and large or shaded tables | XLS, XLSX | — |
  | Annotated images | PPT, PPTX (300 dpi embedded) | — |

  Do not submit low-resolution files, disproportionately small text, or several
  graphs combined into one file (comparison panels are the exception).
- Colour is free online; print colour is charged after acceptance.
- **Generative AI and figures:** AI may be used to create explanatory images and
  data visualisations "when the output is directly derived from underlying data
  using reproducible analytical, computational, or statistical methods". "AI
  tools must not be used to create or alter images that represent primary
  observed or experimental data." Disclose AI use in each relevant caption and
  in the general AI statement.

## 7. Supplementary material, video, and data

- **Supplementary material:** must be accurate and relevant, cited in the text,
  supplied with the manuscript, with a concise caption for each file. It "will
  appear online exactly as received" — no reformatting by the publisher.
- **Video:** accepted as supplementary; preferred maximum 150 MB per file and
  1 GB in total; clear file names; provide still frames and descriptive text for
  accessibility.
- **Research data:** the "Option B" instructions of Elsevier's research data
  guidelines apply — deposit data in a repository and cite/link it, or explain
  why the data cannot be shared.
- **Data statement:** "you are encouraged to state the availability of any data
  at submission"; the statement "will appear with your published article on
  ScienceDirect".
- **Data linking:** use the format `Database: 12345` to link datasets to the
  article.
- Related data, methods, or protocols may be co-submitted to *Data in Brief* or
  *MethodsX*.

## 8. References

- "This journal does not set strict requirements on reference formatting at
  submission. References can be in any style or format as long as the style is
  consistent." The journal applies its own style at proof stage.
- Every reference must be cited in the text, and every cited work must appear in
  the list. Avoid references in the abstract; if essential, give them in full.
- Include author names, journal/book title, year, volume or article number, and
  pagination.
- DOIs are encouraged. Web references need the full URL and a last-access date.
- Data references: author, dataset title, repository, version, year, and
  persistent identifier, with **"[dataset]" immediately before the reference**.
- Preprints may be cited if marked with the word "preprint" and the preprint
  DOI.
- Remove field codes before submitting.

## 9. Declarations

All declarations are placed **at the end of the manuscript, before the reference
list**, except competing interests (see below).

- **Competing interests (changed in 2025–2026):** "The declarations tool should
  always be completed." All authors disclose interests; authors with none select
  "I have nothing to declare." The tool produces a **Word document that must be
  uploaded as a separate file in `.doc`/`.docx` format** at the "attach/upload
  files" step; author signatures are not required. The old instruction to write
  "Declarations of interest: none" in the manuscript is no longer in the guide —
  keep a short statement in the manuscript if desired, but the tool output is
  what the journal requires. Editors must declare their own role: "Given their
  role as …, … had no involvement in the peer review …".
- **Funding:** disclose sources and the role of the sponsor, in the standard
  format "Funding: This work was supported by … [grant numbers …]". If there was
  no funding, use "(This research did not receive any specific grant from
  funding agencies in the public, commercial, or not-for-profit sectors.)"
- **Generative AI:** "Authors must declare the use of generative AI tools in the
  manuscript preparation process upon submission." Add a statement at the end of
  the manuscript **when first submitted, in a new section before the reference
  list**, titled "Declaration of generative AI and AI-assisted technologies in
  the manuscript preparation process." Use the template: "During the preparation
  of this work the author(s) used [NAME OF TOOL/SERVICE] to [REASON]. After using
  this tool/service, the author(s) reviewed and edited the content as needed and
  take(s) full responsibility for the content of the published article." The
  requirement excludes basic tools used to check grammar, spelling, and
  references. If there is nothing to disclose, no statement is needed. AI tools
  may not be listed or cited as authors.
- **Acknowledgements:** a separate section **directly before the reference
  list**; not on the title page, not as a title footnote, not elsewhere. *The
  old "Your Paper Your Way" layout is gone.*

## 10. Ethics, authorship, and integrity

- **Submission declaration:** the work is not published elsewhere (preprints,
  abstracts, theses excepted), is not under consideration by another journal, is
  approved by all authors and the responsible authorities, and will not be
  published elsewhere in the same form.
- **Authorship:** substantial contribution to conception/design, acquisition,
  analysis or interpretation; drafting or critical revision; final approval;
  accountability. Single corresponding author, whose affiliation determines
  publishing-agreement eligibility. Group authorship declared on the title page.
- **Changes to authorship:** only before acceptance, via the Authorship Change
  Request form with editor approval; "never after acceptance."
- **Inclusive language** and **sex- and gender-based analysis (SAGER)**: report
  sex/gender where relevant.
- **Image manipulation:** no specific feature enhancement, obscuring, or
  selective deletion; unmatched control images must be labelled. "Use of AI tools
  to create or manipulate primary research images is not permitted." Images may
  be screened by third-party tools.
- Jurisdictional claims must be neutral; the journal may delay publication for
  legal/ethical reasons; withdrawal, correction, and retraction follow Elsevier
  policy.

## 11. Submission, review, and licensing

- **Submit at <https://submit.elsevier.com/CSDA>** — the live page labels it a
  "new submission service". (`editorialmanager.com/CSDA` still resolves but
  shows "Site under development. Do not use for live manuscript submission.")
- **Peer review:** single anonymized; an editor screens for suitability, then a
  minimum of two independent reviewers; the editor's decision is final. "Only
  one appeal per submission will be considered." Special issues are handled by
  guest editors.
- The live guide gives **no reviewer-suggestion requirement** and no rule about
  the editorial board; the old referee-suggestion paragraph has been dropped
  (the submission system may still offer the field).
- **Publishing agreement** is signed after acceptance; **license options**
  (CC BY, CC BY-NC-ND) and the **Article Transfer Service** are described on the
  site.
- **APC:** USD 3,540 (excluding taxes) for open access, per the journal insights
  page. Subscription publication remains free — the old "no page charges"
  sentence is gone from the guide but no page or submission charge is listed.
- **Proofs:** corrections "within two days"; proofs are for checking
  typesetting, editing, completeness, and correctness only.
- **Sharing:** authors receive a Share Link valid for 50 days.

## 12. Drafting implications for this paper

- **Keywords:** the paper currently lists **6**, within the new 1–7 range.
- **Abstract:** 177 words, within the 250-word limit.
- **Highlights:** `csda/paper/highlights.md` holds 4 bullets, all under the
  85-character limit; export it to `.docx` and upload it as a "Highlights" file.
- **Competing interests:** the manuscript's "Declarations of interest: none"
  line is no longer the required mechanism; complete Elsevier's declarations
  tool and upload the generated Word file, keeping a manuscript statement for
  the record.
- **Generative AI:** retitle the section to "Declaration of generative AI and
  AI-assisted technologies in the manuscript preparation process" (the current
  Rmd heading says "Declaration of Generative AI"), keep the prescribed
  template wording, and confirm it sits immediately before the references.
- **File format:** `.tex` source plus a compiled PDF for reading; a PDF alone is
  not accepted as the source. Single-column Word output only if a Word fallback
  is ever needed.
- **Data statement:** the GitHub repository already satisfies the encouraged
  data statement; add a `[dataset]` reference if the training data are cited.
- **Computational reporting:** the guide requires the computer(s), programming
  language(s), and major software components; the Rmd has a TODO covering RNG,
  algorithms, hardware, languages, and software versions.

## 13. R Markdown / R package support

- **`rticles::elsevier_article()`** is the closest ready-made template: it emits
  `elsarticle.cls`, `elsarticle-harv` (author–year) or numeric bibliography
  style, and front matter for title, authors, affiliations, abstract, keywords,
  and journal. `rticles` also bundles `.bst` files for the common Elsevier
  bibliography styles.
- `elsarticle.cls` is no longer mandated by the live guide, but it remains the
  standard Elsevier LaTeX class and is what the template produces.
- **Affiliation markers:** the YAML `affiliation` field must be a comma-separated
  list **without spaces** (`affiliation: "1,2"`). With a space (`"1, 2"`),
  `elsarticle.cls` fails to resolve the second label and silently prints "1".
- Single-column Elsevier submissions are the default; `3p` (three-page) and
  `5p` are layout options, and `preprint` produces the review layout.
- Set the journal name in the YAML `journal:` field so the footer and running
  head are correct.

## Sources

- Guide for Authors (live, browser-only):
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/publish/guide-for-authors>
- Journal home and insights (metrics, timeline, APC):
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis>
- Aims and scope:
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/about/aims-and-scope>
- Submission portal: <https://submit.elsevier.com/CSDA>
- Elsevier LaTeX instructions:
  <https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions>
- Elsevier research data guidelines:
  <https://www.elsevier.com/about/policies-and-standards/research-data>
- Elsevier generative AI policies:
  <https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals>
- Highlights guidance (Elsevier support):
  <https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript>
- Archived 2024 guide (for comparison, since the live page cannot be archived by
  most tools):
  <https://web.archive.org/web/20240415204941/https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/publish/guide-for-authors>
