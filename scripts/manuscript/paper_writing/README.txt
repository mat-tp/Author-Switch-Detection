HOW TO BUILD THE FINAL PDF
1. Put Springer's official llncs.cls and splncs04.bst (LNCS author kit) next to main.tex. They are not included.
2. pdflatex main; bibtex main; pdflatex main; pdflatex main
3. Check the page count in the real layout (target 8-10 pages including references).
main_preview_APPROX_layout.pdf uses a stand-in class and bibliography style and no T1 fonts: preview only.
archive_pan_notebook_version.tex is the earlier version framed as a PAN 2026 notebook (with the reviewer responses).
figs/fig_results.pdf is the per-level chart, not used in main.tex (room for it: about 1/3 page).
