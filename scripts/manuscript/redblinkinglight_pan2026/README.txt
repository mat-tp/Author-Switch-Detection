HOW TO BUILD THE FINAL PDF
1. Put Springer's official llncs.cls and splncs04.bst (from the LNCS author kit) next to main.tex.
   They were NOT available where this paper was written, so they are not included.
2. pdflatex main; bibtex main; pdflatex main; pdflatex main
3. Check the page count in the real layout (target 8-10 pages including references).
main_preview_APPROX_layout.pdf was built with a stand-in class (same text block 122x193 mm, 10 pt) and a generic
bibliography style, WITHOUT the T1 font encoding (not installed there). It is a preview only; page breaks and the
reference list format will differ slightly in the real build.
figs/fig_results.pdf is the per-level results chart. It is not used in main.tex (it was dropped for space; Table 1 holds
the same numbers). To add it back, insert it as a figure after Table 1 (about 1/3 page).
