Build: place Springer's llncs.cls (checked with v2.24, 29-Jan-2024) and splncs04.bst next to main.tex, then
pdflatex main; bibtex main; pdflatex main; pdflatex main
main_preview_real_llncs.pdf was built with the real llncs.cls v2.24 and splncs04.bst. The only difference from your build:
the T1 font encoding and microtype font expansion were switched off in the preview because the scalable T1 fonts (cm-super)
were not installed where it was built. Expect the same layout, with a small change in line breaks.
Preview length: 10 pages including references, the last page about half full.
