# DocumentaČ›ie ArtAdvisor (Text Only)

Acest pachet conČ›ine sursele LaTeX fÄrÄ imagini (folderul figs/ exclus).

## FiČ™iere copiate
- licenta.tex
- foi_de_capat.tex
- ghid.tex
- introducere.tex
- obiective.tex
- studiu_biblio.tex
- analiza_fundamentare.tex
- proiectare_implementare.tex
- testare_validare.tex
- instalare_utilizare.tex
- concluzii.tex
- anexe.tex
- rezumat.tex
- cs.sty
- romanian.sty
- licenta.bib

## Imagini
Figurile lipsesc. Compilarea foloseČ™te modul *draft* pentru graphicx (vedeČ›i modificarea scriptului).

## Compilare
Recomandat:
`
latexmk -pdf licenta.tex
`
Alternativ:
`
pdflatex licenta.tex
bibtex licenta
pdflatex licenta.tex
pdflatex licenta.tex
`

## Re-activare imagini
ČtergeČ›i opČ›iunea [draft] din linia \\usepackage[draft]{graphicx} Č™i adÄugaČ›i directorul igs/.

## LicenČ›Ä / Note
Bibliografia este Ă®n licenta.bib Č™i stilul Ă®n IEEEtran.
