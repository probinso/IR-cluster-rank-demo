default: report.pdf

report.pdf: report.tex ref.bib
	pdflatex report.tex
	bibtex report.aux
	pdflatex report.tex

