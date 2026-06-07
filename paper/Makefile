# Regenerate auto-filled assets, then build the PDF.
PDF=main.pdf

all: assets $(PDF)

assets:
	cd .. && python experiments/make_paper_assets.py

$(PDF): main.tex refs.bib dataset_stats.tex results_table.tex
	pdflatex -interaction=nonstopmode main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out $(PDF)
