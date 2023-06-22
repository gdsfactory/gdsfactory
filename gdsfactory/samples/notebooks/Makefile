format:
	jupytext --pipe black *.py

clear:
	jupytext --clear-metadata *.py

sync:
	jupytext --sync *.ipynb

md_to_py:
	jupytext *.md --to py

notebooks:
	jupytext *.md --to ipynb

markdown:
	jupytext *.py --to md
	jupytext **/*.py --to md
	jupytext **/*.py --to md

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb

rm_ipynb:
	rm *.ipynb
	rm **/*.ipynb

set:
	jupytext --set-formats md,ipynb *.md
