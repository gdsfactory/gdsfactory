format:
	jupytext --pipe black *.py

merge:
	jupytext --to markdown *.ipynb
	cat 00_geometry.md  01_references.md 03_cells_autoname_and_cache.md 03_layer_stack.md 03_Path_CrossSection.md 04_components_geometry.md 04_components_hierarchy.md 04_components_pack.md 04_components_shapes.md 04_routing.md 04_routing_electrical.md 04_routing_non_manhattan.md 07_mask.md 07_mask_labels.md 08_pdk.md 08_pdk_examples.md 09_pdk_import.md 10_yaml_component.md 11_best_practices.md > combined.md
	jupytext --to notebook combined.md


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
