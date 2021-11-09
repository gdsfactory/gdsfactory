SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = gdsfactory
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

open:
	open build/html/index.html

see: html open

clean:
	rm -rf build

docs:
	python write_components_doc.py

remove-plugins:
	rm -rf notebooks/plugins/tidy3d

.PHONY: help Makefile upload
