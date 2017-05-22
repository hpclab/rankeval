PYTHON ?= python
NOSETESTS ?= nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf rankeval.egg-info
	rm -rf build *.so *.pyc

test:
	$(NOSETESTS)


DOCDIR=./doc/doc
SRCDIR=./rankeval

# documentation is compiled by using shpinx
# exclude from documentation
DOCEXCLUDED=./rankeval/test

### Handling Sphinx for generating documentation
.PHONY: doc
doc: 
	@echo "==================================="
	@echo "Producing documentation..."

	# generate sphinx data
	sphinx-apidoc -o $(DOCDIR) -d 1 -f -F -H "RankEval" -A "HPC Lab" -V 0 -R 0.00 $(SRCDIR) $(DOCEXCLUDED)
	@cp doc/static-index.rst $(DOCDIR)/index.rst

	# customize sphinx generation
	@echo "# custom" >> $(DOCDIR)/conf.py
	@echo "extensions += ['sphinx.ext.todo']" >> $(DOCDIR)/conf.py
	@echo "todo_include_todos = True" >> $(DOCDIR)/conf.py
	@echo "extensions += ['numpydoc']" >> $(DOCDIR)/conf.py
	#@echo "extensions += ['sphinxcontrib.napoleon']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.autosummary']" >> $(DOCDIR)/conf.py
	@echo "extensions += ['sphinx.ext.imgmath']" >> $(DOCDIR)/conf.py
	@echo "numpydoc_show_class_members = False" >> $(DOCDIR)/conf.py
	# customize themes
	@echo "html_theme = \"sphinx_rtd_theme\"" >> $(DOCDIR)/conf.py

	# compile HTML files
	make -C $(DOCDIR) html
