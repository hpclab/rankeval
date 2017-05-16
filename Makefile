PYTHON ?= python
NOSETESTS ?= nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf rankeval.egg-info
	rm -rf build *.so *.pyc

test:
	$(NOSETESTS)

# documentation is compiled by using shpinx
# exclude from documentation
DOCEXCLUDED=


### Handling Sphinx for generating documentation
.PHONY: doc
doc: 
	@echo "==================================="
	@echo "Producing documentation..."

	# generate sphinx data
	sphinx-apidoc -o ./doc -f -F -H "RankEval" -A "HPC Lab" -V 0 -R 0.00 ./rankeval $(DOCEXCLUDED)

	# customize sphinx generation
	@echo "# custom" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.todo']" >> doc/conf.py
	@echo "todo_include_todos = True" >> doc/conf.py
	@echo "extensions += ['numpydoc']" >> doc/conf.py
	#@echo "extensions += ['sphinxcontrib.napoleon']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.autosummary']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.imgmath']" >> doc/conf.py
	@echo "numpydoc_show_class_members = False" >> doc/conf.py
	
	# customize themes
	@echo "html_theme = \"sphinx_rtd_theme\"" >> doc/conf.py

	# compile HTML files
	make -C doc html

