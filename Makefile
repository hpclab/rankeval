PYTHON ?= python
NOSETESTS ?= nosetests

build: _svmlight_loader.so

_svmlight_loader.so: rankeval/core/dataset/_svmlight_loader.cpp
	$(PYTHON) setup.py build_ext --inplace

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf rankeval.egg-info
	rm -rf build *.so *.pyc

test:
	$(NOSETESTS)

# documentation is compiled by using shpinx
# exclude from documentation
DOCEXCLUDED=./hpckit/features/extract_ndcg_test.py ./hpckit/features/feature_selection.py ./hpckit/features/test_ndcg.py


### Handling Sphinx for generating documentation
.PHONY: doc
doc: doc/scipy-sphinx-theme
	@echo "==================================="
	@echo "Producing documentation..."

	# generate sphinx data
	sphinx-apidoc -o ./doc -f -F -H "HPC-Kit" -A "HPC Lab" -V 0 -R 0.00 ./hpckit $(DOCEXCLUDED)

	# customize sphinx generation
	@echo "# custom" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.todo']" >> doc/conf.py
	@echo "todo_include_todos = True" >> doc/conf.py
	@echo "extensions += ['numpydoc']" >> doc/conf.py
	#@echo "extensions += ['sphinxcontrib.napoleon']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.autosummary']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.pngmath']" >> doc/conf.py
	@echo "numpydoc_show_class_members = False" >> doc/conf.py
	
	# customize themes
	#@echo "html_theme = \"sphinxdoc\"" >> doc/conf.py
	@echo "html_theme = \"scipy\"" >> doc/conf.py
	@echo "html_theme_path = [\"scipy-sphinx-theme/_theme\"]" >> doc/conf.py

	# compile HTML files
	export PYTHONPATH=${PYTHONPATH}:`pwd`; make -C doc html

doc/scipy-sphinx-theme:
	mkdir -p doc
	cd doc; git clone https://github.com/scipy/scipy-sphinx-theme.git
	rm doc/scipy-sphinx-theme/*.rst doc/scipy-sphinx-theme/Makefile
	@echo "div.admonition-todo {background-color: #f0fdff; border: 1px solid #abd0ff;}" >> doc/scipy-sphinx-theme/_theme/scipy/static/scipy.css_t
