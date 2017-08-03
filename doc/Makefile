DOCDIR=./src
SRCDIR=../rankeval

# documentation is compiled by using shpinx
# exclude from documentation
DOCEXCLUDED=../rankeval/test

### Handling Sphinx for generating documentation
.PHONY: doc
doc: 
	@echo "==================================="
	@echo "Producing documentation..."

	# generate sphinx data
	sphinx-apidoc -o $(DOCDIR) -d 1 -f -F -M -H "RankEval" -A "HPC Lab" -V 0 -R 0.00 $(SRCDIR) $(DOCEXCLUDED)
	@cp $(DOCDIR)/static-index.rst $(DOCDIR)/index.rst

	# customize sphinx generation
	@echo "# custom" >> $(DOCDIR)/conf.py
	@echo "extensions += ['sphinx.ext.todo']" >> $(DOCDIR)/conf.py
	@echo "todo_include_todos = True" >> $(DOCDIR)/conf.py
	#@echo "extensions += ['numpydoc']" >> $(DOCDIR)/conf.py
	#@echo "extensions += ['sphinxcontrib.napoleon']" >> doc/conf.py
	@echo "extensions += ['sphinx.ext.autosummary']" >> $(DOCDIR)/conf.py
	@echo "extensions += ['sphinx.ext.imgmath']" >> $(DOCDIR)/conf.py
	@echo "numpydoc_show_class_members = False" >> $(DOCDIR)/conf.py
	# customize themes
	@echo "html_theme = \"sphinx_rtd_theme\"" >> $(DOCDIR)/conf.py

	@echo "import sys,os" >> $(DOCDIR)/conf.py
	@echo "sys.path.insert(0, os.path.abspath('../../') )" >> $(DOCDIR)/conf.py
	#@echo "print (sys.path)" >> $(DOCDIR)/conf.py

	@echo "from setuptools import sandbox" >> $(DOCDIR)/conf.py
	@echo "sandbox.run_setup(os.path.abspath('../../setup.py'),  ['build_ext','-i'])" >> $(DOCDIR)/conf.py

	@echo "autoclass_content = 'both'" >> $(DOCDIR)/conf.py

	# compile HTML files
	make -C $(DOCDIR) html
