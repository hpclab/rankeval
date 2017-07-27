PYTHON ?= python
NOSETESTS ?= nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf rankeval.egg-info
	rm -rf build *.so *.pyc *.egg *.so
	rm -rf **/**/*.so

test:
	$(NOSETESTS)


DOCDIR=./doc/src
SRCDIR=./rankeval

# documentation is compiled by using shpinx
# exclude from documentation
DOCEXCLUDED=./rankeval/test

### Handling Sphinx for generating documentation
.PHONY: doc
doc: 
	@echo "==================================="
	@echo "Producing documentation..."

	make -C doc doc
