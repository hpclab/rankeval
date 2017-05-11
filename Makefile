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
