PYTHON ?= python
NOSETESTS ?= nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf rankeval.egg-info
	rm -rf build
	rm -rf .ipynb_checkpoints
	rm -rf .eggs
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	find . -name "*.egg" -delete

test:
	$(NOSETESTS)

### Handling Sphinx for generating documentation
.PHONY: doc
doc: 
	@echo "==================================="
	@echo "Producing documentation..."

	make -C doc doc
