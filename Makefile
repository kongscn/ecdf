.PHONY: build upload

build:
	rm -rf dist
	python -m build

upload:
	twine upload dist/* --repository ecdf
