.PHONY: build upload

build:
	rm -rf dist
	uv build

upload:
	uv publish
