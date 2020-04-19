default: clean mask metadata

help:
	@echo 'make :                 makes devices, merges them'
	@echo 'make devices:          makes devices'
	@echo 'make metadata:         merges all devices GDS and JSON'
	@echo 'make show:             show mask in klayout'

venv:
	python3 -m venv env

clean:
	rm -rf build

mask: clean
	python test_mask.py

metadata:
	pf mask write_metadata

show:
	pf show build/mask/sample_mask.gds


.PHONY: devices
