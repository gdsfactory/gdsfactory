default: clean mask metadata
help:
	@echo 'make :                 makes devices, merges them'
	@echo 'make mask:             makes mask'
	@echo 'make merge:            merges all devices GDS and JSON'
	@echo 'make show:             show mask in klayout'


venv:
	python3 -m venv env

clean:
	rm -rf build

mask:
	python test_placer.py

metadata:
	pf mask write_metadata

show:
	pf show build/mask/test_placer.gds



.PHONY: devices
