default: clean mask
help:
	@echo 'make :                 makes devices, merges them'
	@echo 'make mask:             makes mask'
	@echo 'make merge:            merges all devices GDS and JSON'
	@echo 'make show:             show mask in klayout'
	@echo 'make upload            upload mask to maskhub'

venv:
	python3 -m venv env

clean:
	rm -rf build

mask:
	python write_mask.py
	pf mask write_metadata

show:
	pf show build/mask/sample_mask.gds


.PHONY: devices
