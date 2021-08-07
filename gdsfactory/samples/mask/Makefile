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
	python test_mask.py

show:
	pf show build/mask/mask.gds


.PHONY: devices
