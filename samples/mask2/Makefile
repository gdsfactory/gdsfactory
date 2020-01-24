default: clean mask metadata 
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

metadata:
	pf mask write_metadata

show:
	pf show build/mask/sample_mask.gds

show2:
	pf show build/mask/sample_mask_remapped.gds

upload:
	mh post mask

fill:
	python fill.py

.PHONY: devices

