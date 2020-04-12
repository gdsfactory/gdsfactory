help:
	@echo 'make mask:             makes mask'
	@echo 'make show:             show mask in klayout'

venv:
	python3 -m venv env

clean:
	rm -rf build

mask: clean
	pf mask write

show:
	pf show build/mask/sample_mask.gds


.PHONY: devices
