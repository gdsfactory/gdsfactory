mask: clean devices does merge labels
help:
	@echo 'make :                 makes devices, merges them'
	@echo 'make devices:          makes devices'
	@echo 'make does:             makes DOEs'
	@echo 'make merge:            merges all devices GDS and JSON'
	@echo 'make show:             show mask in klayout'
	@echo 'make upload            upload mask to maskhub'


venv:
	python3 -m venv env

clean:
	rm -rf build

devices:
	pf mask build_devices

does:
	pf mask build_does

merge:
	pf mask merge

labels:
	pf mask write_labels

show:
	pf show build/mask/sample_mask.gds

show2:
	pf show build/mask/sample_mask_remapped.gds

upload:
	mh post mask

fill:
	python fill.py

.PHONY: devices
