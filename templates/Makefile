mask: clean devices does merge labels metadata 
help:
	@echo 'make :                 makes devices, merges them'
	@echo 'make devices:          makes devices'
	@echo 'make does:             makes DOEs'
	@echo 'make merge:            merges all devices GDS and JSON'
	@echo 'make show:             show mask in klayout'
	@echo 'make upload            upload mask to maskhub'

install: 
	pip install -r requirements.txt
	pip install -e .

venv:
	python3 -m venv env

clean:
	rm -rf build

devices:
	pf build devices

does:
	python write_does.py

merge:
	python merge_gds.py

metadata:
	pf mask merge

labels:
	pf mask write_labels

show:
	pf show build/mask/sample_mask.gds

upload:
	mh post mask

fill:
	python fill.py

.PHONY: devices

