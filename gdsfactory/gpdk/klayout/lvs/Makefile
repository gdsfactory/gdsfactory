heater-write:
	python3 testing/testcases/unit/heater_devices/layout/straight_heater_metal.py

heater-lvs:
	python3 run_lvs.py --layout=testing/testcases/unit/heater_devices/layout/straight_heater_metal.gds --netlist=testing/testcases/unit/heater_devices/netlist/straight_heater_metal.spice --run_mode=flat --run_dir=lvs_straight_heater_metal

heater-open:
	klayout testing/testcases/unit/heater_devices/layout/straight_heater_metal.gds -mn lvs_straight_heater_metal/straight_heater_metal.lvsdb
