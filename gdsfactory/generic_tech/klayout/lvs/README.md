# LVS Documentation

Explains how to use the runset.

## Folder Structure

```text
📁 lvs
 ┣ 📁testing                        Testing environment directory for GENERIC TECH LVS.
 ┣ 📁rule_decks                     All LVS rule decks used in GENERIC TECH.
 ┣ generic_tech.lvs                 Main LVS rule deck that call all runsets.
 ┣ 📜README.md                      This file to document the LVS run for GENERIC TECH.
 ┗ 📜run_lvs.py                     Main python script used for GENERIC TECH LVS.
 ```

## **Prerequisites**
You need the following set of tools installed to be able to run GENERIC TECH LVS:
- Python 3.6+
- KLayout 0.28.4+

## **Usage**

The `run_lvs.py` script takes your input gds and netlist files to run LVS rule deck of GENERIC TECH technology on it with switches to select subsets of all checks.

```bash
    run_lvs.py (--help| -h)
    run_lvs.py (--layout=<layout_path>) (--netlist=<netlist_path>) [--thr=<thr>] [--run_dir=<run_dir_path>] [--topcell=<topcell_name>] [--run_mode=<run_mode>] [--verbose] [--lvs_sub=<sub_name>] [--no_net_names] [--spice_comments] [--scale] [--schematic_simplify] [--net_only] [--top_lvl_pins] [--combine] [--purge] [--purge_nets]
```

Example:
```bash
    python3 testing/testcases/unit/heater_devices/layout/straight_heater_metal.py
    python3 run_lvs.py --layout=testing/testcases/unit/heater_devices/layout/straight_heater_metal.gds --netlist=testing/testcases/unit/heater_devices/netlist/straight_heater_metal.spice --run_mode=deep --run_dir=lvs_straight_heater_metal
```

### Options

- `--help -h`                           Print this help message.

- `--layout=<layout_path>`              The input GDS file path.

- `--netlist=<netlist_path>`            The input netlist file path.

- `--thr=<thr>`                         The number of threads used in run.

- `--run_dir=<run_dir_path>`            Run directory to save all the results [default: pwd]

- `--topcell=<topcell_name>`            Topcell name to use.

- `--run_mode=<run_mode>`               Select klayout mode Allowed modes (flat , deep, tiling). [default: flat]

- `--lvs_sub=<sub_name>`                Substrate name used in your design.

- `--verbose`                           Detailed rule execution log for debugging.

- `--no_net_names`                      Discard net names in extracted netlist.

- `--spice_comments`                    Enable netlist comments in extracted netlist.

- `--scale`                             Enable scale of 1e6 in extracted netlist.

- `--schematic_simplify`                Enable schematic simplification in input netlist.

- `--net_only`                          Enable netlist object creation only in extracted netlist.

- `--top_lvl_pins`                      Enable top level pins only in extracted netlist.

- `--combine`                           Enable netlist combine only in extracted netlist.

- `--purge`                             Enable netlist purge all only in extracted netlist.

- `--purge_nets`                        Enable netlist purge nets only in extracted netlist.


## **LVS Outputs**

You could find the run results at your run directory if you previously specified it through `--run_dir=<run_dir_path>`. Default path of run directory is `lvs_run_<date>_<time>` in current directory.

### Folder Structure of run results

```text
📁 lvs_run_<date>_<time>
 ┣ 📜 lvs_run_<date>_<time>.log
 ┗ 📜 <your_design_name>.cir
 ┗ 📜 <your_design_name>.lvsdb
 ```

The result is a database file (`<your_design_name>.lvsdb`) contains LVS extractions and comparison results.
You could view it on your file using: `klayout <input_gds_file> -mn <resut_db_file> `, or you could view it on your gds file via netlist browser option in tools menu using klayout GUI.

You could also find the extracted netlist generated from your design at (`<your_design_name>.cir`) in your run directory.

## **DRC-malformed**

You need to make sure that your design is passing [DRC-malformed](drc_malformed/README.md) runset, to be able to use LVS properly.
