from gdsfactory.samples.pdk.fab_e.schema import (
    CrossSectionModel,
    SectionModel,
    TechModel,
)

layers = dict(wg=(1, 0), wg_clad=(1, 2), pin=(100, 0), heater=(11, 0))
xs_strip = CrossSectionModel()
xs_strip_heater_metal = CrossSectionModel(
    width=0.5, sections=[SectionModel(layer=layers["heater"], width=2.5)]
)
cross_sections = dict(xs_strip=xs_strip, xs_strip_heater_metal=xs_strip_heater_metal)
TECH = TechModel(cross_sections=cross_sections, layers=layers)


if __name__ == "__main__":
    print(xs_strip_heater_metal)
