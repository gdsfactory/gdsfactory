import gdsfactory as gf

c = gf.Component()
c.info.update({"test_key": "test_value"})

c3 = gf.Component()
ref = c3 << c
gf.add_pins.add_settings_label(component=c3, reference=ref)
c3.show()
labels = c3.get_labels("LABEL_SETTINGS")

print(labels[0])
