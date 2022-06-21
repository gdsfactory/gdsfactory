import gdsfactory as gf


if __name__ == "__main__":
    gdspath = gf.CONFIG["gdsdir"] / "alphabet.gds"
    c = gf.import_gds(gdspath, demo="hi")
    print(c.info)
    c.show(show_ports=True)
