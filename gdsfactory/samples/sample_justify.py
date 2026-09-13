if __name__ == "__main__":
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    c = gf.Component("fixed_justify")
    text1 = gf.components.text_freetype("centered", size=100, justify="center")
    text2 = gf.components.text_freetype("left", size=100, justify="left")
    text3 = gf.components.text_freetype("right", size=100, justify="right")
    text4 = gf.components.text("1234567890", size=100, justify="center")
    c.add_ref(text1)
    c.add_ref(text2).movey(-100)
    c.add_ref(text3).movey(-200)
    c.add_ref(text4).movey(-300)
    c.show()
