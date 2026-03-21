import gdsfactory as gf

radius = 25.0
offsets = [10, 30, 50, 75, 100, 200, 500]

if __name__ == "__main__":
    top = gf.Component("bend_s_offset_bug")
    x = 0.0
    for offset in offsets:
        bend = gf.c.bend_s_offset(offset=offset, radius=radius, p=1)
        ref = top.add_ref(bend)
        ref.xmin = x
        error = bend.ysize - offset
        label = f"off={offset}" + (f"\nBUG({error:+.0f})" if abs(error) > 1 else "")
        _ = top << gf.c.text(label, size=12, layer="TEXT")
        _.move((x - 10, -20))
        x += bend.xsize + 30

    top.show()
    print(f"({len(offsets)} bends, xsize={top.xsize:.0f} um)")
