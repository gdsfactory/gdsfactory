import time

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    time_start = time.time()
    xsize = ysize = 100
    for i in range(1, 100):
        r = c << gf.components.mzi(delta_length=i)
        r.dmove((i % 10 * xsize, i // 10 * ysize))

    print(f"Elapsed time: {time.time() - time_start:.2f} s")
    c.show()
