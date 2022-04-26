import gdsfactory as gf

sample_rotation = """

name: sample_rotation

instances:
  r1:
    component: rectangle
    settings:
        size: [4, 2]
  r2:
    component: rectangle
    settings:
        size: [2, 4]

placements:
    r1:
        xmin: 0
        ymin: 0
    r2:
        rotation: -90
        xmin: r1,east
        ymin: 0

"""


if __name__ == "__main__":
    c = gf.read.from_yaml(sample_rotation)
    c.show()
