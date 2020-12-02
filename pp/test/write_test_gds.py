from pp.components import _components


# print(len(_components))
components = _components - {"test_via", "test_resistance"}
# print(len(components))

imports = """
import pp
from lytest import contained_phidlDevice, difftest_it
"""

if __name__ == "__main__":
    with open("test_gds.py", "w") as f:
        f.write(imports)
        for component_type in sorted(list(components)):
            f.write(
                f"""

@contained_phidlDevice
def {component_type}(TOP):
    pp.clear_cache()
    TOP.add_ref(pp.c.{component_type}())


def test_gds_{component_type}():
    difftest_it({component_type})()
"""
            )
