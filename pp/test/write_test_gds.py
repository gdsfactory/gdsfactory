from pp.components import _components
from pp.test_containers import container_factory

# print(len(_components))
components = _components - {"test_via", "test_resistance"}
# print(len(components))

imports = """
import pp
from pp.test_containers import container_factory
from lytest import contained_phidlDevice, difftest_it
"""

if __name__ == "__main__":
    with open("test_gds.py", "w") as f:
        f.write(imports)
        for component_type in sorted(list(components)):
            f.write(
                f"""

@contained_phidlDevice
def {component_type}(top):
    # pp.clear_cache()
    top.add_ref(pp.c.{component_type}())


def test_gds_{component_type}():
    difftest_it({component_type})()
"""
            )
        for container_type in container_factory.keys():
            f.write(
                f"""

@contained_phidlDevice
def {container_type}(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["{container_type}"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_{container_type}():
    difftest_it({container_type})()
"""
            )
