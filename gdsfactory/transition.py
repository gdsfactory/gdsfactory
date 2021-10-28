import pydantic

from gdsfactory.cross_section import CrossSection


@pydantic.dataclasses.dataclass
class Transition:
    cross_section1: CrossSection
    cross_section2: CrossSection
    width_type: str = "sine"


if __name__ == "__main__":
    pass
