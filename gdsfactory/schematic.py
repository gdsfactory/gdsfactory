from gdsfactory.types import ComponentModel, Dict, List, Optional


class Schematic:
    instances: Optional[Dict[str, ComponentModel]] = None
    nets: Optional[List[List[str]]] = None
    ports: Optional[Dict[str, str]] = None


if __name__ == "__main__":
    s = Schematic()
