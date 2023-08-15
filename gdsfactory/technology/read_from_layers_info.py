from gdsfactory.typings import Layer, PathType


def extract_layers(input_text: str) -> dict[str, Layer]:
    # Split the input text by lines
    lines = input_text.split("\n")

    # A dictionary to store the mapping of the layer name to its tuple
    layer_mapping = {}

    # For each line in the text
    for line in lines:
        # Split the line by spaces
        parts = line.split()
        if line.startswith("#"):
            continue

        # Check if the line has the expected number of parts (columns)
        if len(parts) == 7:  # Assuming your data structure remains consistent
            layer_name = parts[0]
            layer_num = parts[4]
            data_type = parts[5]

            # Convert layer number and data type to integers
            try:
                layer_num = int(layer_num)
                data_type = int(data_type)

                # Store the layer name mapping
                layer_mapping[layer_name] = (layer_num, data_type)

            except (
                ValueError
            ):  # This handles lines that don't have expected number format
                print(f"Error parsing line: {line}")
                continue

    return layer_mapping


def read_from_layers_info(filepath: PathType) -> str:
    """Returns a layermap python script from layers.info file"""
    input_text = open(filepath).read()
    layer_mapping = extract_layers(input_text)
    output = "class LayerMap(BaseModel):\n"

    for layer_name, value_tuple in layer_mapping.items():
        output += f"    {layer_name}: Layer = {value_tuple}\n"

    return output
