import xml.etree.ElementTree as ET
from xml.dom.minidom import Node, parseString

from gdsfactory.technology.xml_utils import make_pretty_xml, strip_xml


def test_make_pretty_xml() -> None:
    root = ET.Element("root")
    child = ET.SubElement(root, "child")
    child.text = "   text with spaces   "
    grandchild = ET.SubElement(child, "grandchild")
    grandchild.text = "  nested text  "

    pretty_xml = make_pretty_xml(root)

    xml_str = pretty_xml.decode("utf-8")

    assert "<?xml" in xml_str
    assert "<root>" in xml_str
    assert "<child>" in xml_str
    assert "text with spaces" in xml_str
    assert "<grandchild>" in xml_str
    assert "nested text" in xml_str
    assert " " * 1 + "<child>" in xml_str
    assert " " * 2 + "<grandchild>" in xml_str


def test_strip_xml() -> None:
    xml_str = "<root>  <child>  text  </child>  </root>"
    doc = parseString(xml_str)

    text_node = doc.createTextNode("  text with spaces  ")
    assert text_node.nodeType == Node.TEXT_NODE
    strip_xml(text_node)
    assert "text with spaces" in text_node.nodeValue

    element = doc.createElement("test")
    assert element.nodeType == Node.ELEMENT_NODE
    strip_xml(element)

    strip_xml(doc)

    child = doc.getElementsByTagName("child")[0]
    assert child.firstChild is not None
    assert child.firstChild.nodeValue == "text"


if __name__ == "__main__":
    test_make_pretty_xml()
    test_strip_xml()
