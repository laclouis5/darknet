import os
from my_xml_toolbox import XMLTree, XMLTree2
from lxml.etree import Element, SubElement, tostring, parse


def reformat(folder, save_dir):
    xml_files = [item for item in os.listdir(folder) if os.path.splitext(item)[1] == ".xml"]
    print(xml_files)

    for item in xml_files:
        file = os.path.join(folder, item)
        tree = parse(file).getroot()
        dl_document = tree.find("DL_DOCUMENT")
        user = tree.find("USER")
        mask_zones = dl_document.findall("MASK_ZONE")

        new_tree = Element("GEDI")
        dl_document_2 = SubElement(new_tree, "DL_DOCUMENT")
        user_2 = SubElement(new_tree, "USER")

        user_2.attrib["name"] = user.find("NAME").text
        user_2.attrib["date"] = user.find("DATE").text

        dl_document_2.attrib["src"] = dl_document.find("SRC").text
        dl_document_2.attrib["docTag"] = "xml"
        dl_document_2.attrib["width"] = dl_document.find("WIDTH").text
        dl_document_2.attrib["height"] = dl_document.find("HEIGHT").text

        for mask in mask_zones:
            mask_2 = SubElement(dl_document_2, "MASQUE_ZONE")
            mask_2.attrib["id"] = mask.find("ID").text
            mask_2.attrib["type"] = mask.find("TYPE").text
            mask_2.attrib["name"] = mask.find("NAME").text

        save_name = os.path.join(save_dir, os.path.basename(file))
        string = tostring(new_tree, encoding='unicode', pretty_print=True)

        with open(save_name, "w") as f:
            f.write(string)


if __name__ == "__main__":
    reformat("/Users/louislac/Downloads/bipbip_haricot", "/Users/louislac/Downloads/new_haricot")
