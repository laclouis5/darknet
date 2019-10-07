# Created by Louis LAC 2019

from lxml.etree import Element, SubElement, tostring, parse
import datetime
import os

class XMLTree:
    def __init__(self, image_name, width, height, user_name="Bipbip", date=datetime.date.today()):
        """
        Instantiates a XML Tree representation to hold object detection results.
        """
        # XML Tree Structure
        self.tree = Element("GEDI")
        dl_document = SubElement(self.tree, "DL_DOCUMENT")
        user = SubElement(self.tree, "USER")

        user.attrib["name"] = user_name
        user.attrib["date"] = str(date)

        dl_document.attrib["src"] = image_name
        dl_document.attrib["docTag"] = "xml"
        dl_document.attrib["width"] = str(width)
        dl_document.attrib["height"] = str(height)

        # Helper Properties
        bboxes = []
        self.plant_count = 0
        self.save_name = user_name + "_" + os.path.splitext(image_name)[0] + ".xml"


    def add_mask(self, name, type="PlanteInteret"):
        """
        Adds a detection. ID points to a PNG mask representing the detected
        object.
        """
        dl_document = self.tree.find("DL_DOCUMENT")
        mask = SubElement(dl_document, "MASQUE_ZONE")
        mask.attrib["id"] = str(self.plant_count)
        mask.attrib["type"] = type
        mask.attrib["name"] = name

        self.plant_count += 1


    def save(self, save_dir=""):
        """
        Saves the xml file.
        """
        string = tostring(self.tree, encoding='unicode', pretty_print=True)
        save_name = os.path.join(save_dir, self.save_name)

        with open(save_name, 'w') as f:
            f.write(string)
