from xml.etree.ElementTree import Element, SubElement, tostring, parse
import datetime
import os

class XMLTree:
    def __init__(self, image_name, width, height, user_name='Unknown user', date=str(datetime.date.today())):
        """
        Creates a ElementTree containing results of classification.

        Arguments:
        image_name -- String containing the name of the source image used for classification.
        width      -- Image width (integer).
        height     -- Image height (integer).
        user_name  -- String containing the name of the participant.
        date       -- String containing the result production date (YYYY-DD-MM).

        Returns:
        A ElementTree Element

        For additional information about format please refer to 'PE_ROSE_dryrn.pdf'
        """

        # Root node
        self.tree = Element('GEDI')

        # First level nodes
        document = SubElement(self.tree, 'DL_DOCUMENT')
        user = SubElement(self.tree, 'USER')

        # Second level nodes
        src = SubElement(document, 'SRC')
        tag = SubElement(document, 'DOC_TAG')
        w = SubElement(document, 'WIDTH')
        h = SubElement(document, 'HEIGHT')

        _name = SubElement(user, 'NAME')
        _date = SubElement(user, 'DATE')

        # Fill info
        src.text = image_name
        tag.text = 'xml'
        w.text = str(width)
        h.text = str(height)

        _name.text = user_name
        _date.text = date


    def add_mask_zone(self, plant_type, bbox, name=''):
        """
        Creates new mask zones in the input tree.

        Arguments:
        tree       -- The ElementTree to be filled with mask info.
        plant_type -- String containing the type of plant (either 'Adventice' or 'PlanteInteret')
        name       -- String (optional), plant name.
        """
        # Go to 'DL_DOCUMENT' node & retreive mask ID
        doc_node = self.tree.find('DL_DOCUMENT')
        nb_masks = self.get_next_mask_id()

        # Create new mask
        mask = Element('MASK_ZONE')
        _id = SubElement(mask, 'ID')
        _type = SubElement(mask, 'TYPE')
        _name = SubElement(mask, 'NAME')
        _bndbox = SubElement(mask, 'BNDBOX')

        _xmin = SubElement(_bndbox, 'XMIN')
        _ymin = SubElement(_bndbox, 'YMIN')
        _xmax = SubElement(_bndbox, 'XMAX')
        _ymax = SubElement(_bndbox, 'YMAX')

        # Fill info & append
        _id.text = str(nb_masks)
        _type.text = plant_type
        _name.text = name

        _xmin.text = str(bbox[0])
        _ymin.text = str(bbox[1])
        _xmax.text = str(bbox[2])
        _ymax.text = str(bbox[3])

        doc_node.append(mask)


    def get_next_mask_id(self):
        """
        Return the next unique ID for masks.
        """

        # Go to 'DL_DOCUMENT' node
        doc_node = self.tree.find('DL_DOCUMENT')

        # Compute new mask index
        masks_list = doc_node.findall('MASK_ZONE')
        return len(masks_list)


    def get_current_mask_id(self):
        return self.get_next_mask_id() - 1


    def save(self, xml_file_name):
        """
        Saves to disk the tree to XML file.

        Arguments:
        tree -- The ElementTree Element to save.
        xml_file_name -- String containing the name of the XML file. See 'PE_ROSE_dryrn.pdf' for more information.
        """

        # TreeElement to string representation
        tree_str = tostring(self.tree, encoding='unicode')

        # Write data
        with open(xml_file_name, 'w') as xml_file:
            xml_file.write(tree_str)


    def clean_xml(folders):
        for folder in folders:
            for file in os.listdir(folder):

                if(os.path.splitext(file)[1] != '.xml'):
                    continue

                tree = parse(os.path.join(folder, file)).getroot()

                path_field = tree.find('path')
                path_field.text = os.path.join(folder, file)

                with open(os.path.join(folder, file), 'w') as xml_file:
                    tree_str = tostring(tree, encoding='unicode')
                    xml_file.write(tree_str)


    def xlm_to_csv(folders, classes_to_keep=[], cvs_path=''):
        with open(os.path.join(csv_path, 'train_data.csv'), 'w') as csv_file:
            for folder in folders:
                for file in sorted(os.listdir(folder)):
                    # Check if XML file
                    if(os.path.splitext(file)[1] != '.xml'):
                        continue
                    # Retreive the XML etree
                    root = parse(os.path.join(folder, file)).getroot()

                    file_name = root.find('filename').text

                    # Retreive and process each 'object' in the etree
                    for obj in root.findall('object'):
                        name = obj.find('name').text

                        # Save only selected classes
                        # Comment to ignore class selection
                        if (classes_to_keep.count != 0) and (name not in classes_to_keep):
                            continue

                        # Retreive bounding box coordinates
                        bounding_box = obj.find('bndbox')
                        coords = []

                        for coord in bounding_box.getchildren():
                            coords.append(int(coord.text))

                        # Write CSV file
                        csv_file.write(os.path.join(folder, file_name) + ',')
                        for coord in coords:
                            csv_file.write(str(coord) + ',')
                        csv_file.write(name + '\n')
