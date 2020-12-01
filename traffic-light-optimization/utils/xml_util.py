import copyreg

import lxml.etree as etree


copyreg_registered = False

parser = etree.XMLParser(remove_blank_text=True)


def get_xml(file):

    xml = etree.parse(file, parser)

    return xml


def etree_unpickler(data):
    return etree.fromstring(data)


def etree_pickler(tree):
    data = etree.tostring(tree)
    return etree_unpickler, (data,)


def register_copyreg():
    global copyreg_registered

    if not copyreg_registered:
        copyreg.pickle(etree._ElementTree, etree_pickler, etree_unpickler)
        copyreg.pickle(etree._Element, etree_pickler, etree_unpickler)
        copyreg_registered = True


def rename_xml_string(element, old_string, new_string, parser=None):
    if parser is None:
        parser = etree.XMLParser(remove_blank_text=True)

    # method 'c14n2' doesn't add namespace to xml
    string_content = etree.tostring(element, method='c14n2')

    string_content = string_content.decode().replace(old_string, new_string).encode()

    renamed_element = etree.fromstring(string_content, parser)

    return renamed_element