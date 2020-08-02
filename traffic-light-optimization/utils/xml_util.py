
import lxml.etree as etree


def rename_xml_string(element, old_string, new_string, parser=None):
    if parser is None:
        parser = etree.XMLParser(remove_blank_text=True)

    # method 'c14n2' doesn't add namespace to xml
    string_content = etree.tostring(element, method='c14n2')

    string_content = string_content.decode().replace(old_string, new_string).encode()

    renamed_element = etree.fromstring(string_content, parser)

    return renamed_element