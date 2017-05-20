""" Some functions to interface with the Pangloss """

from xml.etree import ElementTree

def sents_times_translations(xml_fn):
    """ Given an XML filename, loads the transcriptions, their start/end times,
    and translations. """

    tree = ElementTree.parse(xml_fn)
    root = tree.getroot()
    assert root.tag == "TEXT"

    for child in root:
        if child.tag == "S":
            assert len(child.findall("FORM")) == 1
            transcription = child.find("FORM").text
            audio_info = child.find("AUDIO")
            if audio_info != None:
                start_time = float(audio_info.attrib["start"])
                end_time = float(audio_info.attrib["end"])
                time = (start_time, end_time)
            translations = [trans.text for trans in child.findall("TRANSL")]
            if translations == None:
                print("no translation")
            print("transcription: ", transcription)
            print("start, end: ", time)
            print("translations", translations)
            input()
