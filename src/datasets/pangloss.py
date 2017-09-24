""" Some functions to interface with the Pangloss """

from xml.etree import ElementTree

def get_sents_times_and_translations(xml_fn):
    """ Given an XML filename, loads the transcriptions, their start/end times,
    and translations. """

    tree = ElementTree.parse(xml_fn)
    root = tree.getroot()
    if root.tag == "WORDLIST" or root.tag == "TEXT":
        transcriptions = []
        times = []
        translations = []
        for child in root:
            if child.tag == "S" or child.tag == "W":
                forms = child.findall("FORM")
                if len(forms) > 1:
                    # Assuming there is kindOf="phono" available.
                    for form in forms:
                        if form.attrib["kindOf"] == "phono":
                            transcription = form.text
                else:
                    transcription = child.find("FORM").text
                audio_info = child.find("AUDIO")
                if audio_info != None:
                    start_time = float(audio_info.attrib["start"])
                    end_time = float(audio_info.attrib["end"])
                    time = (start_time, end_time)
                    translation = [trans.text for trans in child.findall("TRANSL")]
                    transcriptions.append(transcription)
                    times.append(time)
                    translations.append(translation)

        return transcriptions, times, translations
    return [], [], []

def remove_content_in_brackets(sentence, brackets="[]"):
    out_sentence = ''
    skip_c = 0
    for c in sentence:
        if c == brackets[0]:
            skip_c += 1
        elif c == brackets[1] and skip_c > 0:
            skip_c -= 1
        elif skip_c == 0:
            out_sentence += c
    return out_sentence
