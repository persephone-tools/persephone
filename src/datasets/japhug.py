""" Interface to the Japhug data. """

import os
import shutil
import subprocess
from xml.etree import ElementTree

import config

ORG_DIR = config.JAPHUG_DIR
TGT_DIR = os.path.join(config.TGT_DIR, "japhug")

def extract_audio_snippet(src_fn, tgt_fn, start, end):
    print("Audio")
    print(src_fn, tgt_fn, start, end)
    pass

def extract_phonemes(src_fn, tgt_fn):
    print("Phonemes")
    print(src_fn, tgt_fn)
    pass

# This time I will have  prepare function that is separate to the Corpus
# class
def prepare():
    """ Prepares the Japhug data for downstream use. """

    # Copy the data across to the tgt dir. Make the tgt dir if necessary.
    if not os.path.isdir(TGT_DIR):
        shutil.copytree(ORG_DIR, TGT_DIR)

    # Convert mp3 to wav.
    audio_path = os.path.join(TGT_DIR, "audio")
    for fn in os.listdir(audio_path):
        pre, ext = os.path.splitext(fn)
        if ext == ".mp3":
            if not os.path.exists(os.path.join(audio_path, pre + ".wav")):
                args = [config.FFMPEG_PATH,
                        "-i", os.path.join(audio_path, fn),
                        "-ar", str(16000), "-ac", str(1),
                        os.path.join(audio_path, pre + ".wav")]
                subprocess.run(args)

    # Extract phonemes from XML transcripts, and split the WAVs based on the
    # time indications in the XML.
    audio_dir = os.path.join(TGT_DIR, "audio")
    audio_utter_dir = os.path.join(audio_dir, "utterances")
    audio_fns = os.listdir(audio_dir)
    transcript_dir = os.path.join(TGT_DIR, "transcriptions")
    transcript_utter_dir = os.path.join(transcript_dir, "utterances")
    for fn in os.listdir(transcript_dir):
        pre, ext = os.path.splitext(fn)
        if ext == ".xml":
            # Assumes file name is like "crdo-JYA_DIVINATION.xml"
            rec_name = pre.split("_")[-1]
            tree = ElementTree.parse(os.path.join(transcript_dir, fn))
            root = tree.getroot()
            assert root.tag == "TEXT"

            i = 0

            for child in root:
                if child.tag == "S":
                    assert len(child.findall("FORM")) == 1
                    sentence = child.find("FORM").text
                    audio = child.find("AUDIO")
                    if audio != None:
                        start_time = float(audio.attrib["start"])
                        end_time = float(audio.attrib["end"])

                        for audio_fn in audio_fns:
                            if rec_name in audio_fn:
                                src_fn = os.path.join(audio_dir, audio_fn)
                        tgt_fn = os.path.join(
                            audio_utter_dir, "%s.%d.wav" % (rec_name, i))

                        # Trim the audio
                        extract_audio_snippet(src_fn, tgt_fn,
                                              start_time, end_time)

                        # Extract the phones.
                        extract_phonemes(
                            sentence, os.path.join(transcript_utter_dir,
                            "%s.%d.phn" % (rec_name, i)))

                    i += 1

    # Extract features from the WAV files.
