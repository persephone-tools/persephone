""" Interface to the Japhug data. """

import os
import shutil
import subprocess
from xml.etree import ElementTree

import config
import feat_extract
import utils

ORG_DIR = config.JAPHUG_DIR
TGT_DIR = os.path.join(config.TGT_DIR, "japhug")

PHONEMES = ["a", "e", "i", "o", "u", "ɯ", "y",
            "k", "kh", "g", "ŋg", "ŋ", "x", "ɣ",
            "t", "th", "d", "nd", "n",
            "ts", "tsh", "dz", "ndz", "s", "z",
            "tɕ", "tɕh", "dʑ", "ndʑ", "ɕ", "ʑ",
            "tʂ", "tʂh", "dʐ", "ndʐ", "ʂ", "", "r",
            "c", "ch", "ɟ", "ɲɟ", "ɲ", "j"
            "p", "ph", "b", "mb", "m", "w", "β",
            "h"]
PHONEMES_LEN_SORTED = sorted(PHONEMES, key=lambda x: len(x), reverse=True)
print(PHONEMES_LEN_SORTED)

def extract_phonemes(sent, tgt_fn):
    print("Phonemes")

    with open(tgt_fn, "w") as tgt_f:
        words = sent.split()
        input()

def extract_sents_and_times(fn):
    sentences = []
    times = []
    tree = ElementTree.parse(fn)
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
                times.append((start_time, end_time))
                sentences.append(sentence)
    return sentences, times

# This time I will have prepare function that is separate to the Corpus
# class
def prepare(feat_type="log_mel_filterbank"):
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
    if not os.path.isdir(audio_utter_dir):
        os.makedirs(audio_utter_dir)
    audio_fns = os.listdir(audio_dir)
    transcript_dir = os.path.join(TGT_DIR, "transcriptions")
    transcript_utter_dir = os.path.join(transcript_dir, "utterances")
    if not os.path.isdir(transcript_utter_dir):
        os.makedirs(transcript_utter_dir)
    for fn in os.listdir(transcript_dir):
        pre, ext = os.path.splitext(fn)
        if ext == ".xml":
            # Assumes file name is like "crdo-JYA_DIVINATION.xml"
            rec_name = pre.split("_")[-1]
            sentences, times = extract_sents_and_times(
                os.path.join(transcript_dir, fn))
            assert len(sentences) == len(times)

            for audio_fn in audio_fns:
                pre, ext = os.path.splitext(audio_fn)
                if ext == ".wav":
                    if rec_name in audio_fn:
                        src_fn = os.path.join(audio_dir, audio_fn)

            for i, sentence in enumerate(sentences):

                start_time, end_time = times[i]

                tgt_fn = os.path.join(
                    audio_utter_dir, "%s.%d.wav" % (rec_name, i))

                if rec_name == "LWLU":
                    print(src_fn)
                    print(tgt_fn)
                    print(start_time)
                    print(end_time)

                # Trim the audio
                utils.trim_wav(src_fn, tgt_fn, start_time, end_time)

                # Extract the phones.
                extract_phonemes(sentence, os.path.join(transcript_utter_dir,
                                 "%s.%d.phn" % (rec_name, i)))

    # Extract features from the WAV files.
    #feat_extract.from_dir(audio_utter_dir, feat_type)
