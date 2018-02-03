""" Interface to the Japhug data. """

import os
import random
import shutil
import subprocess
from xml.etree import ElementTree

import numpy as np
from sklearn import preprocessing

from .. import config
from .. import corpus
from .. import feat_extract
from .. import utils
from . import pangloss

ORG_DIR = config.JAPHUG_DIR
TGT_DIR = os.path.join(config.TGT_DIR, "japhug")

PHONEMES = ["a", "e", "i", "o", "u", "ɯ", "y",
            "k", "kh", "g", "ŋg", "ŋ", "x", "ɤ",
            "t", "th", "d", "nd", "n",
            "ts", "tsh", "dz", "ndz", "s", "z",
            "tɕ", "tɕh", "dʑ", "ndʑ", "ɕ", "ʑ",
            "tʂ", "tʂh", "dʐ", "ndʐ", "ʂ", "r",
            "c", "ch", "ɟ", "ɲɟ", "ɲ", "j",
            "p", "ph", "b", "mb", "m", "w", "β",
            "h",
            "q", "qh", "ɴɢ", "χ", "ʁ",
            "f", "l", "ɬ", "ɴ"]
PHONEMES_THREE_CHAR = set([phn for phn in PHONEMES if len(phn) == 3])
PHONEMES_TWO_CHAR = set([phn for phn in PHONEMES if len(phn) == 2])
PHONEMES_ONE_CHAR = set([phn for phn in PHONEMES if len(phn) == 1])

def extract_phonemes(sent, tgt_fn):

    org_sent = sent
    sent = datasets.pangloss.remove_content_in_brackets(sent, "()")
    sent = sent.replace(",", " ")
    sent = sent.replace("，", " ")
    sent = sent.replace("-", " ")
    sent = sent.replace('"', " ")
    sent = sent.replace('...', " ")
    sent = sent.replace('.', " ")
    sent = sent.replace("ú", "u")
    sent = sent.replace("ú", "u")
    sent = sent.replace("ɯ́", "ɯ")
    sent = sent.replace("á", "a")
    sent = sent.replace("á", "a")
    sent = sent.replace("í", "i")
    sent = sent.replace("í", "i")
    sent = sent.replace("é", "e")
    sent = sent.replace("é", "e")
    sent = sent.replace("ó", "o")
    sent = sent.replace("ó", "o")
    sent = sent.replace("ɤ́", "ɤ")
    sent = sent.replace("ɣ", "ɤ")
    sent = sent.replace("?", " ")
    sent = sent.replace("!", " ")
    sent = sent.replace("[", " ")
    sent = sent.replace("]", " ")
    sent = sent.replace("{", " ")
    sent = sent.replace("}", " ")
    sent = sent.replace("#", " ")
    sent = sent.replace("mɢ", "mɴɢ")
    phonemes = []
    with open(tgt_fn, "w") as tgt_f:
        words = sent.split()
        for word in words:
            i = 0
            while i < len(word):
                if word[i:i+3] in PHONEMES_THREE_CHAR:
                    phonemes.append(word[i:i+3])
                    i += 3
                    continue
                elif word[i:i+2] in PHONEMES_TWO_CHAR:
                    phonemes.append(word[i:i+2])
                    i += 2
                    continue
                elif word[i:i+1] in PHONEMES_ONE_CHAR:
                    phonemes.append(word[i:i+1])
                    i += 1
                    continue
                else:
                    print("Filename:\n\t", tgt_fn)
                    print("original sentence:\n\t", org_sent)
                    print("Preprocessed sentence:\n\t", sent)
                    print("Word remainder:\n\t", word[i:])
                    raise Exception("Failed to segment word: %s" % word)
    with open(tgt_fn, "w") as out_f:
        print(" ".join(phonemes), file=out_f)

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

                # Trim the audio
                utils.trim_wav(src_fn, tgt_fn, start_time, end_time)

                # Extract the phones.
                extract_phonemes(sentence, os.path.join(transcript_utter_dir,
                                 "%s.%d.phn" % (rec_name, i)))

    # Extract features from the WAV files.
    feat_extract.from_dir(audio_utter_dir, feat_type)

class Corpus(corpus.AbstractCorpus):
    """ Class interface to the Japhug corpus. """

    TGT_DIR = TGT_DIR

    TRAIN_VALID_SPLIT = [800, 157]
    _phonemes = None
    tones = False

    def __init__(self, feat_type, target_type, max_samples=1000, normalize=False):
        super().__init__(feat_type, target_type)

        transcript_dir = os.path.join(
            config.TGT_DIR, "japhug", "transcriptions", "utterances")
        audio_dir = os.path.join(
            config.TGT_DIR, "japhug", "audio", "utterances")

        self.prefixes = [os.path.join(audio_dir, os.path.splitext(fn)[0])
                         for fn in os.listdir(transcript_dir)]

        if max_samples:
            self.prefixes = self.sort_and_filter_by_size(self.prefixes,
                                                         max_samples)

        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_SPLIT[0]]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_SPLIT[0]:]

        self.phonemes = PHONEMES
        self.PHONEME_TO_INDEX = {phn: index for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.INDEX_TO_PHONEME = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}

        self.vocab_size = len(self.phonemes)

        # Potentially normalize the data to have zero mean and unit variance.
        if normalize:
            self.prepare_normalized()
            self.normalized=normalize


    def prepare_normalized(self):
        """ Normalizes each element of the input vectors so that they have zero
        mean and unit variance over the whole training corpus.
        """

        def load_stacked_utters(fns):

            # Load the utterances
            utters = [np.load(fn) for fn in fns]

            # Reshape each utterance to collapse derivative dimensions
            for utter in utters:
                utter.shape = (utter.shape[0], utter.shape[1]*utter.shape[2])

            # Records the duration of each utterance for later reconstructions.
            lens = [a.shape[0] for a in utters]

            # Stack them all into one array and return, along with the original
            # lengths.
            return np.vstack(utters), lens

        def write_stacked_utters(stacked, lens, src_fns):

            # Unstack them into separate arrays.
            utters = []
            for length in lens:
                utters.append(stacked[:length])
                # TODO These dimensions shouldn't be hardcoded.
                utters[-1].shape = (length, 41, 3)
                stacked = stacked[length:]

            # Write the normalized set.
            fns_utters = zip(src_fns, utters)
            for src_fn, utter in fns_utters:
                pre, ext = os.path.splitext(src_fn)
                norm_fn = "%s.norm%s" % (pre, ext)
                np.save(norm_fn, utter)

        train_fns = self.get_train_fns()[0]
        valid_fns = self.get_valid_fns()[0]
        stacked_train, train_lens = load_stacked_utters(train_fns)
        stacked_valid, valid_lens = load_stacked_utters(valid_fns)

        # Scale the data based on the training distribution.
        scaler = preprocessing.StandardScaler().fit(stacked_train)
        norm_train = scaler.transform(stacked_train)
        norm_valid = scaler.transform(stacked_valid)

        write_stacked_utters(norm_train, train_lens, train_fns)
        write_stacked_utters(norm_valid, valid_lens, valid_fns)
