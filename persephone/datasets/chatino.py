""" Interface to the Chatino data."""

from os.path import join
import os
import random
from shutil import copyfile

from .. import config
from .. import corpus
from .. import feat_extract
from .. import utils

### Initialize path variables.###
# Path to unpreprocessed Chatino data from GORILLA.
ORG_DIR = config.CHATINO_DIR
# The directory for storing input features and targets.
TGT_DIR = os.path.join(config.TGT_DIR, "chatino")
ORG_WAV_DIR = os.path.join(ORG_DIR, "wav")
# TODO Consider factoring out as non-Chatino specific
FEAT_DIR = os.path.join(TGT_DIR, "feat")
ORG_TRANSCRIPT_DIR = os.path.join(ORG_DIR, "transcriptions")
# TODO Consider factoring out as non-Chatino specific
LABEL_DIR = os.path.join(TGT_DIR, "label")

# Obtain the filename prefixes that identify recordings and their
# transcriptions
# TODO Consider factoring out as non-Chatino specific

PREFIXES = [os.path.splitext(fn)[0]
            for fn in os.listdir(ORG_TRANSCRIPT_DIR)
            if fn.endswith(".txt")]

# Hardcode the phoneme set
ONE_CHAR_PHONEMES = set(["p", "t", "d", "k", "q", "s", "x", "j", "m", "n", "r", "l",
                         "y", "w", "i", "u", "e", "o", "a"])
TWO_CHAR_PHONEMES = set(["ty", "dy", "kw", "ts", "dz", "ch", "ny", "ly",
                         "in", "en", "On", "an"])
PHONEMES = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)

# TODO Consider factoring out as non-Chatino specific
PHONEMES_TO_INDICES = {phn: index for index, phn in enumerate(PHONEMES)}
INDICES_TO_PHONEMES = {index: phn for index, phn in enumerate(PHONEMES)}

# Hardcode the tone set
ONE_CHAR_TONES = set(["0", "3", "1", "4", "2"])
TWO_CHAR_TONES = set(["04", "14", "24", "20", "42", "32", "40", "10"])
THREE_CHAR_TONES = set(["140"])
TONES = ONE_CHAR_TONES.union(TWO_CHAR_TONES.union(THREE_CHAR_TONES))

def process_transcript(org_transcript_fn, label_fn, label_type):
    """ Splits the Chatino transcript into phonemes. """

    def remove_punctuation(line):
        """ Replace symbols we don't care about.""" 

        line = line.lower()
        line = line.replace(",", "")
        line = line.replace("-", "")
        line = line.replace(".", "")
        line = line.replace("!", "")
        line = line.replace("˜", "")
        line = line.replace("+", "")
        line = line.replace(":", "")
        line = line.replace("?", "")
        line = line.replace("[blurry]", "")
        line = line.replace("[plata]", "")

        return line

    def bad_transcript(line):
        """ True if we should disregard the utterance; false otherwise."""

        if "g" in line:
            return True
        if "ñ" in line:
            return True
        if "<" in line:
            return True
        if ">" in line:
            return True
        if "b" in line:
            return True
        if "(" in line:
            return True
        if "[" in line:
            return True
        if "õ" in line:
            return True
        if "chaqF" in line:
            return True
        if "Bautista" in line:
            return True
        if "ntsaqG" in line:
            return True
        return False

    def extract_labels(line):

        # Do we want phonemes, tones, or both?
        if label_type == "phonemes":
            phonemes = True
            tones = False
        elif label_type == "tones":
            phonemes = False
            tones = True
        elif label_type == "phonemes_and_tones":
            phonemes = True
            tones = True

        words = line.split()
        labels = []
        for word in words:
            i = 0
            while i < len(word):
                if word[i:i+3] in THREE_CHAR_TONES:
                    if tones:
                        labels.append(word[i:i+3])
                    i += 3
                    continue
                if word[i:i+2] in TWO_CHAR_PHONEMES:
                    if phonemes:
                        labels.append(word[i:i+2])
                    i += 2
                    continue
                if word[i:i+2] in TWO_CHAR_TONES:
                    if tones:
                        labels.append(word[i:i+2])
                    i += 2
                    continue
                elif word[i:i+1] in ONE_CHAR_PHONEMES:
                    if phonemes:
                        labels.append(word[i:i+1])
                    i += 1
                    continue
                elif word[i:i+1] in ONE_CHAR_TONES:
                    if tones:
                        labels.append(word[i:i+1])
                    i += 1
                    continue
                elif word[i:i+1] == "'":
                    # Then assume it's distinguishing n'y from ny.
                    i += 1
                    continue
                elif word[i:i+1] == "c":
                    #It wasn't ch, return nothing and remove utterance.
                    return []
                elif word[i:i+1] == "h":
                    #It wasn't ch, return nothing and remove utterance.
                    return []
                else:
                    print(org_transcript_fn)
                    print(words)
                    print(word)
                    print(word[i:], end="")
                    input()
                    break
        return labels

    with open(org_transcript_fn) as org_f:
        line = org_f.readline()
    if bad_transcript(line):
        return
    line = remove_punctuation(line)
    phonemes = extract_labels(line)
    if phonemes == []: # If the transcription is empty, don't use the utterance
        return
    with open(label_fn, "w") as tgt_f:
        print(" ".join(phonemes), file=tgt_f)

def prepare_labels(label_type):
    """ Prepare the neural network output targets."""

    if not os.path.isdir(LABEL_DIR):
        os.makedirs(LABEL_DIR)

    for prefix in PREFIXES:
        org_fn = os.path.join(ORG_TRANSCRIPT_DIR, "%s.txt" % prefix)
        label_fn = os.path.join(
            LABEL_DIR, "%s.%s" % (prefix, label_type))
        process_transcript(org_fn, label_fn, label_type)

# TODO Consider factoring out as non-Chatino specific
def prepare_feats(feat_type):
    """ Prepare the input features."""

    if not os.path.isdir(FEAT_DIR):
        os.makedirs(FEAT_DIR)

    if feat_type=="phonemes_onehot":
        import numpy as np
        prepare_labels("phonemes")
        for prefix in PREFIXES:
            label_fn = os.path.join(LABEL_DIR, "%s.phonemes" % prefix)
            try:
                with open(label_fn) as label_f:
                    labels = label_f.readlines()[0].split()
            except FileNotFoundError:
                continue
            indices = [PHONEMES_TO_INDICES[label] for label in labels]
            one_hots = one_hots = [[0]*len(PHONEMES) for _ in labels]
            for i, index in enumerate(indices):
                one_hots[i][index] = 1
                one_hots = np.array(one_hots)
                np.save(os.path.join(FEAT_DIR, "%s.phonemes_onehot" %  prefix),
                        one_hots)
    else:
        # Otherwise, 
        for prefix in PREFIXES:
            # Convert the wave to 16k mono.
            org_wav_fn = os.path.join(ORG_WAV_DIR, "%s.wav" % prefix)
            mono16k_wav_fn = os.path.join(FEAT_DIR, "%s.wav" % prefix)
            if not os.path.isfile(mono16k_wav_fn):
                feat_extract.convert_wav(org_wav_fn, mono16k_wav_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(FEAT_DIR, feat_type=feat_type)

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Chatino corpus."""

    # TODO Reconsider the place of these splits. Perhaps train/dev/test
    # directories should be used instead, and generated in the prepare() step.
    TRAIN_VALID_TEST_SPLIT = [2048, 207, 206]
    FEAT_DIR = FEAT_DIR
    LABEL_DIR = LABEL_DIR

    def __init__(self, feat_type, label_type, max_samples=1000):
        super().__init__(feat_type, label_type)

        if label_type == "phonemes":
            self.labels = PHONEMES
        elif label_type == "tones":
            self.labels = TONES
        elif label_type == "phonemes_and_tones":
            self.labels = PHONEMES.union(TONES)
        else:
            raise Exception("label_type=%s not supported." % (label_type))

        self.feat_type = feat_type
        self.label_type = label_type

        # Filter prefixes based on what we find in the feat/ dir.
        self.prefixes = [prefix for prefix in PREFIXES
                         if os.path.isfile(os.path.join(
                             FEAT_DIR, "%s.%s.npy" % (prefix, feat_type)))]
        # Filter prefixes based on what we find in the label/ dir.
        self.prefixes = [prefix for prefix in self.prefixes
                         if os.path.isfile(os.path.join(
                             LABEL_DIR, "%s.%s" % (prefix, label_type)))]

        # Remove prefixes whose feature files are too long.
        if max_samples:
            self.prefixes = utils.sort_and_filter_by_size(
                FEAT_DIR, self.prefixes, feat_type, max_samples)

        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_TEST_SPLIT[0]]
        valid_end = self.TRAIN_VALID_TEST_SPLIT[0]+self.TRAIN_VALID_TEST_SPLIT[1]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_TEST_SPLIT[0]:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

        # Writing validation set prefixes so that latex_output can give
        # meaningful output.
        with open(join(TGT_DIR, "valid_prefixes.txt"), "w") as valid_f:
            for prefix in self.valid_prefixes:
                print(prefix, file=valid_f)

        # TODO Make the distinction between this and the constants at the start
        # of the file clear.
        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: label for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.vocab_size = len(self.labels)

    # TODO Use 'labels' instead of 'phonemes' here and in corpus.py
    # Also, factor out as non-Chatino-specific.
    def indices_to_phonemes(self, indices):
        return [(self.INDEX_TO_LABEL[index]) for index in indices]
    def phonemes_to_indices(self, labels):
        return [self.LABEL_TO_INDEX[label] for label in labels]
