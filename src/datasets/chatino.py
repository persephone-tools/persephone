""" Interface to the Chatino data."""

import os
import random
from shutil import copyfile

import config
import corpus
import feat_extract

### Initialize path variables.###
# Path to unpreprocessed Chatino data from GORILLA.
ORG_DIR = config.CHATINO_DIR
# The directory for storing input features and targets.
TGT_DIR = os.path.join(config.TGT_DIR, "chatino")
ORG_WAV_DIR = os.path.join(ORG_DIR, "wav")
FEAT_DIR = os.path.join(TGT_DIR, "feat")
ORG_TRANSCRIPT_DIR = os.path.join(ORG_DIR, "transcriptions")
LABEL_DIR = os.path.join(TGT_DIR, "label")

ONE_CHAR_PHONEMES = set(["p", "t", "d", "k", "q", "s", "x", "j", "m", "n", "r", "l",
                         "y", "w", "i", "u", "e", "o", "a"])
ONE_CHAR_TONES = set(["0", "3", "1", "4", "2"])
TWO_CHAR_TONES = set(["04", "14", "24", "20", "42", "32", "40", "10"])
TWO_CHAR_PHONEMES = set(["ty", "dy", "kw", "ts", "dz", "ch", "ny", "ly",
                         "in", "en", "On", "an"])
THREE_CHAR_TONES = set(["140"])

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

def get_target_prefix(prefix):
    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
    corresponding target file name."""

    filename = os.path.basename(prefix)
    return os.path.join(TGT_DIR, "transcriptions", filename)

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Chatino corpus."""

    # TODO Reconsider the place of these splits. Perhaps train/dev/test
    # directories should be used instead, and generated in the prepare() step.
    TRAIN_VALID_TEST_SPLIT = [2048, 207, 206]
    _phonemes = None
    _chars = None
    tones = False

    def __init__(self, feat_type, target_type, max_samples=1000):
        super().__init__(feat_type, target_type)

        org_transcriptions_dir = os.path.join(ORG_DIR, "transcriptions")

        if tones:
            self.phonemes = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)
            self.phonemes = self.phonemes.union(ONE_CHAR_TONES)
            self.phonemes = self.phonemes.union(TWO_CHAR_TONES)
            self.phonemes = self.phonemes.union(THREE_CHAR_TONES)
            self.tones = True
        else:
            self.phonemes = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)

        self.prefixes = [os.path.join(FEAT_DIR, fn.replace(".wav", ""))
                         for fn in os.listdir(FEAT_DIR) if fn.endswith(".wav")]

        if max_samples:
            self.prefixes = self.sort_and_filter_by_size(self.prefixes, max_samples)

        print(len(self.prefixes))

        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_TEST_SPLIT[0]]
        valid_end = self.TRAIN_VALID_TEST_SPLIT[0]+self.TRAIN_VALID_TEST_SPLIT[1]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_TEST_SPLIT[0]:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

        self.PHONEME_TO_INDEX = {phn: index for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.INDEX_TO_PHONEME = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.vocab_size = len(self.phonemes)

    @staticmethod
    def prepare(feat_type, label_type):
        """ Preprocess the Chatino data."""

        def prepare_feats(feat_type):
            """ Prepare the input features."""

            if not os.path.isdir(FEAT_DIR):
                os.makedirs(FEAT_DIR)

        def prepare_labels(label_type):
            """ Prepare the neural network output targets."""

            if not os.path.isdir(LABEL_DIR):
                os.makedirs(LABEL_DIR)

            for prefix in prefixes:
                org_fn = os.path.join(ORG_TRANSCRIPT_DIR, "%s.txt" % prefix)
                label_fn = os.path.join(
                    LABEL_DIR, "%s.%s" % (prefix, label_type))
                process_transcript(org_fn, label_fn, label_type)

        # Obtain the filename prefixes that identify recordings and their
        # transcriptions
        prefixes = [os.path.splitext(fn)[0]
                    for fn in os.listdir(ORG_TRANSCRIPT_DIR)
                    if fn.endswith(".txt")]

        prepare_feats(feat_type)
        prepare_labels(label_type)


        """
        for prefix in org_prefixes:
            prefix = os.path.basename(prefix)
            print(prefix)
            # Split phonemes in the transcript.
            org_transcript_fn = os.path.join(ORG_TRANSCRIPT_DIR, "%s.txt" % prefix)
            label_fn = os.path.join(LABEL_DIR, "%s.tones%s.phn" % (prefix, str(tones)))
            process_transcript(org_transcript_fn, label_fn, tones)

        prefixes = [os.path.join(LABEL_DIR, fn.strip(".phn"))
                        for fn in os.listdir(LABEL_DIR)]
        # For each of the prefixes we kept based on the transcriptions...
        for prefix in prefixes:
            prefix = os.path.basename(prefix)
            # Copy the wav to the local dir.
            org_wav_fn = os.path.join(org_wav_dir, "%s.wav" % prefix)
            feat_fn = os.path.join(FEAT_DIR, "%s.wav" % prefix)
            feat_extract.convert_wav(org_wav_fn, feat_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(FEAT_DIR, feat_type="log_mel_filterbank")
        """

    def indices_to_phonemes(self, indices):
        return [(self.INDEX_TO_PHONEME[index]) for index in indices]

    def phonemes_to_indices(self, phonemes):
        return [self.PHONEME_TO_INDEX[phoneme] for phoneme in phonemes]

    def get_train_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.train_prefixes]
        target_fns = ["%s.tones%s.%s" % (get_target_prefix(prefix), str(self.tones), self.target_type)
                      for prefix in self.train_prefixes]
        return feat_fns, target_fns
    def get_valid_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.valid_prefixes]
        target_fns = ["%s.tones%s.%s" % (get_target_prefix(prefix), str(self.tones), self.target_type)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns
    def get_test_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.test_prefixes]
        target_fns = ["%s.tones%s.%s" % (get_target_prefix(prefix), str(self.tones), self.target_type)
                      for prefix in self.test_prefixes]
        return feat_fns, target_fns
