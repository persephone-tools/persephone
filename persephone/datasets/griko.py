""" Interface to the Griko data."""

import os
import random
from shutil import copyfile

from .. import config
from .. import corpus
from .. import feat_extract

ORG_DIR = config.GRIKO_DIR
TGT_DIR = "../data/griko"

def get_target_prefix(prefix):
    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
    corresponding target file name."""

    filename = os.path.basename(prefix)
    return os.path.join(TGT_DIR, "transcriptions", filename)

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Griko corpus."""

    TRAIN_VALID_TEST_SPLIT = [256, 30, 44]
    _chars = None

    def __init__(self, feat_type, target_type):
        super().__init__(feat_type, target_type)

        tgt_wav_dir = os.path.join(TGT_DIR, "wav")

        if not os.path.isdir(tgt_wav_dir):
            self.prepare()

        self.prefixes = [os.path.join(tgt_wav_dir, fn.strip(".wav"))
                         for fn in os.listdir(tgt_wav_dir) if fn.endswith(".wav")]
        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_TEST_SPLIT[0]]
        valid_end = self.TRAIN_VALID_TEST_SPLIT[0]+self.TRAIN_VALID_TEST_SPLIT[1]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_TEST_SPLIT[0]:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

        self.CHARS2INDICES = {char: index for index, char in enumerate(sorted(list(self.chars)))}
        self.INDICES2CHARS = {index: char for index, char in enumerate(sorted(list(self.chars)))}
        self.vocab_size = len(self.chars)

    @property
    def chars(self):
        """ Determine the characters present in the corpus. """

        if not self._chars:
            self._chars = set()
            for prefix in self.prefixes:
                prefix = os.path.basename(prefix)
                transcript_fn = os.path.join(ORG_DIR, "transcriptions", "%s.gr" % prefix)
                with open(transcript_fn) as transcript_f:
                    line = transcript_f.readline()
                    for char in line.lower():
                        self._chars.add(char)
            self._chars = self._chars - {" ", "\n"}

        return self._chars

    def prepare(self):
        """ Preprocess the Griko data."""

        input_dir = os.path.join(ORG_DIR, "raw")
        wav_dir = os.path.join(TGT_DIR, "wav")
        os.makedirs(wav_dir)

        transcript_dir = os.path.join(TGT_DIR, "transcriptions")
        if not os.path.isdir(transcript_dir):
            os.makedirs(transcript_dir)

        org_prefixes = [os.path.join(input_dir, fn.strip(".wav"))
                        for fn in os.listdir(input_dir) if fn.endswith(".wav")]
        for prefix in org_prefixes:
            prefix = os.path.basename(prefix)
            # Split characters in the transcript.
            org_transcript_fn = os.path.join(ORG_DIR, "transcriptions", "%s.gr" % prefix)
            tgt_transcript_fn = os.path.join(transcript_dir, "%s.char" % prefix)
            with open(org_transcript_fn) as org_f, open(tgt_transcript_fn, "w") as tgt_f:
                line = org_f.readline()
                chars = []
                for char in line.lower():
                    if char not in {" ", "\n"}:
                        chars.append(char)
                print(" ".join(chars), file=tgt_f)

            # Copy the wav to the local dir.
            org_wav_fn = os.path.join(ORG_DIR, "raw", "%s.wav" % prefix)
            tgt_wav_fn = os.path.join(TGT_DIR, "wav", "%s.wav" % prefix)
            copyfile(org_wav_fn, tgt_wav_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(wav_dir, feat_type="log_mel_filterbank")

    # TODO change to indices_to_chars
    def indices_to_phonemes(self, indices):
        return [(self.INDICES2CHARS[index-1] if index > 0 else "pad") for index in indices]

    # TODO change to chars_to_indices
    def phonemes_to_indices(self, chars):
        return [self.CHARS2INDICES[char]+1 for char in chars]

    def get_train_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.train_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                      for prefix in self.train_prefixes]
        return feat_fns, target_fns
    def get_valid_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.valid_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns
    def get_test_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.test_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                      for prefix in self.test_prefixes]
        return feat_fns, target_fns
