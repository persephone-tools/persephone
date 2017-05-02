import os
import random
from shutil import copyfile

import config
import corpus
import feat_extract

ORG_DIR = config.GRIKO_DIR
TGT_DIR = "../data/griko"

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Griko corpus."""

    TRAIN_VALID_TEST_SPLIT = [256, 30, 44]
    _chars = None

    def __init__(self, feat_type, target_type):
        super().__init__(feat_type, target_type)

        input_dir = os.path.join(ORG_DIR, "raw")
        self.prefixes = [os.path.join(input_dir, fn.strip(".wav"))
                    for fn in os.listdir(input_dir) if fn.endswith(".wav")]
        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_TEST_SPLIT[0]]
        valid_end = self.TRAIN_VALID_TEST_SPLIT[0]+self.TRAIN_VALID_TEST_SPLIT[1]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_TEST_SPLIT[0]:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

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

        transcript_dir = os.path.join(TGT_DIR, "transcriptions")
        if not os.path.isdir(transcript_dir):
            os.makedirs(transcript_dir)
        wav_dir = os.path.join(TGT_DIR, "wav")
        if not os.path.isdir(wav_dir):
            os.makedirs(wav_dir)

        for prefix in self.prefixes:
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

            # Extract features from the wav.
            feat_extract.from_dir(wav_dir, feat_type="log_mel_filterbank")

    def indices_to_phonemes(self):
        pass

    def phonemes_to_indices(self):
        pass

    def get_train_fns(self):
        pass
    def get_valid_fns(self):
        pass
    def get_test_fns(self):
        pass
