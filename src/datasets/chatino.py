""" Interface to the Griko data."""

import os
import random
from shutil import copyfile
import subprocess

import config
import corpus
import feat_extract

ORG_DIR = config.CHATINO_DIR
TGT_DIR = "../data/chatino"

ONE_CHAR_PHONEMES = set(["p", "t", "d", "k", "q", "s", "x", "j", "m", "n", "r", "l",
                         "y", "w", "i", "u", "e", "o", "a"])
ONE_CHAR_TONES = set(["0", "3", "1", "4", "2"])
TWO_CHAR_TONES = set(["04", "14", "24", "20", "42", "32", "40", "10"])
TWO_CHAR_PHONEMES = set(["ty", "dy", "kw", "ts", "dz", "ch", "ny", "ly",
                         "in", "en", "On", "an"])
THREE_CHAR_TONES = set(["140"])

def convert_wav(org_wav_fn, tgt_wav_fn):
    """ Converts the wav into a 16bit mono 16000Hz wav."""
    home = os.path.expanduser("~")
    args = [os.path.join(home, "tools", "ffmpeg-3.3", "ffmpeg"),
            "-i", org_wav_fn, "-ac", "1", "-ar", "16000", tgt_wav_fn]
    subprocess.run(args)

def convert_transcript(org_transcript_fn, tgt_transcript_fn):
    """ Splits the Chatino transcript into phonemes. """

    def extract_phonemes(line):
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
        if "g" in line:
            return
        if "ñ" in line:
            return
        if "<" in line:
            return
        if ">" in line:
            return
        if "b" in line:
            return
        if "(" in line:
            return
        if "[" in line:
            return
        if "õ" in line:
            return
        words = line.split()
        phonemes = []
        for word in words:
            i = 0
            while i < len(word):
                if word[i:i+3] in THREE_CHAR_TONES:
                    phonemes.append(word[i:i+3])
                    i += 3
                    continue
                if word[i:i+2] in TWO_CHAR_PHONEMES:
                    phonemes.append(word[i:i+2])
                    i += 2
                    continue
                if word[i:i+2] in TWO_CHAR_TONES:
                    phonemes.append(word[i:i+2])
                    i += 2
                    continue
                elif word[i:i+1] in ONE_CHAR_PHONEMES:
                    phonemes.append(word[i:i+1])
                    i += 1
                    continue
                elif word[i:i+1] in ONE_CHAR_TONES:
                    phonemes.append(word[i:i+1])
                    i += 1
                    continue
                elif word[i:i+1] == "'":
                    # Then assume it's distinguishing n'y from ny.
                    i += 1
                    continue
                elif word[i:i+1] == "c": #It wasn't ch
                    return
                elif word[i:i+1] == "h": #It wasn't ch
                    return
                else:
                    print(word)
                    print(word[i:], end="")
                    input()
        return phonemes

    with open(org_transcript_fn) as org_f:
        line = org_f.readline()
        phonemes = extract_phonemes(line)
        if phonemes == None:
            return
        if phonemes == []:
            # If the transcription is empty
            return
        with open(tgt_transcript_fn, "w") as tgt_f:
            print(" ".join(phonemes), file=tgt_f)

def get_target_prefix(prefix):
    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
    corresponding target file name."""

    filename = os.path.basename(prefix)
    return os.path.join(TGT_DIR, "transcriptions", filename)

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Griko corpus."""

    TRAIN_VALID_TEST_SPLIT = [2048, 207, 206]
    _phonemes = None
    _chars = None

    def __init__(self, feat_type, target_type, tones=False, max_samples=1000):
        super().__init__(feat_type, target_type)

        org_transcriptions_dir = os.path.join(ORG_DIR, "transcriptions")

        if tones:
            self.phonemes = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)
            self.phonemes = self.phonemes.union(ONE_CHAR_TONES)
            self.phonemes = self.phonemes.union(TWO_CHAR_TONES)
            self.phonemes = self.phonemes.union(THREE_CHAR_TONES)
        else:
            self.phonemes = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)
            raise Exception("Not implemented yet.")

        tgt_wav_dir = os.path.join(TGT_DIR, "wav")
        if not os.path.isdir(tgt_wav_dir):
            self.prepare()

        self.prefixes = [os.path.join(tgt_wav_dir, fn.replace(".wav", ""))
                         for fn in os.listdir(tgt_wav_dir) if fn.endswith(".wav")]

        if max_samples:
            self.prefixes = self.sort_and_filter_by_size(self.prefixes, max_samples)

        print(len(self.prefixes))

        random.seed(0)
        random.shuffle(self.prefixes)
        self.train_prefixes = self.prefixes[:self.TRAIN_VALID_TEST_SPLIT[0]]
        valid_end = self.TRAIN_VALID_TEST_SPLIT[0]+self.TRAIN_VALID_TEST_SPLIT[1]
        self.valid_prefixes = self.prefixes[self.TRAIN_VALID_TEST_SPLIT[0]:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

        self.PHONEMES2INDICES = {phn: index for index, phn in enumerate(sorted(list(self.phonemes)))}
        self.INDICES2PHONEMES = {index: phn for index, phn in enumerate(sorted(list(self.phonemes)))}
        self.vocab_size = len(self.phonemes)

    def prepare(self):
        """ Preprocess the Griko data."""

        org_wav_dir = os.path.join(ORG_DIR, "wav")
        tgt_wav_dir = os.path.join(TGT_DIR, "wav")
        if not os.path.isdir(tgt_wav_dir):
            os.makedirs(tgt_wav_dir)

        org_transcript_dir = os.path.join(ORG_DIR, "transcriptions")
        tgt_transcript_dir = os.path.join(TGT_DIR, "transcriptions")
        if not os.path.isdir(tgt_transcript_dir):
            os.makedirs(tgt_transcript_dir)

        org_prefixes = [os.path.join(tgt_transcript_dir, fn.strip(".txt"))
                        for fn in os.listdir(org_transcript_dir) if fn.endswith(".txt")]
        for prefix in org_prefixes:
            prefix = os.path.basename(prefix)
            print(prefix)
            # Split phonemes in the transcript.
            org_transcript_fn = os.path.join(org_transcript_dir, "%s.txt" % prefix)
            tgt_transcript_fn = os.path.join(tgt_transcript_dir, "%s.phn" % prefix)
            convert_transcript(org_transcript_fn, tgt_transcript_fn)

        prefixes = [os.path.join(tgt_transcript_dir, fn.strip(".phn"))
                        for fn in os.listdir(tgt_transcript_dir)]
        # For each of the prefixes we kept based on the transcriptions...
        for prefix in prefixes:
            prefix = os.path.basename(prefix)
            # Copy the wav to the local dir.
            org_wav_fn = os.path.join(org_wav_dir, "%s.wav" % prefix)
            tgt_wav_fn = os.path.join(tgt_wav_dir, "%s.wav" % prefix)
            convert_wav(org_wav_fn, tgt_wav_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(tgt_wav_dir, feat_type="log_mel_filterbank")

    def indices_to_phonemes(self, indices):
        return [(self.INDICES2PHONEMES[index-1] if index > 0 else "pad") for index in indices]

    def phonemes_to_indices(self, phonemes):
        return [self.PHONEMES2INDICES[phoneme]+1 for phoneme in phonemes]

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
