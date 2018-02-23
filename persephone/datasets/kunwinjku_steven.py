""" Interface to Steven's Kunwinjku data. """

import glob
import os
from os.path import join
from pathlib import Path
import string
import sys
from typing import List, NamedTuple, Set

import nltk # type: ignore
nltk.download("punkt") # type: ignore
from pympi.Elan import Eaf

from .. import corpus
from .. import config
from ..transcription_preprocessing import segment_into_tokens
from .. import utils
from ..utterance import Utterance
from ..preprocess import elan

BASIC_PHONEMES = set(["a", "b", "d", "dj", "rd", "e", "h", "i", "k", "l",
            "rl", "m", "n", "ng", "nj", "rn", "o", "r", "rr", "u",
            "w", "y",])
DOUBLE_STOPS = set(["bb", "dd", "djdj", "rdd", "kk"])
DIPHTHONGS = set(["ay", "aw", "ey", "ew", "iw", "oy", "ow", "uy"])
PHONEMES = BASIC_PHONEMES | DOUBLE_STOPS | DIPHTHONGS

def get_en_words() -> Set[str]:
    """
    Returns a list of English words which can be used to filter out
    code-switched sentences.
    """

    with open(config.EN_WORDS_PATH) as words_f:
        raw_words = words_f.readlines()
    en_words = set([word.strip().lower() for word in raw_words])
    NA_WORDS_IN_EN_DICT = set(["kore", "nani", "karri", "imi", "o", "yaw", "i",
                           "bi", "aye", "imi", "ane", "kubba", "kab", "a-",
                           "ad", "a", "mak", "selim", "ngai", "en", "yo",
                           "wud", "mani", "yak", "manu", "ka-", "mong",
                           "manga", "ka-", "mane", "kala", "name", "kayo",
                           "kare", "laik", "bale", "ni", "rey", "bu",
                           "re", "iman", "bom", "wam",
                           "alu", "nan", "kure", "kuri", "wam", "ka", "ng",
                           "yi", "na", "m", "arri", "e", "kele", "arri", "nga",
                           "kakan", "ai", "ning", "mala", "ti", "wolk",
                           "bo", "andi", "ken", "ba", "aa", "kun", "bini",
                           "wo", "bim", "man", "bord", "al", "mah", "won",
                           "ku", "ay", "belen", "wen", "yah", "muni",
                           "bah", "di", "mm", "anu", "nane", "ma", "kum",
                           "birri", "ray", "h", "kane", "mumu", "bi", "ah",
                           "i-", "n", "mi", "bedman",
                           ])
    EN_WORDS_NOT_IN_EN_DICT = set(["screenprinting"])
    en_words = en_words.difference(NA_WORDS_IN_EN_DICT)
    en_words = en_words | EN_WORDS_NOT_IN_EN_DICT
    return en_words

EN_WORDS = get_en_words()

def explore_elan_files(elan_paths):
    """
    A function to explore the tiers of ELAN files.
    """

    for elan_path in elan_paths:
        print(elan_path)
        eafob = Eaf(elan_path)
        tier_names = eafob.get_tier_names()
        for tier in tier_names:
            print("\t", tier)
            try:
                for annotation in eafob.get_annotation_data_for_tier(tier):
                    print("\t\t", annotation)
            except KeyError:
                continue

        input()

def filter_codeswitched(utterances):
    for utter in utterances:
        toks = nltk.word_tokenize(utter.text)
        words = [tok.lower() for tok in toks]
        codeswitched = False
        for word in words:
            if word in EN_WORDS:
                codeswitched = True
                break
        if not codeswitched:
            yield utter

def segment_phonemes(text: str, phoneme_inventory: Set[str] = PHONEMES) -> str:
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    text = text.lower()
    text = segment_into_tokens(text, phoneme_inventory)
    return text

def explore_code_switching(f=sys.stdout):

    utters = elan_utterances()

    en_count = 0
    for i, utter in enumerate(utters):
        toks = nltk.word_tokenize(utter.text)
        words = [tok.lower() for tok in toks]
        for word in words:
            if word in EN_WORDS:
                en_count += 1
                print("Utterance #%s" % i, file=f)
                print("Original: %s" % utter.text, file=f)
                print("Tokenized: %s" % words, file=f)
                print("Phonemic: %s" % segment_phonemes(utter.text), file=f)
                print("En word: %s" % word, file=f)
                print("---------------------------------------------", file=f)
                break
    print(en_count)
    print(len(utters))

class Corpus(preprocess.elan.Corpus):
    def __init__(self, org_dir: Path = Path(config.KUNWINJKU_STEVEN_DIR),
                 tgt_dir: Path = Path(config.TGT_DIR, "BKW"),
                 feat_type: str = "fbank", label_type: str = "phonemes") -> None:

        wav_dir = tgt_dir / "wav"
        label_dir = tgt_dir / "label"
        wav_dir.mkdir(parents=True, exist_ok=True) # pylint: disable=no-member
        label_dir.mkdir(parents=True, exist_ok=True) # pylint: disable=no-member

        if label_type == "phonemes":
            labels = PHONEMES
        else:
            raise NotImplementedError(
                "label_type {} not implemented.".format(label_type))

        # 0. Fetch the utterances from the ELAN files that aren't codeswitched.
        utterances = list(filter_codeswitched(elan_utterances()))

        # 1. Preprocess transcriptions and put them in the label/ directory
        for utterance in utterances:
            phoneme_str = segment_phonemes(utterance.text)
            label_fn = "{}.{}".format(utterance.prefix, label_type)
            label_path = Path(label_dir, label_fn)
            if not label_path.is_file():
                with label_path.open("w") as f:
                    print(phoneme_str, file=f)

        # 2. Split the WAV files and put them in the wav/
        for utterance in utterances:
            start_time = utterance.start_time
            end_time = utterance.end_time
            wav_fn = "{}.{}".format(utterance.prefix, "wav")
            out_wav_path = Path(wav_dir, wav_fn)
            if not out_wav_path.is_file():
                in_wav_path = utterance.wav_file
                utils.trim_wav_ms(in_wav_path, str(out_wav_path), start_time, end_time)

        # super() will then do feature extraction and create train/valid/test
        super().__init__(tgt_dir, feat_type, label_type)
