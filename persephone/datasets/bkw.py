""" Interface to Alex/Steven's Kunwinjku data. """

from pathlib import Path
import subprocess
from typing import List, Set

import nltk # type: ignore
# TODO This download should be conditional, since a complaint is raised if
# there is no net connection
nltk.download("punkt") # type: ignore
from pympi.Elan import Eaf

from .. import corpus
from .. import config
from ..preprocess.labels import segment_into_tokens
from ..utterance import Utterance
from ..preprocess import elan
from ..preprocess.labels import LabelSegmenter
from ..corpus import Corpus

BASIC_PHONEMES = set(["a", "b", "d", "dj", "rd", "e", "h", "i", "k", "l",
            "rl", "m", "n", "ng", "nj", "rn", "o", "r", "rr", "u",
            "w", "y",])
DOUBLE_STOPS = set(["bb", "dd", "djdj", "rdd", "kk"])
DIPHTHONGS = set(["ay", "aw", "ey", "ew", "iw", "oy", "ow", "uy"])
PHONEMES = BASIC_PHONEMES | DOUBLE_STOPS | DIPHTHONGS

def pull_en_words() -> None:
    """ Fetches a repository containing English words. """

    ENGLISH_WORDS_URL = "https://github.com/dwyl/english-words.git"
    en_words_path = Path(config.EN_WORDS_PATH)
    if not en_words_path.is_file():
        subprocess.run(["git", "clone",
                        ENGLISH_WORDS_URL, str(en_words_path.parent)])

def get_en_words() -> Set[str]:
    """
    Returns a list of English words which can be used to filter out
    code-switched sentences.
    """

    pull_en_words()

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
                           "i-", "n", "mi", "bedman", "rud", "le", "babu",
                           "da", "kakkak", "yun", "ande", "naw", "kam", "bolk",
                           "woy", "u", "bi-",
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

def segment_utterance(utterance: Utterance) -> Utterance:
    fields = utterance._asdict()
    fields["text"] = segment_str(fields["text"])
    return Utterance(**fields)

def segment_str(text: str, phoneme_inventory: Set[str] = PHONEMES) -> str:
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    text = text.lower()
    text = segment_into_tokens(text, phoneme_inventory)
    return text

bkw_label_segmenter = LabelSegmenter(segment_utterance, PHONEMES)

def explore_code_switching(utterances: List[Utterance], out_path: Path) -> None:

    with out_path.open("w") as f:
        en_count = 0
        for i, utter in enumerate(utterances):
            toks = nltk.word_tokenize(utter.text) # type: ignore
            words = [tok.lower() for tok in toks]
            for word in words:
                if word in EN_WORDS:
                    en_count += 1
                    print("Prefix: {}".format(utter.prefix), file=f)
                    print("Utterance #%s" % i, file=f)
                    print("Original: %s" % utter.text, file=f)
                    print("Tokenized: %s" % words, file=f)
                    print("Phonemic: %s" % segment_str(utter.text), file=f)
                    print("En word: %s" % word, file=f)
                    print("---------------------------------------------", file=f)
                    break
        print("Num of codeswitched/English sentences: {}".format(en_count), file=f)
        print("Total number of utterances: {}".format(len(utterances)), file=f)

def filter_for_not_codeswitched(utter: Utterance) -> bool:
    toks = nltk.word_tokenize(utter.text) # type: ignore
    words = [tok.lower() for tok in toks]
    for word in words:
        if word in EN_WORDS:
            return False
    return True

def filter_for_not_empty(utter: Utterance) -> bool:
    return not utter.text.strip() == ""

def bkw_filter(utter: Utterance) -> bool:
    return filter_for_not_codeswitched(utter) and filter_for_not_empty(utter)

def create_corpus(org_dir: Path = Path(config.BKW_PATH),
                 tgt_dir: Path = Path(config.TGT_DIR, "BKW"),
                 feat_type: str = "fbank", label_type: str = "phonemes",
                 speakers: List[str] = None) -> Corpus:

    if label_type != "phonemes":
        raise NotImplementedError(
            "label_type {} not implemented.".format(label_type))

    return corpus.Corpus.from_elan(org_dir, tgt_dir,
                     feat_type=feat_type, label_type=label_type,
                     utterance_filter=bkw_filter,
                     label_segmenter=bkw_label_segmenter,
                     speakers=speakers, tier_prefixes=["xv", "rf"])
