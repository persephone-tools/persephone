""" An interface with the Na data. """

import os

import config

ORG_DIR = config.NA_DIR
TGT_DIR = "../data/na"
ORG_TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")
TGT_TXT_NORM_DIR = os.path.join(TGT_DIR, "txt_norm")

TO_REMOVE = {"|", "ǀ", "↑", "«", "»", "¨", "“", "”", "D", "F"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩"}
WORDS_TO_REMOVE = {"CHEVRON", "audible", "qʰʰʰʰʰ", "qʰʰʰʰ", "D"}
TONES = ["˧˥", "˩˥", "˩˧", "˧˩", "˩", "˥", "˧"]

if not os.path.isdir(TGT_DIR):
    os.makedirs(TGT_DIR)

def load_phonemes(na_dir=TGT_DIR):
    """ Loads the phoneme set."""

    uni_phns = set()
    with open(os.path.join(na_dir, "uni_char_phonemes.txt")) as uni_f:
        for line in uni_f:
            uni_phns.add(line.strip())

    bi_phns = set()
    with open(os.path.join(na_dir, "bi_char_phonemes.txt")) as bi_f:
        for line in bi_f:
            bi_phns.add(line.strip())

    return uni_phns, bi_phns, TRI_PHNS

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def remove_multi(a, ys):
    """ Removes instances of a from the list ys."""
    return list(filter(lambda x: x != a, ys))

def contains_forbidden_word(line):
    for word in WORDS_TO_REMOVE:
        if word in line:
            return True
    return False

def trim_wavs(segmentation, remove_tones=True):
    """ Trims available wavs into the sentence or utterance-level."""

    if not os.path.exists(TGT_TXT_NORM_DIR):
        os.makedirs(TGT_TXT_NORM_DIR)

    syl_inv = set()
    for fn in os.listdir(ORG_TXT_NORM_DIR):
        with open(os.path.join(ORG_TXT_NORM_DIR, fn)) as f:
            i = 0
            for line in f:

                # Remove lines with certain words in it.
                if contains_forbidden_word(line):
                    continue

                # Remove certain symbols from lines.
                for symbol in TO_REMOVE:
                    line = line.replace(symbol, "")
                if remove_tones:
                    for tone in TONES:
                        line = line.replace(tone, "")

                sp = line.split()

                #Ensure the line has utterance time markers.
                assert is_number(sp[0])
                assert is_number(sp[1])

                syls = sp[2:]
                syl_inv = syl_inv.union(syls)

                labels = syls

                assert fn.endswith(".txt")
                out_fn = fn.strip(".txt")
                i += 1
                if segmentation == "syllables":
                    out_fn = out_fn + "." + str(i) + ".syl"
                elif segmentation == "phonemes":
                    out_fn = out_fn + "." + str(i) + ".phn"
                with open(os.path.join(TGT_TXT_NORM_DIR, out_fn), "w") as out_f:
                    out_f.write(" ".join(labels))

    print(syl_inv)
    print(len(syl_inv))

    uni_phns, bi_phns, tri_phns = load_phonemes()

    print(uni_phns)
    print(bi_phns)
    print(tri_phns)

    for syl in syl_inv:
        i = 0
        while i < len(syl):
            if syl[i:i+3] in tri_phns:
                i += 3
                continue
            elif syl[i:i+2] in bi_phns:
                i += 2
                continue
            elif syl[i:i+1] in uni_phns:
                i += 1
                continue
            else:
                print(syl)
                print(len(syl))
                print(i)
                input()
                i += 1
                continue

class CorpusBatches:

    def __init__(self, feat_type, batch_size, total_size):
        self.feat_type = feat_type
        self.batch_size = batch_size
        self.total_size = total_size
