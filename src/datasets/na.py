""" An interface with the Na data. """

import os

import config

ORG_DIR = config.NA_DIR
TGT_DIR = "../data/na"
ORG_TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")
TGT_TXT_NORM_DIR = os.path.join(TGT_DIR, "txt_norm")

TO_REMOVE = {"|", "ǀ", "↑", "«", "»", "¨", "“", "”", "D", "F"}
WORDS_TO_REMOVE = {"CHEVRON", "audible", "qʰʰʰʰʰ", "qʰʰʰʰ", "D"}
TONES = ["˧˥", "˩˥", "˩˧", "˧˩", "˩", "˥", "˧"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g', 'w̃',
        'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'ṽ̩', 'ɻ̩', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
        't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ', 's',
        'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'ɻ̃', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'ṽ̩', 'o̥', 'ts',
        'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ', 'ĩ',
        'õ', 'dz'}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩"}

if not os.path.isdir(TGT_DIR):
    os.makedirs(TGT_DIR)

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

def segment_phonemes(syls):
    """ Segments a list of syllables into phonemes. """

    phonemes = []
    for syl in syls:
        i = 0
        while i < len(syl):
            if syl[i:i+3] in TRI_PHNS:
                phonemes.append(syl[i:i+3])
                i += 3
                continue
            elif syl[i:i+2] in BI_PHNS:
                phonemes.append(syl[i:i+2])
                i += 2
                continue
            elif syl[i:i+1] in UNI_PHNS:
                phonemes.append(syl[i:i+1])
                i += 1
                continue
            else:
                raise Exception("Failed to segment syllable: %s" % syl)
    return phonemes

def prepare(segmentation, remove_tones=True):
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
                start_time = sp[0]
                end_time = sp[1]
                #Ensure the line has utterance time markers.
                assert is_number(start_time)
                assert is_number(end_time)

                syls = sp[2:]
                syl_inv = syl_inv.union(syls)

                assert fn.endswith(".txt")
                out_fn = fn.strip(".txt")
                i += 1
                if segmentation == "syllables":
                    out_fn = out_fn + "." + str(i) + ".syl"
                    labels = syls
                elif segmentation == "phonemes":
                    out_fn = out_fn + "." + str(i) + ".phn"
                    labels = segment_phonemes(syls)

                with open(os.path.join(TGT_TXT_NORM_DIR, out_fn), "w") as out_f:
                    out_f.write(" ".join(labels))

    print(syl_inv)
    print(len(syl_inv))


class CorpusBatches:

    def __init__(self, feat_type, batch_size, total_size):
        self.feat_type = feat_type
        self.batch_size = batch_size
        self.total_size = total_size
