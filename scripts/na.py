import os

import config

ORG_DIR = config.NA_DIR
TGT_DIR = "../data/na"
TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")

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

    return uni_phns, bi_phns

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def remove_multi(a, ys):
    """ Removes instances of a from the list ys."""
    return list(filter(lambda x: x != a, ys))

def strip_tones(syl):
    """ Strips tones from Na syllables."""
    tones = ["˧˥", "˩˥", "˩˧", "˧˩", "˩", "˥", "˧"]
    for tone in tones:
        if syl.endswith(tone):
            return syl[:-len(tone)]
    return syl

def trim_wavs(remove_tones=True):
    """ Trims available wavs into the sentence or utterance-level."""


    utters = []
    syl_inv = set()
    for fn in os.listdir(TXT_NORM_DIR):
        with open(os.path.join(TXT_NORM_DIR, fn)) as f:
            for line in f:
                sp = line.split()
                assert is_number(sp[0])
                assert is_number(sp[1])
                syls = remove_multi("|", sp[2:])
                #print(syls)
                if remove_tones:
                    syls = [strip_tones(syl) for syl in syls]
                #print(syls)
                syls_join = " ".join(syls)
                if "CHEVRON" not in syls_join:
                    utters.append(syls_join)
                    syl_inv = syl_inv.union(syls)

    print(syl_inv)
    print(len(syl_inv))

    uni_phns, bi_phns = load_phonemes()

    print(uni_phns)
    print(bi_phns)

    for syl in syl_inv:
        i = 0
        while i < len(syl):
            if syl[i:i+2] in bi_phns:
                i += 2
                continue
            elif syl[i:i+1] in uni_phns:
                i += 1
                continue
            else:
                print(syl[i:i+2])
                print(syl)
                print(i)
                input()
                i += 1
                continue

