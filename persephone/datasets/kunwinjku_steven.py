""" Interface to Steven's Kunwinjku data. """

from collections import namedtuple
import glob
import os
from os.path import join

import pympi

from .. import config

ORG_DIR = config.KUNWINJKU_STEVEN_DIR
TGT_DIR = join(config.TGT_DIR, "kunwinjku-steven")

with open(config.EN_WORDS_PATH) as words_f:
    EN_WORDS = words_f.readlines()
    EN_WORDS = set([word.strip() for word in EN_WORDS])
    NA_WORDS_IN_EN_DICT = set(["Kore", "Nani", "karri", "imi", "o", "yaw", "i-",
                           "bi-", "aye", "imi", "ane", "kubba", "kab", "a-",
                           "ad", "a", "Mak", "Selim", "ngai", "en", "yo",
                           "wud", "Mani", "yak", "Manu", "ka-", "mong",
                           "manga", "ka-", "mane", "Kala", "name", "kayo",
                           "Kare", "laik", "Bale"])
    EN_WORDS = EN_WORDS.difference(NA_WORDS_IN_EN_DICT)

print("imi" in EN_WORDS)

def good_elan_paths():
    """
    Returns a list of ELAN files for recordings with good quality audio, as
    designated by Steven.
    """

    with open(join(ORG_DIR, "good-files.txt")) as path_list:
        good_paths = [path.strip() for path in path_list]

    elan_paths = []
    for path in good_paths:
        _, ext = os.path.splitext(path)
        if ext == ".eaf":
            elan_paths.append(join(ORG_DIR, path))
        else:
            full_path = join(ORG_DIR, path)
            if os.path.isdir(full_path):
                for elan_path in glob.glob('{}/**/*.eaf'.format(full_path),
                                           recursive=True):
                    elan_paths.append(elan_path)

    return elan_paths

def explore_elan_files(elan_paths):
    """
    A function to explore the tiers of ELAN files.
    """

    for elan_path in elan_paths:
        print(elan_path)
        eafob = pympi.Elan.Eaf(elan_path)
        tier_names = eafob.get_tier_names()
        for tier in tier_names:
            print("\t", tier)
            try:
                for annotation in eafob.get_annotation_data_for_tier(tier):
                    print("\t\t", annotation)
            except KeyError:
                continue

        input()

def elan_utterances():
    """ Collects utterances from various ELAN tiers. This is based on analysis
    of hte 'good' ELAN files, and may not generalize."""

    elan_tiers = {"rf", "rf@RN", "rf@MARK",
                  "xv", "xv@RN", "xv@MN", "xv@JN", "xv@EN", "xv@MARK", "xv@GN",
                  "nt@RN", "nt@JN",
                  "PRN_free", "PRN_Pfx", "NmCl_Gen", "ng_DROP",
                  "Other",
                 }

    Utterance = namedtuple("Utterance", ["file", "start_time", "end_time", "text"])

    utterances = []
    all_utts = []
    for elan_path in good_elan_paths():
        eafob = pympi.Elan.Eaf(elan_path)
        #import pprint; pprint.pprint(dir(eafob))
        can_find_path = False
        for md in eafob.media_descriptors:
            try:
                media_path = os.path.join(os.path.dirname(elan_path),
                                          md["RELATIVE_MEDIA_URL"])
                if os.path.exists(media_path):
                    # Only one media_path file is needed, as long as it exists.
                    can_find_path = True
                    break
                # Try just looking for the basename specified in the
                # RELATIVE_MEDIA_URL
                media_path = os.path.join(os.path.dirname(elan_path),
                                          os.path.basename(md["RELATIVE_MEDIA_URL"]))
                if os.path.exists(media_path):
                    can_find_path = True
                    break
            except KeyError:
                # Then it might be hard to find the MEDIA URL if its not
                # relative. Keep trying.
                continue
        if can_find_path:
            tier_names = eafob.get_tier_names()
            for tier in tier_names:
                if tier.startswith("rf") or tier.startswith("xv") or tier in elan_tiers:
                    for annotation in eafob.get_annotation_data_for_tier(tier):
                        utterance = Utterance(media_path, *annotation[:3])
                        if not utterance.text.strip() == "":
                            utterances.append(utterance)
        else:
            print("Warning: Can't find the media file for {}".format(elan_path))

    return utterances

def segment_gup_phonemes(utterance):
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    utterance = utterance.lower()
    print(utterance)

def explore_code_switching():

    utters = elan_utterances()

    en_count = 0
    for utter in utters:
        for word in utter.text.split():
            if word in EN_WORDS:
                en_count += 1
                print(utter.text)
                print("\t" + repr(word))
                #input()
                break
    print(en_count)
    print(len(utters))

"""
rf
rf@RN
xv (Esther_1.eaf, but has some English in it)
PRN_free, PRN_Pfx, NmCl_Gen, ng_DROP (Esther_2.eaf. Code switching might be an
	issue. Otherwise Esther files are promising for the finer granularity that
	is additionally included.)
xv@RN (Kabo.eaf, but its a duplicate of the xv tier. More pure in kunboy.eaf)
xv@MN, LEX (manbarndarr.eaf)
xv@JN, PRN_pfx, PRFX (Manbedgje.eaf)
Other (Manbulu.eaf)
nt@JN (Mandeb.eaf, but also has English in Mandjarduk_b.eaf)
xv@EN (mandjimdjim.eaf)
nt@RN (20161013_manmorlak.eaf)
rf@MARK, xv@MARK (Mark on rock. Has a few different tiers, some of which have
multiple text fields. A bit of code switching).
xv@GN (Marys_Yirlinkirrkirr.eaf)
A bunch more xv@s (Njanjma_Injalak_weaving.eaf)
Topic Index (Terrah_ngalwarrngurru.eaf)

Can just check across all these tiers and any that are empty or contain empty
strings I can just ignore.
"""
