""" Interface to Steven's Kunwinjku data. """

import glob
import os
from os.path import join

import pympi

import config

ORG_DIR = config.KUNWINJKU_STEVEN_DIR
TGT_DIR = join(config.TGT_DIR, "kunwinjku-steven")

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
