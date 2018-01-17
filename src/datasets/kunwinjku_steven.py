""" Interface to Steven's Kunwinjku data. """

import glob
import os
from os.path import join

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
