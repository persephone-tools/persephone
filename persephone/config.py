""" Configuration for some variables that may be system dependent. """
import configparser
import os


config_file = configparser.ConfigParser()
config_file.read('settings.ini')

CORPORA_BASE_PATH = config_file.get("PATHS", "CORPORA_BASE_PATH",
                                    fallback="/lt/work/oadams/")

# The directory of the original source corpora, un-preprocessed.
# Shared across research group
TIMIT_ORG_DIR = "/lt/data/timit/timit/"
# Personal corpora files
GRIKO_DIR = os.path.join(CORPORA_BASE_PATH, "griko-data")
CHATINO_DIR = os.path.join(CORPORA_BASE_PATH, "chatino", "CTP")
JAPHUG_DIR = os.path.join(CORPORA_BASE_PATH, "japhug")
NA_DIR = os.path.join(CORPORA_BASE_PATH, "Na")

# The directory where the preprocessed data will be held.

TGT_DIR = config_file.get("PATHS", "TARGET", fallback="./data")
# The path for experiments
EXP_DIR = config_file.get("PATHS", "EXPERIMENTS", fallback="./exp")


# TODO: fix these stubs
KUNWINJKU_STEVEN_DIR = ""
EN_WORDS_PATH = ""
