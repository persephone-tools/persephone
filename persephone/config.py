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
TGT_DIR = "./data"
# The path for experiments
EXP_DIR = "./exp"

# The path to the sox tool for converting NIST format to WAVs.
SOX_PATH = "/home/oadams/tools/sox-14.4.2/src/sox"
OPENFST_BIN_PATH = "/home/oadams/tools/openfst-1.6.2/src/bin"
FFMPEG_PATH = "/home/oadams/tools/ffmpeg-3.3/ffmpeg"
KALDI_ROOT = "/home/oadams/tools/kaldi"

# TODO: fix these stubs
KUNWINJKU_STEVEN_DIR = ""
EN_WORDS_PATH = ""
