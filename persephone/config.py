""" Configuration for some variables that may be system dependent.

Supply a settings.ini file to override values. The format of the file will
have the following sections

[PATHS]
CORPORA_BASE_PATH = /my/path/to/corpora
TARGET = /path/to/preprocessed/data
EXPERIMENTS = /path/to/experiment/output
"""
import configparser
import os


config_file = configparser.ConfigParser()
config_file.read('settings.ini')

CORPORA_BASE_PATH = config_file.get("PATHS", "CORPORA_BASE_PATH",
                                    fallback="/lt/work/oadams/")

# The directory of the original source corpora, un-preprocessed.
CHATINO_DIR = os.path.join(CORPORA_BASE_PATH, "chatino", "CTP")
NA_DIR = os.path.join(CORPORA_BASE_PATH, "Na")
# Personal corpora files
# TODO: fix these stubs
KUNWINJKU_STEVEN_DIR = config_file.get("PATHS", "KUNWINJKU_STEVEN_PATH", fallback="")
EN_WORDS_PATH = config_file.get("PATHS", "EN_WORDS_PATH", fallback="")
BUTCHER_DIR = os.path.join(CORPORA_BASE_PATH, "OA_Data")

# The directory where the preprocessed data will be held.
TGT_DIR = config_file.get("PATHS", "TARGET", fallback="./data")
# The path for experiments
EXP_DIR = config_file.get("PATHS", "EXPERIMENTS", fallback="./exp")

# The path to the sox tool; currently used for splitting WAVs, but we can
# replace with pydub TODO
SOX_PATH = config_file.get("PATHS", "SOX_BASE_PATH", fallback="/home/oadams/tools/sox-14.4.2/src/sox")
# FFMPEG is used for normalizing WAVs
FFMPEG_PATH = config_file.get("PATHS", "FFMPEG_PATH", fallback="ffmpeg")
# Kaldi is used for pitch extraction
KALDI_ROOT = config_file.get("PATHS", "KALDI_ROOT_PATH", fallback="/home/oadams/tools/kaldi")
# Used for lattice output
OPENFST_BIN_PATH = config_file.get("PATHS", "OPEN_FST_BIN_PATH", fallback="/home/oadams/tools/openfst-1.6.2/src/bin")
