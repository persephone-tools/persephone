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
from pathlib import Path

config_file = configparser.ConfigParser()
config_file.read('settings.ini')

CORPORA_BASE_PATH = config_file.get("PATHS", "CORPORA_BASE_PATH",
                                    fallback="./data/org/")

# The directory of the original source corpora, un-preprocessed.
NA_DIR = config_file.get("PATHS", "NA_DIR", fallback=os.path.join(CORPORA_BASE_PATH, "Na"))

# For Kunwinjku data:
BKW_PATH = config_file.get("PATHS", "BKW_PATH",
    fallback=os.path.join(CORPORA_BASE_PATH, "BKW-good"))
EN_WORDS_PATH = config_file.get("PATHS", "EN_WORDS_PATH",
    fallback=os.path.join(CORPORA_BASE_PATH, "english-words/words.txt"))

# The directory where the preprocessed data will be held.
TGT_DIR = config_file.get("PATHS", "TARGET", fallback="./data")
# The path for experiments
EXP_DIR = config_file.get("PATHS", "EXPERIMENTS", fallback="./exp")

# The path to the sox tool; currently used for splitting WAVs, but we can
# replace with pydub. Actually, the pydub approach is slow so now
# wav.trim_wav_ms tries to use sox and then fallsback to pydub/ffmpeg
SOX_PATH = config_file.get("PATHS", "SOX_PATH", fallback="sox")
# FFMPEG is used for normalizing WAVs
FFMPEG_PATH = config_file.get("PATHS", "FFMPEG_PATH", fallback="ffmpeg")
# Kaldi is used for pitch extraction
KALDI_ROOT = config_file.get("PATHS", "KALDI_ROOT_PATH", fallback="/home/oadams/tools/kaldi")
# Used for lattice output
OPENFST_BIN_PATH = config_file.get("PATHS", "OPEN_FST_BIN_PATH", fallback="/home/oadams/tools/openfst-1.6.2/src/bin")

TEST_TGT_DATA_ROOT = config_file.get("PATHS", "TEST_TGT_DATA_ROOT",
                                      fallback="./testing/data/")

LOGGING_INI_PATH = config_file.get("PATHS", "log_ini_path", fallback="./logging.ini")
