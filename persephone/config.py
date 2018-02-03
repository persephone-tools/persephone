""" Some variables that may be system dependent. """
import os
PERSONAL_CORPORA_BASE_DIR = "/lt/work/oadams/"

# The directory of the original source corpora, un-preprocessed.
# Shared across research group
TIMIT_ORG_DIR = "/lt/data/timit/timit/"
# Personal corpora files
GRIKO_DIR = os.path.join(PERSONAL_CORPORA_BASE_DIR, "griko-data")
CHATINO_DIR = os.path.join(PERSONAL_CORPORA_BASE_DIR, "chatino", "CTP")
JAPHUG_DIR = os.path.join(PERSONAL_CORPORA_BASE_DIR, "japhug")
NA_DIR = os.path.join(PERSONAL_CORPORA_BASE_DIR, "Na")

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
