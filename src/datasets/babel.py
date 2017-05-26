""" Preprocess and interface with the LORELEI Babel data. """

import os
import subprocess

from context_manager import cd

ORG_BABEL_DIR = ("/scratch/ariel/data/IARPA-BABEL-unpacked/oasis/projects/"
                 "nsf/cmu131/fmetze/babel-corpus/")
WORK_BABEL_DIR = "/lt/work/oadams/babel/"

LANG_DIR_MAP = {"turkish":("105-B/BABEL_BP_105", "105-turkish"),
                "tagalog":("106-B/BABEL_BP_106", "106-tagalog"),
                "assamese":("102-B/BABEL_OP1_102", "102-assamese"),
                "bengali":("103-B/BABEL_OP1_103", "103-bengali"),
                "pashto":("104-B/BABEL_BP_104", "104-pashto"),
                "vietnamese":("107-B/babel107b-v0.7-build/BABEL_BP_107", "107-vietnamese"),
                "creole":("201-B/BABEL_OP1_201", "201-creole"),
                "lao":("203-B/BABEL_OP1_203-v3.1a", "203-lao"),
                "tamil":("204-B/IARPA-babel204b-v1.1b/BABEL_OP1_204", "204-tamil"),
                "kurdish":("IARPA-babel205b-v1.0a-build/BABEL_OP2_205", "205-kurdish"),
                "zulu":("206-B/BABEL_OP1_206-v0.1e", "206-zulu"),
                "tokpisin":("IARPA-babel207b-v1.0e-build/BABEL_OP2_207", "207-tokpisin"),
                "cebuano":("IARPA-babel301b-v2.0b-build/BABEL_OP2_301", "301-cebuano"),
                "kazakh":("IARPA-babel302b-v1.0a-build/BABEL_OP2_302", "302-kazakh")}
LANGS = list(LANG_DIR_MAP.keys())

def sph2wav(input_fn, output_fn):
    """ Call sph2pipe to convert alaw .sph files to ulaw .wav files, which will
    allow for trimming with sox.
    """
    args = ["sph2pipe", "-p", "-f", "rif", input_fn, output_fn]
    subprocess.run(args)

def babel_sph2wav(langs=LANGS):
    """ Converts the Babel data into WAV formats."""

    for lang in langs:
        org_lang_dir = os.path.join(ORG_BABEL_DIR, LANG_DIR_MAP[lang][0])
        with cd(org_lang_dir):
            for input_dir, _, input_fns in os.walk("."):
                for input_fn in input_fns:
                    if input_fn.endswith(".sph"):
                        output_dir = os.path.join(WORK_BABEL_DIR,
                                               LANG_DIR_MAP[lang][1], input_dir)
                        if not os.path.isdir(output_dir):
                            os.makedirs(output_dir)
                        output_fn = os.path.splitext(input_fn)[0] + ".wav"
                        output_path = os.path.join(output_dir, output_fn)
                        input_path = os.path.join(input_dir, input_fn)
#                        print(input_path)
                        print(output_path)
                        sph2wav(input_path, output_path)
