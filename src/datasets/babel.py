""" Preprocess and interface with the LORELEI Babel data. """

import os
import shutil
import subprocess

from context_manager import cd
import utils

ORG_BABEL_DIR = ("/scratch/ariel/data/IARPA-BABEL-unpacked/oasis/projects/"
                 "nsf/cmu131/fmetze/babel-corpus/")
WORK_BABEL_DIR = "/lt/work/oadams/babel/"
BABELIPA_DIR = "/home/oadams/data/babelipa/"

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
    """ Converts the Babel sphere files in the original directory to WAV files
    in the working directory.
    """

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
                        sph2wav(input_path, output_path)

def babelipa_transcriptions(langs=LANGS):
    """ Puts IPA transcriptions of the Babel data in the working directory. """
    # TODO Currently just moves the conversational transcripts Antonis
    # generated. Should generalize to the scripted too, as we'll want that for
    # training data.

    for lang in langs:
        org_lang_dir = os.path.join(BABELIPA_DIR, LANG_DIR_MAP[lang][1])
        with cd(org_lang_dir):
            for input_dir, leaf_dirs, input_fns in os.walk("."):
                for leaf_dir in leaf_dirs:
                    if leaf_dir == "ipatranscription":
                        input_path = os.path.join(input_dir, leaf_dir)
                        print(input_path)
                        output_path = os.path.join(WORK_BABEL_DIR,
                                                   LANG_DIR_MAP[lang][1],
                                                   "conversational",
                                                   input_dir,
                                                   leaf_dir)
                        print(output_path)
                        shutil.copytree(input_path, output_path)

def split_wavs_and_txts(langs=LANGS):
    """ Based on timing information in transcriptions, splits the WAV files and
    the transcriptions such that there is one file per utterance. Puts the
    split files in an 'utters' subdir of our working Babel dir.
    """

    def get_wav_path(txt_path):
        """ Given a transcription path, gets the corresponding wav."""

        pre, ext = os.path.splitext(os.path.basename(txt_path))
        parent_dir = os.path.dirname(os.path.dirname(txt_path))
        wav_path = os.path.join(parent_dir, "audio", "%s.wav" % pre)
        return wav_path

    def split_wav_and_txt(input_wav_path, input_txt_path):
        """ Takes the path to an individual WAV and txt file and outputs a
        number of wavs and txt files corresponding to invidual utterances."""

        times = []
        utter_txts = []
        with open(input_txt_path) as input_txt_f:
            for line in input_txt_f:
                if line.startswith("[") and line.endswith("]\n"):
                    # Len 9 line indicates times. Eg. "[577.805]"
                    num_str = line.replace("[", "").replace("]", "")
                    times.append(float(num_str))
                else:
                    utter_txts.append(line)

        assert len(times) == len(utter_txts) + 1

        for i, utter_txt in enumerate(utter_txts):

            if utter_txt.strip() != "<no-speech>":
                # Split the wav
                output_wav_path = os.path.join(output_dir, input_wav_path)
                pre, ext = os.path.splitext(output_wav_path)
                output_wav_path = pre + "_utter%s" % i + ext
                utils.make_parent(output_wav_path)
                start_time = times[i]
                end_time = times[i+1]
                utils.trim_wav(input_wav_path, output_wav_path,
                               start_time, end_time)

                # Split the transcription
                output_txt_path = os.path.join(output_dir, input_txt_path)
                pre, ext = os.path.splitext(output_txt_path)
                output_txt_path = pre + "_utter%s" % i + ext
                utils.make_parent(output_txt_path)
                print(output_txt_path)
                print(output_wav_path)
                with open(output_txt_path, "w") as output_txt_f:
                    print(utter_txt, end="", file=output_txt_f)

    for lang in langs:
        org_lang_dir = os.path.join(WORK_BABEL_DIR, LANG_DIR_MAP[lang][1])
        output_dir = os.path.join(WORK_BABEL_DIR, "utters",
                                  LANG_DIR_MAP[lang][1])
        with cd(org_lang_dir):
            for input_dir, _, input_fns in os.walk("."):
                for input_fn in input_fns:
                    if input_fn.endswith(".txt"):
                        input_txt_path = os.path.join(input_dir, input_fn)
                        input_wav_path = get_wav_path(input_txt_path)
                        print(input_txt_path)
                        print(input_wav_path)
                        split_wav_and_txt(input_wav_path, input_txt_path)
