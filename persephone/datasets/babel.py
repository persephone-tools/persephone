""" Preprocess and interface with the LORELEI Babel data. """

import os
import shutil
import subprocess

from ..context_manager import cd
from .. import corpus
from .. import feat_extract
from .. import utils

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

def prepare(langs=LANGS, feat_type="log_mel_filterbank"):
#    langs.remove("assamese")
    babel_sph2wav(langs)
    babelipa_transcriptions(langs)
    split_wavs_and_txts(langs)
    feat_extraction(langs, feat_type)

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
        print("babel_sph2wav %s" % lang)
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
        print("babelipa_transcriptions %s" % lang)
        org_lang_dir = os.path.join(BABELIPA_DIR, LANG_DIR_MAP[lang][1])
        with cd(org_lang_dir):
            for input_dir, leaf_dirs, input_fns in os.walk("."):
#                print(input_dir, leaf_dirs, input_fns)
                for leaf_dir in leaf_dirs:
                    if leaf_dir == "ipatranscription":
#                        for input_fn in input_fns:
#                        print(input_dir)
                        input_path = os.path.join(input_dir, leaf_dir)
                        print("Input path:", input_path)
                        output_path = os.path.join(WORK_BABEL_DIR,
                                                   LANG_DIR_MAP[lang][1],
                                                   "conversational",
                                                   input_path)
#                                                   leaf_dir)
                        print("Output path:", output_path)
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

                start_time = times[i]
                end_time = times[i+1]

                if end_time - start_time < 10:
                    # To ensure we don't end up with utterances too long

                    # Split the wav
                    output_wav_path = os.path.join(output_dir, input_wav_path)
                    pre, ext = os.path.splitext(output_wav_path)
                    output_wav_path = pre + "_utter%s" % i + ext
                    utils.make_parent(output_wav_path)
                    utils.trim_wav(input_wav_path, output_wav_path,
                                   start_time, end_time)

                    # Split the transcription
                    output_txt_path = os.path.join(output_dir, input_txt_path)
                    pre, ext = os.path.splitext(output_txt_path)
                    output_txt_path = pre + "_utter%s" % i + ext
                    utils.make_parent(output_txt_path)
                    with open(output_txt_path, "w") as output_txt_f:
                        print(utter_txt, end="", file=output_txt_f)

    for lang in langs:
        print("Splitting wavs for %s" % lang)
        org_lang_dir = os.path.join(WORK_BABEL_DIR, LANG_DIR_MAP[lang][1])
        output_dir = os.path.join(WORK_BABEL_DIR, "utters",
                                  LANG_DIR_MAP[lang][1])
        with cd(org_lang_dir):
            for input_dir, _, input_fns in os.walk("."):
                for input_fn in input_fns:
                    if input_fn.endswith(".txt"):
                        input_txt_path = os.path.join(input_dir, input_fn)
                        input_wav_path = get_wav_path(input_txt_path)
                        split_wav_and_txt(input_wav_path, input_txt_path)


def feat_extraction(langs, feat_type):
    """ Extracts features from all the utterances. """

    if feat_type != "log_mel_filterbank":
        raise Exception("Feature type %s not implemented." % feat_type)

    count = 0
    utters_dir = os.path.join(WORK_BABEL_DIR, "utters")
    for lang in langs:
        print("Feature extraction for %s" % lang)
        lang_dir = os.path.join(utters_dir, LANG_DIR_MAP[lang][1])
        for input_dir, _, input_fns in os.walk(lang_dir):
            for input_fn in input_fns:
                if input_fn.endswith(".wav"):
                    count += 1
                    input_path = os.path.join(input_dir, input_fn)
                    print(count, input_path)
                    feat_extract.logfbank_feature_extraction(input_path)

def load_phns(lang):
    """ Loads the phone lexicon for the given language."""

    phn_lex_path = os.path.join(BABELIPA_DIR, LANG_DIR_MAP[lang][1],
                                "phone.lexicon.txt")
    phones = []
    with open(phn_lex_path) as phn_f:
        for line in phn_f:
            phones.append(line.split()[0])

    return phones

def get_txt_path(feat_path):
    """ Given a transcription path, gets the corresponding wav."""

    pre, ext = os.path.splitext(os.path.basename(feat_path))
    parent_dir = os.path.dirname(os.path.dirname(feat_path))
    txt_path = os.path.join(parent_dir, "ipatranscription", "%s.txt" % pre)
    return txt_path

def filter_by_size(feat_paths, max_samples):
    new_feat_paths = []
    for feat_path in feat_paths:
        _, batch_x_lens = utils.load_batch_x([feat_path], flatten=True)
        if batch_x_lens[0] <= max_samples:
            new_feat_paths.append(feat_path)
    return new_feat_paths


class Corpus(corpus.AbstractCorpus):

    def __init__(self, langs,
                 feat_type="log_mel_filterbank", tgt_type="phn",
                 max_samples=1000, scripted=False):

        if tgt_type != "phn":
            raise Exception("Target type %s not implemented." % tgt_type)

        dev_utters = []
        eval_utters = []
        train_utters = []

        feat_paths = []
        # Create list of utterance_ids based on the languages, 
        for lang in langs:
            lang_dir = os.path.join(WORK_BABEL_DIR, "utters",
                                    LANG_DIR_MAP[lang][1])
            if not scripted:
                # Then we exclusively use conversational training data.
                lang_dir = os.path.join(lang_dir, "conversational")

            # Load transcription paths
            for input_dir, _, input_fns in os.walk(lang_dir):
                for input_fn in input_fns:
                    if input_fn.endswith(".log_mel_filterbank.npy"):
                        feat_path = os.path.join(input_dir, input_fn)
                        feat_paths.append(feat_path)

        #if max_samples:
        #    feat_paths = filter_by_size(feat_paths, max_samples)

        print(len(feat_paths))

        txt_paths = [get_txt_path(feat_path) for feat_path in feat_paths]
        train_feat_paths = [feat_path for feat_path in feat_paths
                            if "/training/" in feat_path]
        train_txt_paths = [txt_path for txt_path in txt_paths
                           if "/training/" in txt_path]
        dev_feat_paths = [feat_path for feat_path in feat_paths
                          if "/dev/" in feat_path]
        dev_txt_paths = [txt_path for txt_path in txt_paths
                         if "/dev/" in txt_path]

        self.phonemes = load_phns(lang)

        self.PHONEME_TO_INDEX = {phn: index for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.INDEX_TO_PHONEME = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.vocab_size = len(self.phonemes)

    def get_valid_fns(self):
        return dev_feat_paths, dev_txt_paths

    def get_train_fns(self):
        return train_feat_paths, train_txt_paths

