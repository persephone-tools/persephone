""" An interface with the Na data. """

import os
import random
import subprocess
from subprocess import PIPE

import numpy as np
import xml.etree.ElementTree as ET

from .. import config
from .. import corpus
from .. import feat_extract
from . import pangloss
from .. import utils

ORG_DIR = config.NA_DIR
# TODO eventually remove "new" when ALTA experiments are finished.
TGT_DIR = os.path.join(config.TGT_DIR, "na", "new")
ORG_XML_DIR = os.path.join(ORG_DIR, "xml")
ORG_WAV_DIR = os.path.join(ORG_DIR, "wav")
TGT_WAV_DIR = os.path.join(TGT_DIR, "wav")
FEAT_DIR = os.path.join(TGT_DIR, "feat")
LABEL_DIR = os.path.join(TGT_DIR, "label")
TRANSL_DIR = os.path.join(TGT_DIR, "transl")

# The directory for untranscribed audio we want to transcribe with automatic
# methods.
UNTRAN_DIR = os.path.join(TGT_DIR, "untranscribed")

#PREFIXES = [os.path.splitext(fn)[0]
#            for fn in os.listdir(ORG_TRANSCRIPT_DIR)
#            if fn.endswith(".txt")]

# TODO Move into feat creation functions.
if not os.path.isdir(TGT_DIR):
    os.makedirs(TGT_DIR)

if not os.path.isdir(FEAT_DIR):
    os.makedirs(FEAT_DIR)

# HARDCODED values
MISC_SYMBOLS = [' ̩', '~', '=', ':', 'F', '¨', '↑', '“', '”', '…', '«', '»',
'D', 'a', 'ː', '#', '$', "‡"]
BAD_NA_SYMBOLS = ['D', 'F', '~', '…', '=', '↑', ':']
PUNC_SYMBOLS = [',', '!', '.', ';', '?', "'", '"', '*', ':', '«', '»', '“', '”', "ʔ"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ',
           'ĩ', 'õ', 'dz', "ɻ̍", "wæ", "wɑ", "wɤ", "jæ", "jɤ", "jo"}
FILLERS = {"əəə…", "mmm…"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩", "ɻ̩̃", "wæ̃", "w̃æ"}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)
SYMBOLS_TO_PREDICT = {"|"}

PHONEMES = UNI_PHNS.union(BI_PHNS).union(TRI_PHNS).union(FILLERS)

# TODO Get rid of these variables, as they're not used in the class, only for
# preparing phonemes_onehot feats.
PHONEMES_TO_INDICES = {phn: index for index, phn in enumerate(PHONEMES)}
INDICES_TO_PHONEMES = {index: phn for index, phn in enumerate(PHONEMES)}

# TODO Potentially remove?
#PHONES_TONES = sorted(list(PHONES.union(set(TONES)))) # Sort for determinism
#PHONESTONES2INDICES = {phn_tone: index for index, phn_tone in enumerate(PHONES_TONES)}
#INDICES2PHONESTONES = {index: phn_tone for index, phn_tone in enumerate(PHONES_TONES)}
#TONES2INDICES = {tone: index for index, tone in enumerate(TONES)}
#INDICES2TONES = {index: tone for index, tone in enumerate(TONES)}

def preprocess_na(sent, label_type):

    if label_type == "phonemes_and_tones":
        phonemes = True
        tones = True
        tgm = True
    elif label_type == "phonemes_and_tones_no_tgm":
        phonemes = True
        tones = True
        tgm = False
    elif label_type == "phonemes":
        phonemes = True
        tones = False
    elif label_type == "tones":
        phonemes = False
        tones = True
        tgm = True
    elif label_type == "tones_notgm":
        phonemes = False
        tones = True
        tgm = False
    else:
        raise Exception("Unrecognized label type: %s" % label_type)

    def pop_phoneme(sentence):
        # TODO desperately needs refactoring

        # Treating fillers as single tokens; normalizing to əəə and mmm
        if phonemes:
            if sentence[:4] in ["əəə…", "mmm…"]:
                return sentence[:4], sentence[4:]
            if sentence.startswith("ə…"):
                return "əəə…", sentence[2:]
            if sentence.startswith("m…"):
                return "mmm…", sentence[2:]
            if sentence.startswith("mm…"):
                return "mmm…", sentence[3:]

        # Normalizing some stuff
        if sentence[:3] == "wæ̃":
            if phonemes:
                return "w̃æ", sentence[3:]
            else:
                return None, sentence[3:]
        if sentence[:3] == "ṽ̩":
            if phonemes:
                return "ṽ̩", sentence[3:]
            else:
                return None, sentence[3:]

        if sentence[:3] in TRI_PHNS:
            if phonemes:
                return sentence[:3], sentence[3:]
            else:
                return None, sentence[3:]
        if sentence[:2] in BI_PHNS:
            if phonemes:
                return sentence[:2], sentence[2:]
            else:
                return None, sentence[2:]
        if sentence[0] in UNI_PHNS:
            if phonemes:
                return sentence[0], sentence[1:]
            else:
                return None, sentence[1:]
        if sentence[:2] in BI_TONES:
            if tones:
                return sentence[:2], sentence[2:]
            else:
                return None, sentence[2:]
        if sentence[0] in UNI_TONES:
            if tones:
                return sentence[0], sentence[1:]
            else:
                return None, sentence[1:]
        if sentence[0] in MISC_SYMBOLS:
            # We assume these symbols cannot be captured.
            return None, sentence[1:]
        if sentence[0] in BAD_NA_SYMBOLS:
            return None, sentence[1:]
        if sentence[0] in PUNC_SYMBOLS:
            return None, sentence[1:]
        if sentence[0] in ["-", "ʰ", "/"]:
            return None, sentence[1:]
        if sentence[0] in set(["<", ">"]):
            # We keep everything literal, thus including what is in <>
            # brackets; so we just remove these tokens"
            return None, sentence[1:]
        if sentence[0] == "[":
            # It's an opening square bracket, so ignore everything until we
            # find a closing one.
            if sentence.find("]") == len(sentence)-1:
                # If the closing bracket is the last char
                return None, ""
            else:
                return None, sentence[sentence.find("]")+1:]
        if sentence[0] in set([" ", "\t", "\n"]):
            # Return a space char so that it can be identified in word segmentation
            # processing.
            return " ", sentence[1:]
        if sentence[0] == "|" or sentence[0] == "ǀ":
            if tgm:
                return "|", sentence[1:]
            else:
                return None, sentence[1:]
        print("***" + sentence)
        raise Exception("Next character not recognized: " + sentence[:1])

    def filter_for_phonemes(sentence):
        """ Returns a sequence of phonemes and pipes (word delimiters). Tones,
        syllable boundaries, whitespace are all removed."""

        filtered_sentence = []
        while sentence != "":
            phoneme, sentence = pop_phoneme(sentence)
            if phoneme != " ":
                filtered_sentence.append(phoneme)
        filtered_sentence = [item for item in filtered_sentence if item != None]
        return " ".join(filtered_sentence)

    # Filter utterances with certain words
    if "BEGAIEMENT" in sent:
        return ""
    sent = filter_for_phonemes(sent)
    return sent

def preprocess_french(trans, fr_nlp, remove_brackets_content=True):
    """ Takes a list of sentences in french and preprocesses them."""

    if remove_brackets_content:
        trans = datasets.pangloss.remove_content_in_brackets(trans, "[]")
    # Not sure why I have to split and rejoin, but that fixes a Spacy token
    # error.
    trans = fr_nlp(" ".join(trans.split()[:]))
    #trans = fr_nlp(trans)
    trans = " ".join([token.lower_ for token in trans if not token.is_punct])

    return trans

def trim_wavs():
    """ Extracts sentence-level transcriptions, translations and wavs from the
    Na Pangloss XML and WAV files. But otherwise doesn't preprocess them."""

    print("Trimming wavs...")

    if not os.path.exists(os.path.join(TGT_WAV_DIR, "TEXT")):
        os.makedirs(os.path.join(TGT_WAV_DIR, "TEXT"))
    if not os.path.exists(os.path.join(TGT_WAV_DIR, "WORDLIST")):
        os.makedirs(os.path.join(TGT_WAV_DIR, "WORDLIST"))

    for fn in os.listdir(ORG_XML_DIR):
        print(fn)
        path = os.path.join(ORG_XML_DIR, fn)
        prefix, _ = os.path.splitext(fn)

        rec_type, sents, times, transls = datasets.pangloss.get_sents_times_and_translations(path)

        # Extract the wavs given the times.
        for i, (start_time, end_time) in enumerate(times):
            if prefix.endswith("PLUSEGG"):
                in_wav_path = os.path.join(ORG_WAV_DIR, prefix.upper()[:-len("PLUSEGG")]) + ".wav"
            else:
                in_wav_path = os.path.join(ORG_WAV_DIR, prefix.upper()) + ".wav"
            headmic_path = os.path.join(ORG_WAV_DIR, prefix.upper()) + "_HEADMIC.wav"
            if os.path.isfile(headmic_path):
                in_wav_path = headmic_path

            out_wav_path = os.path.join(TGT_WAV_DIR, rec_type, "%s.%d.wav" % (prefix, i))
            assert os.path.isfile(in_wav_path)
            utils.trim_wav(in_wav_path, out_wav_path, start_time, end_time)

def prepare_transls():
    """ Prepares the French translations. """

    import spacy
    fr_nlp = spacy.load("fr")

    if not os.path.exists(os.path.join(TRANSL_DIR, "TEXT")):
        os.makedirs(os.path.join(TRANSL_DIR, "TEXT"))
    if not os.path.exists(os.path.join(TRANSL_DIR, "WORDLIST")):
        os.makedirs(os.path.join(TRANSL_DIR, "WORDLIST"))

    for fn in os.listdir(ORG_XML_DIR):
        print(fn)
        path = os.path.join(ORG_XML_DIR, fn)
        prefix, _ = os.path.splitext(fn)

        rec_type, sents, times, transls = datasets.pangloss.get_sents_times_and_translations(path)

        # Tokenize the French translations and write them to file.
        transls = [preprocess_french(transl[0], fr_nlp) for transl in transls]
        for i, transl in enumerate(transls):
            out_prefix = "%s.%d" % (prefix, i)
            transl_path = os.path.join(TRANSL_DIR, rec_type, out_prefix + ".fr.txt")
            with open(transl_path, "w") as transl_f:
                print(transl, file=transl_f)

def prepare_labels(label_type):
    """ Prepare the neural network output targets."""

    if not os.path.exists(os.path.join(LABEL_DIR, "TEXT")):
        os.makedirs(os.path.join(LABEL_DIR, "TEXT"))
    if not os.path.exists(os.path.join(LABEL_DIR, "WORDLIST")):
        os.makedirs(os.path.join(LABEL_DIR, "WORDLIST"))

    for fn in os.listdir(ORG_XML_DIR):
        print(fn)
        path = os.path.join(ORG_XML_DIR, fn)
        prefix, _ = os.path.splitext(fn)

        rec_type, sents, times, transls = datasets.pangloss.get_sents_times_and_translations(path)
        # Write the sentence transcriptions to file
        sents = [preprocess_na(sent, label_type) for sent in sents]
        for i, sent in enumerate(sents):
            if sent.strip() == "":
                # Then there's no transcription, so ignore this.
                continue
            out_fn = "%s.%d.%s" % (prefix, i, label_type)
            sent_path = os.path.join(LABEL_DIR, rec_type, out_fn)
            with open(sent_path, "w") as sent_f:
                print(sent, file=sent_f)

# TODO Consider factoring out as non-Na specific.
def prepare_untran(feat_type="fbank_and_pitch"):
    """ Preprocesses untranscribed audio."""
    org_dir = os.path.join(UNTRAN_DIR, "org")
    wav_dir = os.path.join(UNTRAN_DIR, "wav")
    feat_dir = os.path.join(UNTRAN_DIR, "feat")
    if not os.path.isdir(wav_dir):
        os.makedirs(wav_dir)
    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)

    # Standardize into wav files.
    for fn in os.listdir(org_dir):
        in_path = os.path.join(org_dir, fn)
        prefix, _ = os.path.splitext(fn)
        mono16k_wav_path = os.path.join(wav_dir, "%s.wav" % prefix)
        if not os.path.isfile(mono16k_wav_path):
            feat_extract.convert_wav(in_path, mono16k_wav_path)

    # Split up the wavs
    wav_fns = os.listdir(wav_dir)
    for fn in wav_fns:
        in_fn = os.path.join(wav_dir, fn)
        prefix, _ = os.path.splitext(fn)
        # Split into sub-wavs and perform feat extraction.
        split_id = 0
        start, end = 0, 10 #in seconds
        length = utils.wav_length(in_fn)
        while True:
            out_fn = os.path.join(feat_dir, "%s.%d.wav" % (prefix, split_id))
            utils.trim_wav(in_fn, out_fn, start, end)
            if end > length:
                break
            start += 10
            end += 10
            split_id += 1

    # Do feat extraction.
    feat_extract.from_dir(os.path.join(feat_dir), feat_type=feat_type)

# TODO Consider factoring out as non-Na specific
def prepare_feats(feat_type):
    """ Prepare the input features."""

    if not os.path.isdir(os.path.join(FEAT_DIR, "WORDLIST")):
        os.makedirs(os.path.join(FEAT_DIR, "WORDLIST"))
    if not os.path.isdir(os.path.join(FEAT_DIR, "TEXT")):
        os.makedirs(os.path.join(FEAT_DIR, "TEXT"))

    # Extract utterances from WAVS.
    trim_wavs()

    # TODO Currently assumes that the wav trimming from XML has already been
    # done.
    PREFIXES = []
    for fn in os.listdir(os.path.join(TGT_WAV_DIR, "WORDLIST")):
        if fn.endswith(".wav"):
            pre, _ = os.path.splitext(fn)
            PREFIXES.append(os.path.join("WORDLIST", pre))
    for fn in os.listdir(os.path.join(TGT_WAV_DIR, "TEXT")):
        if fn.endswith(".wav"):
            pre, _ = os.path.splitext(fn)
            PREFIXES.append(os.path.join("TEXT", pre))

    if feat_type=="phonemes_onehot":
        import numpy as np
        #prepare_labels("phonemes")
        for prefix in PREFIXES:
            label_fn = os.path.join(LABEL_DIR, "%s.phonemes" % prefix)
            out_fn = os.path.join(FEAT_DIR, "%s.phonemes_onehot" %  prefix)
            try:
                with open(label_fn) as label_f:
                    labels = label_f.readlines()[0].split()
            except FileNotFoundError:
                continue
            indices = [PHONEMES_TO_INDICES[label] for label in labels]
            one_hots = one_hots = [[0]*len(PHONEMES) for _ in labels]
            for i, index in enumerate(indices):
                one_hots[i][index] = 1
                one_hots = np.array(one_hots)
                np.save(out_fn, one_hots)
    else:
        # Otherwise, 
        for prefix in PREFIXES:
            # Convert the wave to 16k mono.
            wav_fn = os.path.join(TGT_WAV_DIR, "%s.wav" % prefix)
            mono16k_wav_fn = os.path.join(FEAT_DIR, "%s.wav" % prefix)
            if not os.path.isfile(mono16k_wav_fn):
                feat_extract.convert_wav(wav_fn, mono16k_wav_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(os.path.join(FEAT_DIR, "WORDLIST"), feat_type=feat_type)
        feat_extract.from_dir(os.path.join(FEAT_DIR, "TEXT"), feat_type=feat_type)

def get_story_prefixes():
    """ Gets the Na text prefixes. """
    prefixes = [prefix for prefix in os.listdir(os.path.join(LABEL_DIR, "TEXT"))
                if prefix.endswith("phonemes")]
    prefixes = [os.path.splitext(os.path.join("TEXT", prefix))[0]
                for prefix in prefixes]
    return prefixes

def make_data_splits(train_rec_type="text_and_wordlist", max_samples=1000, seed=0):
    """ Creates a file with a list of prefixes (identifiers) of utterances to
    include in the test set. Test utterances must never be wordlists. Assumes
    preprocessing of label dir has already been done."""

    test_prefix_fn = os.path.join(TGT_DIR, "test_prefixes.txt")
    valid_prefix_fn = os.path.join(TGT_DIR, "valid_prefixes.txt")
    with open(test_prefix_fn) as f:
        prefixes = f.readlines()
        test_prefixes = [("TEXT/" + prefix).strip() for prefix in prefixes]
    with open(valid_prefix_fn) as f:
        prefixes = f.readlines()
        valid_prefixes = [("TEXT/" + prefix).strip() for prefix in prefixes]

    prefixes = get_story_prefixes()
    prefixes = list(set(prefixes) - set(valid_prefixes))
    prefixes = list(set(prefixes) - set(test_prefixes))
    prefixes = utils.filter_by_size(
        FEAT_DIR, prefixes, "fbank", max_samples)

    if train_rec_type == "text":
        train_prefixes = prefixes
    else:
        wordlist_prefixes = [prefix for prefix in os.listdir(os.path.join(LABEL_DIR, "WORDLIST"))
                             if prefix.endswith("phonemes")]
        wordlist_prefixes = [os.path.splitext(os.path.join("WORDLIST", prefix))[0]
                             for prefix in wordlist_prefixes]
        wordlist_prefixes = utils.filter_by_size(
                FEAT_DIR, wordlist_prefixes, "fbank", max_samples)
        if train_rec_type == "wordlist":
            prefixes = wordlist_prefixes
        elif train_rec_type == "text_and_wordlist":
            prefixes.extend(wordlist_prefixes)
        else:
            raise Exception("train_rec_type='%s' not supported." % train_rec_type)
        train_prefixes = prefixes
    random.seed(0)
    random.shuffle(train_prefixes)

    return train_prefixes, valid_prefixes, test_prefixes

def get_stories():
    """ Returns a list of the stories in the Na corpus. """

    prefixes = get_story_prefixes()
    texts = list(set([prefix.split(".")[0].split("/")[1] for prefix in prefixes]))
    return texts

def make_story_splits(valid_story, test_story, max_samples):

    prefixes = get_story_prefixes()
    prefixes = utils.filter_by_size(
        FEAT_DIR, prefixes, "fbank", max_samples)

    train = []
    valid = []
    test = []
    for prefix in prefixes:
        if valid_story == os.path.basename(prefix).split(".")[0]:
            valid.append(prefix)
        elif test_story == os.path.basename(prefix).split(".")[0]:
            test.append(prefix)
        else:
            train.append(prefix)

    # Sort by utterance integer value
    test.sort(key=lambda x: int(x.split(".")[-1]))
    valid.sort(key=lambda x: int(x.split(".")[-1]))

    return train, valid, test

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Na corpus. """

    FEAT_DIR = FEAT_DIR
    LABEL_DIR = LABEL_DIR
    UNTRAN_FEAT_DIR = os.path.join(UNTRAN_DIR, "feat")

    def __init__(self,
                 feat_type="fbank_and_pitch",
                 label_type="phonemes_and_tones",
                 train_rec_type="text", max_samples=1000,
                 valid_story=None, test_story=None):
        super().__init__(feat_type, label_type)

        self.max_samples = max_samples
        self.train_rec_type = train_rec_type

        if label_type == "phonemes_and_tones":
            self.labels = PHONEMES.union(TONES).union(SYMBOLS_TO_PREDICT)
        elif label_type == "phonemes_and_tones_no_tgm":
            self.labels = PHONEMES.union(TONES)
        elif label_type == "phonemes":
            self.labels = PHONEMES
        elif label_type == "tones":
            self.labels = TONES.union(SYMBOLS_TO_PREDICT)
        elif label_type == "tones_notgm":
            self.labels = TONES
        else:
            raise Exception("label_type %s not implemented." % label_type)

        self.feat_type = feat_type
        self.label_type = label_type

        self.valid_story = valid_story
        self.test_story = test_story

        # TODO Make this also work with wordlists.
        if valid_story or test_story:
            if not (valid_story and test_story):
                raise Exception(
                    "We need a valid story if we specify a test story "
                    "and vice versa. This shouldn't be required but for "
                    "now it is.")

            train, valid, test = make_story_splits(valid_story, test_story,
                                                   max_samples)
        else:
            train, valid, test = make_data_splits(train_rec_type=train_rec_type,
                                                  max_samples=max_samples)
        self.train_prefixes = train
        self.valid_prefixes = valid
        self.test_prefixes = test

        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.vocab_size = len(self.labels)

    def output_story_prefixes(self):
        """ Writes the set of prefixes to a file this is useful for pretty
        printing in results.latex_output. """

        if not self.test_story:
            raise NotImplementedError(
                "I want to write the prefixes to a file"
                "called <test_story>_prefixes.txt, but there's no test_story.")

        fn = os.path.join(TGT_DIR, "%s_prefixes.txt" % self.test_story)
        with open(fn, "w") as f:
            for utter_id in self.test_prefixes:
                print(utter_id.split("/")[1], file=f)

    # TODO Use 'labels' instead of 'phonemes' here and in corpus.py
    # Also, factor out as non-Chatino-specific.
    def indices_to_phonemes(self, indices):
        return [(self.INDEX_TO_LABEL[index]) for index in indices]
    def phonemes_to_indices(self, labels):
        return [self.LABEL_TO_INDEX[label] for label in labels]

    def __repr__(self):
        return ("%s(" % self.__class__.__name__ +
                "feat_type=\"%s\",\n" % self.feat_type +
                "\tlabel_type=\"%s\",\n" % self.label_type +
                "\ttrain_rec_type=\"%s\",\n" % self.train_rec_type +
                "\tmax_samples=%s,\n" % self.max_samples +
                "\tvalid_story=%s,\n" % repr(self.valid_story) +
                "\ttest_story=%s)\n" % repr(self.test_story))

