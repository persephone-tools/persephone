""" An interface with the Na data. """

import os
import random
import subprocess
from subprocess import PIPE

import numpy as np
import xml.etree.ElementTree as ET

import config
import corpus
import feat_extract
import datasets.pangloss
import utils

random.seed(0)

ORG_DIR = config.NA_DIR
# TODO eventually remove "new" when ALTA experiments are finished.
TGT_DIR = os.path.join(config.TGT_DIR, "na", "new")
#ORG_TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")
#TGT_TXT_NORM_DIR = os.path.join(TGT_DIR, "txt_norm")
ORG_XML_DIR = os.path.join(ORG_DIR, "xml")
ORG_WAV_DIR = os.path.join(ORG_DIR, "wav")
TGT_WAV_DIR = os.path.join(TGT_DIR, "wav")
FEAT_DIR = os.path.join(TGT_DIR, "feat")
LABEL_DIR = os.path.join(TGT_DIR, "label")
TRANSL_DIR = os.path.join(TGT_DIR, "transl")

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
           'ĩ', 'õ', 'dz', "ɻ̍"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩", "ɻ̩̃"}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)

PHONEMES = UNI_PHNS.union(BI_PHNS).union(TRI_PHNS)
NUM_PHONEMES = len(PHONEMES)
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
    elif label_type == "phonemes":
        phonemes = True
        tones = False
    elif label_type == "tones":
        phonemes = False
        tones = True
    else:
        raise Exception("Unrecognized label type: %s" % label_type)

    def pop_phoneme(sentence):
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
        if sentence[0] in UNI_TONES:
            if tones:
                return sentence[0], sentence[1:]
            else:
                return None, sentence[1:]
        if sentence[:2] in BI_TONES:
            if tones:
                return sentence[:2], sentence[2:]
            else:
                return None, sentence[2:]
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
                return None, sentence[sentence.find("]")+1]
        if sentence[0] in set([" ", "\t", "\n"]):
            # Return a space char so that it can be identified in word segmentation
            # processing.
            return " ", sentence[1:]
        if sentence[0] == "|" or sentence[0] == "ǀ":
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

def preprocess_from_xml(org_xml_dir, org_wav_dir,
                        tgt_label_dir, tgt_transl_dir, tgt_wav_dir,
                        label_type):
    # TODO Remove label_type from here and use the TGT_DIR/txt dir for the
    # unprocessed transcription from XML; then extract labels with
    # prepare_labels()
    """ Extracts sentence-level transcriptions, translations and wavs from the
    Na Pangloss XML and WAV files. But otherwise doesn't preprocess them."""

    if not os.path.exists(os.path.join(tgt_label_dir, "TEXT")):
        os.makedirs(os.path.join(tgt_label_dir, "TEXT"))
    if not os.path.exists(os.path.join(tgt_label_dir, "WORDLIST")):
        os.makedirs(os.path.join(tgt_label_dir, "WORDLIST"))

    import spacy
    fr_nlp = spacy.load("fr")

    for fn in os.listdir(org_xml_dir):
        print(fn)
        path = os.path.join(org_xml_dir, fn)
        prefix, _ = os.path.splitext(fn)

        rec_type, sents, times, transls = datasets.pangloss.get_sents_times_and_translations(path)
        # Write the sentence transcriptions to file
        sents = [preprocess_na(sent, label_type) for sent in sents]
        for i, sent in enumerate(sents):
            if sent.strip() == "":
                # Then there's no transcription, so ignore this.
                continue
            out_fn = "%s.%d.%s" % (prefix, i, label_type)
            sent_path = os.path.join(tgt_label_dir, rec_type, out_fn)
            with open(sent_path, "w") as sent_f:
                print(sent, file=sent_f)

        """
        # Extract the wavs given the times.
        for i, (start_time, end_time) in enumerate(times):
            if prefix.endswith("PLUSEGG"):
                in_wav_path = os.path.join(org_wav_dir, prefix.upper()[:-len("PLUSEGG")]) + ".wav"
            else:
                in_wav_path = os.path.join(org_wav_dir, prefix.upper()) + ".wav"
            headmic_path = os.path.join(org_wav_dir, prefix.upper()) + "_HEADMIC.wav"
            if os.path.isfile(headmic_path):
                in_wav_path = headmic_path

            out_wav_path = os.path.join(tgt_wav_dir, "%s.%d.wav" % (prefix, i))
            assert os.path.isfile(in_wav_path)
            utils.trim_wav(in_wav_path, out_wav_path, start_time, end_time)
        """

        """
        # Tokenize the French translations and write them to file.
        transls = [preprocess_french(transl[0], fr_nlp) for transl in transls]
        for i, transl in enumerate(transls):
            out_prefix = "%s.%d" % (prefix, i)
            transl_path = os.path.join(tgt_transl_dir, out_prefix + ".fr.txt")
            with open(transl_path, "w") as transl_f:
                print(transl, file=transl_f)
        """

def prepare_labels(label_type):
    """ Prepare the neural network output targets."""

    # TODO This is very computationally wasteful right now as all the wavs get
    # trimmed again. Better to pull out Na preprocessing from the XML
    # extraction.
    # If the XML hasn't been preprocessed.
    preprocess_from_xml(ORG_XML_DIR, ORG_WAV_DIR,
                        LABEL_DIR, TRANSL_DIR, TGT_WAV_DIR,
                        label_type)

# TODO Consider factoring out as non-Na specific
def prepare_feats(feat_type):
    """ Prepare the input features."""

    # TODO Currently assumes that the wav trimming from XML has already been
    # done.
    PREFIXES = []
    for fn in os.listdir(TGT_WAV_DIR):
        if fn.endswith(".wav"):
            pre, _ = os.path.splitext(fn)
            PREFIXES.append(pre)

    if not os.path.isdir(FEAT_DIR):
        os.makedirs(FEAT_DIR)

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
        feat_extract.from_dir(FEAT_DIR, feat_type=feat_type)


class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Na corpus. """

    # TODO Probably should be hardcoding the list of train/dev/test utterances
    # values externally? Slight changes to the list means the shuffling will
    # probably completely change the test set.
    TRAIN_VALID_TEST_RATIOS = [.92,.04,.04]
    FEAT_DIR = FEAT_DIR
    LABEL_DIR = LABEL_DIR

    def __init__(self, feat_type, label_type="phonemes_and_tones", max_samples=1000):
        super().__init__(feat_type, label_type)

        if label_type == "phonemes_and_tones":
            self.labels = PHONEMES.union(set(TONES))
        elif label_type == "phonemes":
            self.labels = PHONEMES
        elif label_type == "tones":
            self.labels = TONES
        else:
            raise Exception("label_type %s not implemented." % label_type)

        self.feat_type = feat_type
        self.label_type = label_type

        self.prefixes = [fn.strip("." + label_type)
                    for fn in os.listdir(LABEL_DIR) if fn.endswith(label_type)]

        # TODO Reintegrate transcribing untranscribed stuff.
        #untranscribed_dir = os.path.join(TGT_DIR, "untranscribed_wav")
        #self.untranscribed_prefixes = [os.path.join(
        #    untranscribed_dir, fn.strip(".wav"))
        #    for fn in os.listdir(untranscribed_dir) if fn.endswith(".wav")]

        if max_samples:
            self.prefixes = utils.sort_and_filter_by_size(
                FEAT_DIR, self.prefixes, feat_type, max_samples)

        # To ensure we always get the same train/valid/test split, but
        # to shuffle it nonetheless.
        random.seed(0)
        random.shuffle(self.prefixes)

        # Get indices of the end points of the train/valid/test parts of the
        # data.
        train_end = round(len(self.prefixes)*self.TRAIN_VALID_TEST_RATIOS[0])
        valid_end = round(len(self.prefixes)*self.TRAIN_VALID_TEST_RATIOS[0] +
                          len(self.prefixes)*self.TRAIN_VALID_TEST_RATIOS[1])

        self.train_prefixes = self.prefixes[:train_end]
        self.valid_prefixes = self.prefixes[train_end:valid_end]
        self.test_prefixes = self.prefixes[valid_end:]

        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.vocab_size = len(self.labels)

    # TODO Use 'labels' instead of 'phonemes' here and in corpus.py
    # Also, factor out as non-Chatino-specific.
    def indices_to_phonemes(self, indices):
        return [(self.INDEX_TO_LABEL[index]) for index in indices]
    def phonemes_to_indices(self, labels):
        return [self.LABEL_TO_INDEX[label] for label in labels]
