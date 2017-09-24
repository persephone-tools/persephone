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
FEAT_DIR = os.path.join(TGT_DIR, "feat")
LABEL_DIR = os.path.join(TGT_DIR, "label")

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
'D', 'a', 'ː', '#', '$']
BAD_NA_SYMBOLS = ['D', 'F', '~', '…', '=', '↑', ':']
PUNC_SYMBOLS = [',', '!', '.', ';', '?', "'", '"', '*', ':', '«', '»', '“', '”']
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ',
           'ĩ', 'õ', 'dz'}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩"}


TO_REMOVE = {"|", "ǀ", "↑", "«", "»", "¨", "“", "”", "D", "F"}
WORDS_TO_REMOVE = {"CHEVRON", "audible", "qʰʰʰʰʰ", "qʰʰʰʰ", "D"}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)
# TODO Change to "PHONEMES"?
PHONES = UNI_PHNS.union(BI_PHNS).union(TRI_PHNS)
NUM_PHONES = len(PHONES)
PHONES2INDICES = {phn: index for index, phn in enumerate(PHONES)}
INDICES2PHONES = {index: phn for index, phn in enumerate(PHONES)}
PHONES_TONES = sorted(list(PHONES.union(set(TONES)))) # Sort for determinism
PHONESTONES2INDICES = {phn_tone: index for index, phn_tone in enumerate(PHONES_TONES)}
INDICES2PHONESTONES = {index: phn_tone for index, phn_tone in enumerate(PHONES_TONES)}
TONES2INDICES = {tone: index for index, tone in enumerate(TONES)}
INDICES2TONES = {index: tone for index, tone in enumerate(TONES)}

# TODO rename this to "tokens2indices" or something similar.
#def phones2indices(tokens, target_type):
#    """ Converts a list of phones to a list of indices. Increments the index by
#    1 to avoid issues to do with dynamic padding in Tensorflow. """
#    if target_type == "phonemes_and_tones":
#        return [PHONESTONES2INDICES[token]+1 for token in tokens]
#    elif target_type == "phonemes":
#        return [PHONES2INDICES[token]+1 for token in tokens]
#    elif target_type == "tones":
#        return [TONES2INDICES[token]+1 for token in tokens]
#    else:
#        raise Exception("Target type %s not supported." % target_type)

#def indices2phones(indices, target_type):
#    """ Converts integer representations of phones to human-readable characters. """
#
#    if target_type == "phonemes_and_tones":
#        return [(INDICES2PHONESTONES[index-1] if index > 0 else "pad") for index in indices]
#    elif target_type == "phonemes":
#        return [(INDICES2PHONES[index-1] if index > 0 else "pad") for index in indices]
#    elif target_type == "tones":
#        return [(INDICES2TONES[index-1] if index > 0 else "pad") for index in indices]

def contains_forbidden_word(line):
    """ Tests if a line contains a non-Na word to remove."""
    for word in WORDS_TO_REMOVE:
        if word in line:
            return True
    return False

def segment_phonemes(syls):
    """ Segments a list of syllables into phonemes. """

    phonemes = []
    for syl in syls:
        i = 0
        while i < len(syl):
            if syl[i:i+3] in TRI_PHNS:
                phonemes.append(syl[i:i+3])
                i += 3
                continue
            elif syl[i:i+2] in BI_PHNS.union(TONES):
                phonemes.append(syl[i:i+2])
                i += 2
                continue
            elif syl[i:i+1] in UNI_PHNS.union(TONES):
                phonemes.append(syl[i:i+1])
                i += 1
                continue
            else:
                raise Exception("Failed to segment syllable: %s" % syl)
    return phonemes

def prepare_na_sent(sent):
    """ New function phasing into MAM. Processes sentences obtained from
    datasets.pangloss.get_sents_times_and_translations() so that we can bypass
    use of the text_norm directory and generalize to more Na Pangloss data. """

    def remove_symbols(line):
        """ Remove certain symbols from the line."""
        for symbol in TO_REMOVE:
            line = line.replace(symbol, " ")
        #if not tones:
        #    for tone in TONES:
        #        line = line.replace(tone, " ")
        return line

    return remove_symbols(sent)

def prepare_wavs_and_transcripts(filenames, target_type):
    """ Trims available wavs into the sentence or utterance-level."""
    # TODO To be deprecated. This functionality should be broken down into
    # smaller parts and made a part of the method of the class "Corpus".

    import spacy
    fr_nlp = spacy.load("fr")

    def remove_symbols(line):
        """ Remove certain symbols from the line."""
        for symbol in TO_REMOVE:
            line = line.replace(symbol, "")
        if not tones:
            for tone in TONES:
                line = line.replace(tone, "")
        return line

    if not os.path.exists(TGT_TXT_NORM_DIR):
        os.makedirs(TGT_TXT_NORM_DIR)

    TGT_TRANS_DIR = os.path.join(TGT_DIR, "translations")
    if not os.path.exists(TGT_TRANS_DIR):
        os.makedirs(TGT_TRANS_DIR)

    wav_dir = os.path.join(TGT_DIR, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    syl_inv = set()

    def process_utterance(line, line_id):
        """ Given a line in a txt_norm/ transcript, processes it and extracts the
        relevant segment from a WAV file.

            Returns True if the utterance is to be kept and had its
            transcription written; False otherwise.
        """

        # Remove lines with certain words in it.
        if contains_forbidden_word(line):
            return

        # Remove certain symbols from lines.
        line = remove_symbols(line)

        times = line.split()[:2]
        start_time = times[0]
        end_time = times[1]
        #Ensure the line has utterance time markers.
        assert utils.is_number(start_time)
        assert utils.is_number(end_time)

        syls = line.split()[2:]
        #syl_inv = syl_inv.union(syls)

        assert text_fn.endswith(".txt")
        prefix = text_fn.strip(".txt")

        out_fn = prefix + "." + str(line_id)
        if tones:
            out_fn += ".tones"
        if segmentation == "syllables":
            out_fn += ".syl"
            labels = syls
        elif segmentation == "phonemes":
            out_fn += ".phn"
            labels = segment_phonemes(syls)

        out_path = os.path.join(TGT_TXT_NORM_DIR, out_fn)
        with open(out_path, "w") as out_f:
            out_f.write(" ".join(labels))

        in_wav_fn = os.path.join(ORG_DIR, "wav", "%s.wav" % prefix)
        out_wav_fn = os.path.join(wav_dir, "%s.%d.wav" % (prefix, line_id))
        utils.trim_wav(in_wav_fn, out_wav_fn, start_time, end_time)

        return out_path

    lattm_fn = os.path.join(TGT_DIR, "latticetm_filelist.tones%s.txt" % (str(tones)))
    with open(lattm_fn, "w") as lattm_f:
        for text_fn in filenames:
            xml_fn = get_xml_fn(text_fn)
            translations = datasets.pangloss.get_sents_times_and_translations(xml_fn)[2]
            pre, ext = os.path.splitext(text_fn)
            with open(os.path.join(ORG_TXT_NORM_DIR, text_fn)) as f:
                lines = f.readlines()
                assert len(lines) == len(translations)
            with open(os.path.join(ORG_TXT_NORM_DIR, text_fn)) as f:
                line_id = 0
                for line in f:
                    transcript_path = process_utterance(line, line_id)
                    translation_fn = os.path.join(
                        TGT_TRANS_DIR, "%s.%d.removebracs.%s" % (pre, line_id, "fr"))
                    print(transcript_path)
                    if transcript_path:
                        print("%s\t%s" % (os.path.abspath(transcript_path), os.path.abspath(translation_fn)), file=lattm_f)
                    translation = process_translation(translations[line_id])
                    with open(translation_fn, "w") as transl_f:
                        print(translation, file=transl_f)
                    line_id += 1

#def wordlists_and_texts_fns():
#    """ Determine which transcript and WAV prefixes correspond to wordlists,
#    and which to stories.
#    """
#
#    wordlists = []
#    texts = []
#    XML_DIR = os.path.join(ORG_DIR, "xml")
#    txt_norm_files = os.listdir(ORG_TXT_NORM_DIR)
#    for filename in os.listdir(XML_DIR):
#        tree = ET.parse(os.path.join(XML_DIR, filename))
#        root = tree.getroot()
#        if "TEXT" in root.tag:
#            prefix = filename.strip(".xml").upper()
#            if prefix + "_HEADMIC.txt" in txt_norm_files:
#                texts.append(prefix + "_HEADMIC.txt")
#            elif prefix + ".txt" in txt_norm_files:
#                texts.append(prefix + ".txt")
#            else:
#                print("Couldn't find: %s" % prefix)
#        elif "WORDLIST" in root.tag:
#            wordlists.append(filename.strip(".xml").upper())
#        else:
#            raise Exception("Unexpected type of transcription: %s" % root.tag)
#    return wordlists, texts

#def get_target_prefix(prefix):
#    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
#    corresponding target file name."""
#
#    bn = os.path.basename(prefix)
#    return os.path.join(TGT_DIR, "txt_norm", bn)

#def get_transl_prefix(prefix):
#    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
#    corresponding target file name."""
#
#    bn = os.path.basename(prefix)
#    return os.path.join(TGT_DIR, "translations", bn)

#def get_xml_fn(text_fn):
#    """ Fetches the filenames of the xml file corresponding to the txt_norm
#    transcription. (The formatting is slightly different, XML files are
#    prefixed with a lowercase "crdo" and do not have "HEADMIC" or "TABLEMIC"
#    suffixes.
#    """
#
#    pre, ext = os.path.splitext(text_fn)
#    sp = pre.split("-")
#    assert len(sp) == 2
#    headmic = "_HEADMIC"
#    if pre.endswith(headmic):
#        xml_fn = "crdo-" + sp[1][:-len(headmic)] + ".xml"
#    else:
#        xml_fn = "crdo-" + sp[1] + ".xml"
#    xml_fn = os.path.join(ORG_XML_DIR, xml_fn)
#
#    return xml_fn

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
        if sentence[0] in set(["<", ">", "[", "]"]):
            # Brackets will get processed afterwards.
            return sentence[0], sentence[1:]
        if sentence[0] in set([" ", "\t", "\n"]):
            # Return a space char so that it can be identified in word segmentation
            # processing.
            return " ", sentence[1:]
        if sentence[0] == "|":
            return None, sentence[1:]
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

    print(sent)
    sent = filter_for_phonemes(sent)
    print(sent)
    #input()
    #sent = remove_pipes(sent)
    #sent = sent.replace("\n", " ")
    #sent = sent.replace("\t", " ")
    #sent = process_brackets(sent)
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
                        tgt_sent_dir, tgt_transl_dir, tgt_wav_dir,
                        label_type):
    """ Extracts sentence-level transcriptions, translations and wavs from the
    Na Pangloss XML and WAV files. But otherwise doesn't preprocess them."""

    import spacy
    fr_nlp = spacy.load("fr")

    for fn in os.listdir(org_xml_dir):
        print(fn)
        path = os.path.join(org_xml_dir, fn)
        sents, times, transls = datasets.pangloss.get_sents_times_and_translations(path)

        assert len(sents) == len(times)
        assert len(sents) == len(transls)

        prefix, _ = os.path.splitext(fn)

        # Write the transcriptions to file
        sents = [preprocess_na(sent, label_type) for sent in sents]
        for i, sent in enumerate(sents):
            out_fn = "%s.%d.%s" % (prefix, i, label_type)
            sent_path = os.path.join(tgt_sent_dir, out_fn)
            with open(sent_path, "w") as sent_f:
                print(sent, file=sent_f)

        """
        # Extract the wavs given the times.
        for i, (start_time, end_time) in enumerate(times):
            headmic_path = os.path.join(org_wav_dir, prefix.upper()) + "_HEADMIC.wav"
            in_wav_path = os.path.join(org_wav_dir, prefix.upper()) + ".wav"
            if os.path.isfile(headmic_path):
                in_wav_path = headmic_path

            out_wav_path = os.path.join(tgt_wav_dir, "%s.%d.wav" % (prefix, i))
            #utils.trim_wav(in_wav_path, out_wav_path, start_time, end_time)

        # Tokenize the French translations and write them to file.
        transls = [preprocess_french(transl[0], fr_nlp) for transl in transls]
        for i, transl in enumerate(transls):
            out_prefix = "%s.%d" % (prefix, i)
            transl_path = os.path.join(tgt_transl_dir, out_prefix + ".fr.txt")
            with open(transl_path, "w") as transl_f:
                print(transl, file=transl_f)
        """

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Na corpus. """

    TRAIN_VALID_TEST_RATIOS = [.8,.1,.1]

    def __init__(self, feat_type, target_type="phonemes_and_tones", max_samples=1000):
        super().__init__(feat_type, target_type)

        if target_type == "phonemes_and_tones":
            self.labels = PHONES.union(set(TONES))
        elif target_type == "phonemes":
            self.labels = PHONES
        elif target_type == "tones":
            self.labels = TONES
        else:
            raise Exception("target_type %s not implemented." % target_type)

        if feat_type == "phonemes_onehot":
            # We assume we are predicting tones given phonemes.
            assert target_type == "tones"

        # TODO Change self.phonemes field to self.tgt_labels, and related
        # variables names that might represent tones as well, or just tones.
        self.target_set = self.labels

        # TODO Make prefixes not include the path ../data/na/wav/. But note
        # that doing so might change what the training and test breakdown is
        # because of the shuffling... I should hardcode the selection
        # somewhere."
        input_dir = os.path.join(TGT_DIR, "wav")
        prefixes = [os.path.join(input_dir, fn.strip(".wav"))
                    for fn in os.listdir(input_dir) if fn.endswith(".wav")]
        untranscribed_dir = os.path.join(TGT_DIR, "untranscribed_wav")
        #self.untranscribed_prefixes = [os.path.join(
        #    untranscribed_dir, fn.strip(".wav"))
        #    for fn in os.listdir(untranscribed_dir) if fn.endswith(".wav")]

        #if max_samples:
        #    prefixes = self.sort_and_filter_by_size(prefixes, max_samples)

        # To ensure we always get the same train/valid/test split, but
        # to shuffle it nonetheless.
        random.seed(0)
        random.shuffle(prefixes)

        # Get indices of the end points of the train/valid/test parts of the
        # data.
        train_end = round(len(prefixes)*self.TRAIN_VALID_TEST_RATIOS[0])
        valid_end = round(len(prefixes)*self.TRAIN_VALID_TEST_RATIOS[0] +
                          len(prefixes)*self.TRAIN_VALID_TEST_RATIOS[1])

        self.train_prefixes = prefixes[:train_end]
        self.valid_prefixes = prefixes[train_end:valid_end]
        self.test_prefixes = prefixes[valid_end:]

        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.vocab_size = len(self.labels)

    @staticmethod
    def prepare(feat_type, target_type):
        """ Preprocessing the Na data."""

        def remove_symbols(line):
            """ Remove certain symbols from the line."""
            for symbol in TO_REMOVE:
                line = line.replace(symbol, "")
            return line

        def prepare_transcripts(texts_fns, target_set, target_type=target_type):

            if not os.path.exists(TGT_TXT_NORM_DIR):
                os.makedirs(TGT_TXT_NORM_DIR)

            transcript_fns = []
            for text_fn in texts_fns:
                #pre, ext = os.path.splitext(text_fn)
                with open(os.path.join(ORG_TXT_NORM_DIR, text_fn)) as f:
                    line_id = 0
                    for line in f:
                        #transcript_path = process_utterance(line, line_id)
                        # Remove lines with certain words in it.
                        if contains_forbidden_word(line):
                            line_id += 1
                            continue
                        # Remove certain symbols from lines.
                        line = remove_symbols(line)
                        # Get syllables
                        syls = line.split()[2:]
                        # Break syllables tokens into phonemes and tones
                        phones_and_tones = segment_phonemes(syls)
                        # Filter for the tokens we want (phonemes, tones or
                        # both)
                        tokens = [tok for tok in phones_and_tones if tok in target_set]

                        assert text_fn.endswith(".txt")
                        prefix = text_fn.strip(".txt")

                        out_fn = prefix + "." + str(line_id) + "." + target_type
                        out_path = os.path.join(TGT_TXT_NORM_DIR, out_fn)
                        transcript_fns.append(out_path)
                        with open(out_path, "w") as out_f:
                            out_f.write(" ".join(tokens))
                        line_id += 1

            return transcript_fns

        def prepare_phoneme_feats(texts_fns):
            """ Prepare one-hot phoneme representations as input features so
            that tones can be predicted from phonemes."""

            # Prepare the phonemes so they can be converted to one-hot vectors.
            phoneme_fns = prepare_transcripts(texts_fns, target_set=PHONES,
                                              target_type="phonemes")

            for utterance_fn in phoneme_fns:
                with open(utterance_fn) as f:
                    phonemes = f.readlines()[0].split()
                indices = [PHONES2INDICES[phoneme] for phoneme in phonemes]
                one_hots = [[0]*len(PHONES) for _ in phonemes]
                for i, index in enumerate(indices):
                    one_hots[i][index] = 1
                one_hots = np.array(one_hots)

                prefix = os.path.basename(utterance_fn)
                np.save(os.path.join(FEAT_DIR, prefix + "_onehot"), one_hots)

        texts_fns = wordlists_and_texts_fns()[1]

        if target_type == "phonemes_and_tones":
            target_set = PHONES.union(set(TONES))
        elif target_type == "phonemes":
            target_set = PHONES
        elif target_type == "tones":
            target_set = TONES
        else:
            raise Exception("target_type %s not implemented." % target_type)

        prepare_transcripts(texts_fns, target_set)

        if feat_type == "phonemes_onehot":
            # We assume we are predicting tones given phonemes.
            assert target_type == "tones"
            prepare_phoneme_feats(texts_fns)

        # TODO prepare_wavs_and_transcripts should be a method of this class.
        #prepare_wavs_and_transcripts(texts_fns, target_type)
        #input_dir = os.path.join(TGT_DIR, "wav")
        #feat_extract.from_dir(input_dir, feat_type)

        # Prepare the untranscribed WAV files.
        """
        org_untranscribed_dir = os.path.join(ORG_DIR, "untranscribed_wav")
        untranscribed_dir = os.path.join(TGT_DIR, "untranscribed_wav")
        from shutil import copyfile
        for fn in os.listdir(org_untranscribed_dir):
            if fn.endswith(".wav"):
                in_fn = os.path.join(org_untranscribed_dir, fn)
                length = wav_length(in_fn)
                t = 0.0
                trim_id = 0
                while t < length:
                    prefix = fn.split(".")[0]
                    out_fn = os.path.join(
                        untranscribed_dir, "%s.%d.wav" % (prefix, trim_id))
                    utils.trim_wav(in_fn, out_fn, t, t+10)
                    t += 10
                    trim_id += 1

        feat_extract.from_dir(os.path.join(TGT_DIR, "untranscribed_wav"), feat_type="log_mel_filterbank")
        """

    def indices_to_phonemes(self, indices):
        return indices2phones(indices, self.target_type)

    def phonemes_to_indices(self, phonemes):
        return phones2indices(phonemes, self.target_type)

    def get_train_fns(self):

        feat_fns = ["%s.%s.npy" % (os.path.join(FEAT_DIR, os.path.basename(prefix)), self.feat_type)
                    for prefix in self.train_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                    for prefix in self.train_prefixes]
        # TODO Make more general
        transl_fns = ["%s.removebracs.fr" % get_transl_prefix(prefix)
                      for prefix in self.train_prefixes]
        return feat_fns, target_fns, transl_fns

    def get_valid_fns(self):
        feat_fns = ["%s.%s.npy" % (os.path.join(FEAT_DIR, os.path.basename(prefix)), self.feat_type)
                    for prefix in self.valid_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                    for prefix in self.valid_prefixes]
        transl_fns = ["%s.removebracs.fr" % get_transl_prefix(prefix)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns, transl_fns

    def get_test_fns(self):
        feat_fns = ["%s.%s.npy" % (os.path.join(FEAT_DIR, os.path.basename(prefix)), self.feat_type)
                    for prefix in self.test_prefixes]
        target_fns = ["%s.%s" % (get_target_prefix(prefix), self.target_type)
                    for prefix in self.test_prefixes]
        transl_fns = ["%s.removebracs.fr" % get_transl_prefix(prefix)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns, transl_fns

    def get_untranscribed_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.untranscribed_prefixes]
        feat_fns = [fn for fn in feat_fns if "HOUSEBUILDING2" in fn]
        # Sort by the id of the wav slice.
        fn_id_pairs = [("".join(fn.split(".")[:-3]), int(fn.split(".")[-3])) for fn in feat_fns]
        fn_id_pairs.sort()
        feat_fns = ["..%s.%d.%s.npy" % (fn, fn_id, self.feat_type) for fn, fn_id in fn_id_pairs]

        return feat_fns
