""" An interface with the Na data. """

import os
import random
import subprocess
from subprocess import PIPE

import xml.etree.ElementTree as ET
import spacy

import config
import corpus
import feat_extract
import datasets.pangloss
import utils

fr_nlp = spacy.load("fr")

random.seed(0)

ORG_DIR = config.NA_DIR
TGT_DIR = "../data/na"
ORG_TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")
TGT_TXT_NORM_DIR = os.path.join(TGT_DIR, "txt_norm")
ORG_XML_DIR = os.path.join(ORG_DIR, "xml")

if not os.path.isdir(TGT_DIR):
    os.makedirs(TGT_DIR)

TO_REMOVE = {"|", "ǀ", "↑", "«", "»", "¨", "“", "”", "D", "F"}
WORDS_TO_REMOVE = {"CHEVRON", "audible", "qʰʰʰʰʰ", "qʰʰʰʰ", "D"}
TONES = ["˧˥", "˩˥", "˩˧", "˧˩", "˩", "˥", "˧"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ',
           'ĩ', 'õ', 'dz'}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩"}
PHONES = UNI_PHNS.union(BI_PHNS).union(TRI_PHNS)
NUM_PHONES = len(PHONES)
PHONES2INDICES = {phn: index for index, phn in enumerate(PHONES)}
INDICES2PHONES = {index: phn for index, phn in enumerate(PHONES)}
PHONES_TONES = sorted(list(PHONES.union(set(TONES)))) # Sort for determinism
PHONESTONES2INDICES = {phn_tone: index for index, phn_tone in enumerate(PHONES_TONES)}
INDICES2PHONESTONES = {index: phn_tone for index, phn_tone in enumerate(PHONES_TONES)}

def phones2indices(phones, tones=False):
    """ Converts a list of phones to a list of indices. Increments the index by
    1 to avoid issues to do with dynamic padding in Tensorflow. """
    if tones:
        return [PHONESTONES2INDICES[phone]+1 for phone in phones]
    else:
        return [PHONES2INDICES[phone]+1 for phone in phones]

def indices2phones(indices, tones=False):
    """ Converts integer representations of phones to human-readable characters. """

    if tones:
        return [(INDICES2PHONESTONES[index-1] if index > 0 else "pad") for index in indices]
    else:
        return [(INDICES2PHONES[index-1] if index > 0 else "pad") for index in indices]

def is_number(string):
    """ Tests if a string is valid float. """
    try:
        float(string)
        return True
    except ValueError:
        return False

def remove_multi(to_remove, target_list):
    """ Removes instances of a from the list ys."""
    return list(filter(lambda x: x != to_remove, target_list))

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

def wav_length(fn):
    """ Returns the length of the WAV file in seconds."""

    args = [config.SOX_PATH, fn, "-n", "stat"]
    p = subprocess.Popen(
        [config.SOX_PATH, fn, "-n", "stat"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    length_line = str(p.communicate()[1]).split("\\n")[1].split()
    assert length_line[0] == "Length"
    return float(length_line[-1])

def prepare_wavs_and_transcripts(filenames, segmentation, tones):
    """ Trims available wavs into the sentence or utterance-level."""

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
        """ Given a line in a transcript, processes it and extracts the
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
        assert is_number(start_time)
        assert is_number(end_time)

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

    def process_translation(translation, remove_brackets_content=True):
        """ Takes the translation(s) of some utterance, preprocesses it and
        writes it to file.
        """

        if remove_brackets_content:
            trans = datasets.pangloss.remove_content_in_brackets(
                translation[0], "[]")
        # Not sure why I have to split and rejoin, but that fixes a Spacy token
        # error.
        trans = fr_nlp(" ".join(trans.split()[:]))
        #trans = fr_nlp(trans)
        trans = " ".join([token.lower_ for token in trans if not token.is_punct])
        return trans

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

def wordlists_and_texts_fns():
    """ Determine which transcript and WAV prefixes correspond to wordlists,
    and which to stories.
    """

    wordlists = []
    texts = []
    XML_DIR = os.path.join(ORG_DIR, "xml")
    txt_norm_files = os.listdir(ORG_TXT_NORM_DIR)
    for filename in os.listdir(XML_DIR):
        tree = ET.parse(os.path.join(XML_DIR, filename))
        root = tree.getroot()
        if "TEXT" in root.tag:
            prefix = filename.strip(".xml").upper()
            if prefix + "_HEADMIC.txt" in txt_norm_files:
                texts.append(prefix + "_HEADMIC.txt")
            elif prefix + ".txt" in txt_norm_files:
                texts.append(prefix + ".txt")
            else:
                print("Couldn't find: %s" % prefix)
        elif "WORDLIST" in root.tag:
            wordlists.append(filename.strip(".xml").upper())
        else:
            raise Exception("Unexpected type of transcription: %s" % root.tag)
    return wordlists, texts

def extract_features():
    """ Extract features from wave files in a given path. """

    feat_extract.from_dir(os.path.join(TGT_DIR, "wav"), feat_type="log_mel_filterbank")

class CorpusBatches:
    """ An interface to batches of Na audio/transcriptions."""

    input_dir = os.path.join(TGT_DIR, "wav")
    target_dir = os.path.join(TGT_DIR, "txt_norm")

    def sort_and_filter_by_size(self, prefixes, max_samples):
        """ Sorts the input files by their length and removes those with less
        than or equal to max_samples length. Returns the filename prefixes of
        those files.
        """

        prefix_lens = []
        for prefix in prefixes:
            path = os.path.join(self.input_dir, "%s.%s.npy" % (
                prefix, self.feat_type))
            _, batch_x_lens = utils.load_batch_x([path], flatten=True)
            prefix_lens.append((prefix, batch_x_lens[0]))
        prefix_lens.sort(key=lambda prefix_len: prefix_len[1])
        prefixes = [prefix for prefix, length in prefix_lens
                    if length <= max_samples]
        return prefixes

    def __init__(self, feat_type, seg_type, total_size, batch_size=None,
                 max_samples=1000, rand=True):
        self.feat_type = feat_type
        self.seg_type = seg_type
        self.rand = rand
        if seg_type == "phonemes":
            self.vocab_size = NUM_PHONES
        if batch_size == None:
            # Scale the batch based on the amount of training data.
            self.batch_size = total_size/32.0
        else:
            self.batch_size = batch_size
        self.total_size = total_size

        prefixes = [fn.strip(".wav") for fn in os.listdir(self.input_dir)
                    if fn.endswith(".wav")]
        prefixes = self.sort_and_filter_by_size(prefixes, max_samples)
        random.seed(0)
        random.shuffle(prefixes)

        train_prefixes = prefixes[:-200]
        self.valid_prefixes = prefixes[-200:]

        mod = total_size % batch_size
        if total_size > len(train_prefixes):
            raise Exception(("Num training examples requested (%d) greater " +
                             "than amount of training examples found (%d)") % (
                                 total_size, len(train_prefixes)))
        if mod != 0:
            raise Exception(("Number of training examples (%d) not divisible" +
                             " by batch_size %d.") % (total_size, batch_size))

        train_prefixes = train_prefixes[:total_size-mod]
        self.train_prefix_batches = [train_prefixes[i:i+batch_size]
                                     for i in
                                     range(0, len(train_prefixes), batch_size)]

    def valid_set(self, seed=None): # Seed currently ignored for Na set.

        input_paths = [os.path.join(self.input_dir, "%s.%s.npy" % (
            prefix, self.feat_type))
                       for prefix in self.valid_prefixes]
        if self.seg_type == "phonemes":
            target_paths = [os.path.join(self.target_dir, prefix+".phn")
                            for prefix in self.valid_prefixes]

        batch_x, batch_x_lens = utils.load_batch_x(input_paths,
                                                   flatten=True)

        batch_y = []
        for target_path in target_paths:
            with open(target_path) as phn_f:
                phones = phn_f.readline().split()
                indices = phones2indices(phones)
                batch_y.append(indices)
        batch_y = utils.target_list_to_sparse_tensor(batch_y)

        return batch_x, batch_x_lens, batch_y

    def train_batch_gen(self):

        if self.rand:
            random.shuffle(self.train_prefix_batches)

        for prefix_batch in self.train_prefix_batches:
            input_paths = [os.path.join(self.input_dir, "%s.%s.npy" % (prefix, self.feat_type))
                           for prefix in prefix_batch]
            if self.seg_type == "phonemes":
                target_paths = [os.path.join(self.target_dir, prefix+".phn")
                                for prefix in prefix_batch]

            batch_x, batch_x_lens = utils.load_batch_x(input_paths,
                                                       flatten=True)

            batch_y = []
            for target_path in target_paths:
                with open(target_path) as phn_f:
                    phones = phn_f.readline().split()
                    indices = phones2indices(phones)
                    batch_y.append(indices)
            batch_y = utils.target_list_to_sparse_tensor(batch_y)

            yield batch_x, batch_x_lens, batch_y

    def batch_per(self, dense_y, dense_decoded):
        return utils.batch_per(dense_y, dense_decoded)

    @property
    def num_feats(self):
        """ The number of features per frame in the input audio. """
        bg = self.train_batch_gen()
        batch = next(bg)
        return batch[0].shape[-1]

def get_target_prefix(prefix):
    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
    corresponding target file name."""

    bn = os.path.basename(prefix)
    return os.path.join(TGT_DIR, "txt_norm", bn)

def get_transl_prefix(prefix):
    """ Given a prefix of the form /some/path/here/wav/prefix, returns the
    corresponding target file name."""

    bn = os.path.basename(prefix)
    return os.path.join(TGT_DIR, "translations", bn)

def get_xml_fn(text_fn):
    """ Fetches the filenames of the xml file corresponding to the txt_norm
    transcription. (The formatting is slightly different, XML files are
    prefixed with a lowercase "crdo" and do not have "HEADMIC" or "TABLEMIC"
    suffixes.
    """

    pre, ext = os.path.splitext(text_fn)
    sp = pre.split("-")
    assert len(sp) == 2
    headmic = "_HEADMIC"
    if pre.endswith(headmic):
        xml_fn = "crdo-" + sp[1][:-len(headmic)] + ".xml"
    else:
        xml_fn = "crdo-" + sp[1] + ".xml"
    xml_fn = os.path.join(ORG_XML_DIR, xml_fn)

    return xml_fn

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the Na corpus. """

    TRAIN_VALID_TEST_RATIOS = [.8,.1,.1]

    def __init__(self, feat_type, target_type, tones=False, max_samples=1000):
        super().__init__(feat_type, target_type)

        self.tones = tones

        if tones:
            self.phonemes = PHONES.union(set(TONES))
        else:
            self.phonemes = PHONES

        if target_type != "phn":
            raise Exception("target_type %s not implemented." % target_type)

        input_dir = os.path.join(TGT_DIR, "wav")
        prefixes = [os.path.join(input_dir, fn.strip(".wav"))
                    for fn in os.listdir(input_dir) if fn.endswith(".wav")]
        untranscribed_dir = os.path.join(TGT_DIR, "untranscribed_wav")
        #self.untranscribed_prefixes = [os.path.join(
        #    untranscribed_dir, fn.strip(".wav"))
        #    for fn in os.listdir(untranscribed_dir) if fn.endswith(".wav")]

        if max_samples:
            prefixes = self.sort_and_filter_by_size(prefixes, max_samples)

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

        self.PHONEME_TO_INDEX = {phn: index for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.INDEX_TO_PHONEME = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.phonemes)))}
        self.vocab_size = len(self.phonemes)

    def prepare(tones, feat_type="log_mel_filterbank"):
        """ Preprocessing the Na data."""

        #texts_fns = wordlists_and_texts_fns()[1]
        #prepare_wavs_and_transcripts(texts_fns, "phonemes", tones)
        input_dir = os.path.join(TGT_DIR, "wav")
        feat_extract.from_dir(input_dir, feat_type)

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
        return indices2phones(indices, self.tones)

    def phonemes_to_indices(self, phonemes):
        return phones2indices(phonemes, self.tones)

    def get_train_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.train_prefixes]
        target_fns = ["%s.%s%s" % (get_target_prefix(prefix),
                                    "tones." if self.tones else "", self.target_type)
                    for prefix in self.train_prefixes]
        # TODO Make more general
        transl_fns = ["%s.removebracs.fr" % get_transl_prefix(prefix)
                      for prefix in self.train_prefixes]
        return feat_fns, target_fns, transl_fns

    def get_valid_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.valid_prefixes]
        target_fns = ["%s.%s%s" % (get_target_prefix(prefix),
                                 "tones." if self.tones else "", self.target_type)
                    for prefix in self.valid_prefixes]
        transl_fns = ["%s.removebracs.fr" % get_transl_prefix(prefix)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns, transl_fns

    def get_test_fns(self):
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.test_prefixes]
        target_fns = ["%s.%s%s" % (get_target_prefix(prefix),
                                 "tones." if self.tones else "", self.target_type)
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

def tones_only(sent):
    """ Returns only the Na tones present in the sentence."""
    return [tone for tone in sent if tone in TONES]

def phones_only(sent):
    """ Returns only the Na phones present in the sentence."""
    return [phone for phone in sent if phone in PHONES]

def phones_only_error_rate(hyps_path, refs_path):
    with open(hyps_path) as hyps_f:
        lines = hyps_f.readlines()
        hyps = [phones_only(line.split()) for line in lines]
    with open(refs_path) as refs_f:
        lines = refs_f.readlines()
        refs = [phones_only(line.split()) for line in lines]

    return utils.batch_per(hyps, refs)

def tones_only_error_rate(hyps_path, refs_path):
    with open(hyps_path) as hyps_f:
        lines = hyps_f.readlines()
        hyps = [tones_only(line.split()) for line in lines]
    with open(refs_path) as refs_f:
        lines = refs_f.readlines()
        refs = [tones_only(line.split()) for line in lines]

    print(utils.batch_per(hyps, refs))
