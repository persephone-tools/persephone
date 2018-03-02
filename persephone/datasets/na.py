""" An interface with the Na data. """

from pathlib import Path
import os
import random

import pint # type: ignore

from .. import config
from .. import corpus
from ..preprocess import feat_extract
from ..preprocess import wav
from .. import utils
from ..exceptions import PersephoneException
from ..preprocess import pangloss

ureg = pint.UnitRegistry()

ORG_DIR = config.NA_DIR
TGT_DIR = os.path.join(config.TGT_DIR, "na")
ORG_XML_DIR = os.path.join(ORG_DIR, "xml/TEXT/F4/")
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
        tgm = False
    elif label_type == "tones":
        phonemes = False
        tones = True
        tgm = True
    elif label_type == "tones_notgm":
        phonemes = False
        tones = True
        tgm = False
    else:
        raise ValueError("Unrecognized label type: %s" % label_type)

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
        if sentence[:2] == "˧̩":
            return "˧", sentence[2:]
        if sentence[:2] == "˧̍":
            return "˧", sentence[2:]
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
        if sentence[0] == "|" or sentence[0] == "ǀ" or sentence[0] == "◊":
            # TODO Address extrametrical span symbol ◊ differently. For now,
            # treating it as a tone group boundary marker for consistency with
            # previous work.
            if tgm:
                return "|", sentence[1:]
            else:
                return None, sentence[1:]
        if sentence[0] in "()":
            return None, sentence[1:]
        print("***" + sentence)
        raise ValueError("Next character not recognized: " + sentence[:1])

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
        trans = pangloss.remove_content_in_brackets(trans, "[]")
    # Not sure why I have to split and rejoin, but that fixes a Spacy token
    # error.
    trans = fr_nlp(" ".join(trans.split()[:]))
    #trans = fr_nlp(trans)
    trans = " ".join([token.lower_ for token in trans if not token.is_punct])

    return trans

def trim_wavs(org_wav_dir=ORG_WAV_DIR,
              tgt_wav_dir=TGT_WAV_DIR,
              org_xml_dir=ORG_XML_DIR):
    """ Extracts sentence-level transcriptions, translations and wavs from the
    Na Pangloss XML and WAV files. But otherwise doesn't preprocess them."""

    print("Trimming wavs...")

    if not os.path.exists(os.path.join(tgt_wav_dir, "TEXT")):
        os.makedirs(os.path.join(tgt_wav_dir, "TEXT"))
    if not os.path.exists(os.path.join(tgt_wav_dir, "WORDLIST")):
        os.makedirs(os.path.join(tgt_wav_dir, "WORDLIST"))

    for fn in os.listdir(org_xml_dir):
        print(fn)
        path = os.path.join(org_xml_dir, fn)
        prefix, _ = os.path.splitext(fn)

        if os.path.isdir(path):
            continue
        if not path.endswith(".xml"):
            continue

        rec_type, _, times, _ = pangloss.get_sents_times_and_translations(path)

        # Extract the wavs given the times.
        for i, (start_time, end_time) in enumerate(times):
            if prefix.endswith("PLUSEGG"):
                in_wav_path = os.path.join(org_wav_dir, prefix.upper()[:-len("PLUSEGG")]) + ".wav"
            else:
                in_wav_path = os.path.join(org_wav_dir, prefix.upper()) + ".wav"
            headmic_path = os.path.join(org_wav_dir, prefix.upper()) + "_HEADMIC.wav"
            if os.path.isfile(headmic_path):
                in_wav_path = headmic_path

            out_wav_path = os.path.join(tgt_wav_dir, rec_type, "%s.%d.wav" % (prefix, i))
            assert os.path.isfile(in_wav_path)
            start_time = start_time * ureg.seconds
            end_time = end_time * ureg.seconds
            wav.trim_wav_ms(Path(in_wav_path), Path(out_wav_path),
                            start_time.to(ureg.milliseconds).magnitude,
                            end_time.to(ureg.milliseconds).magnitude)

def prepare_labels(label_type, org_xml_dir=ORG_XML_DIR, label_dir=LABEL_DIR):
    """ Prepare the neural network output targets."""

    if not os.path.exists(os.path.join(label_dir, "TEXT")):
        os.makedirs(os.path.join(label_dir, "TEXT"))
    if not os.path.exists(os.path.join(label_dir, "WORDLIST")):
        os.makedirs(os.path.join(label_dir, "WORDLIST"))

    for path in Path(org_xml_dir).glob("*.xml"):
        fn = path.name
        prefix, _ = os.path.splitext(fn)

        rec_type, sents, _, _ = pangloss.get_sents_times_and_translations(str(path))
        # Write the sentence transcriptions to file
        sents = [preprocess_na(sent, label_type) for sent in sents]
        for i, sent in enumerate(sents):
            if sent.strip() == "":
                # Then there's no transcription, so ignore this.
                continue
            out_fn = "%s.%d.%s" % (prefix, i, label_type)
            sent_path = os.path.join(label_dir, rec_type, out_fn)
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
            feat_extract.convert_wav(Path(in_path), Path(mono16k_wav_path))

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
            start_time = start * ureg.seconds
            end_time = end * ureg.seconds
            wav.trim_wav_ms(Path(in_fn), Path(out_fn),
                            start_time.to(ureg.milliseconds).magnitude,
                            end_time.to(ureg.milliseconds).magnitude)
            if end > length:
                break
            start += 10
            end += 10
            split_id += 1

    # Do feat extraction.
    feat_extract.from_dir(Path(os.path.join(feat_dir)), feat_type=feat_type)

# TODO Consider factoring out as non-Na specific
def prepare_feats(feat_type, org_wav_dir=ORG_WAV_DIR, feat_dir=FEAT_DIR, tgt_wav_dir=TGT_WAV_DIR,
                  org_xml_dir=ORG_XML_DIR, label_dir=LABEL_DIR):
    """ Prepare the input features."""

    if not os.path.isdir(TGT_DIR):
        os.makedirs(TGT_DIR)

    if not os.path.isdir(FEAT_DIR):
        os.makedirs(FEAT_DIR)


    if not os.path.isdir(os.path.join(feat_dir, "WORDLIST")):
        os.makedirs(os.path.join(feat_dir, "WORDLIST"))
    if not os.path.isdir(os.path.join(feat_dir, "TEXT")):
        os.makedirs(os.path.join(feat_dir, "TEXT"))

    # Extract utterances from WAVS.
    trim_wavs(org_wav_dir=org_wav_dir,
              tgt_wav_dir=tgt_wav_dir,
              org_xml_dir=org_xml_dir)

    # TODO Currently assumes that the wav trimming from XML has already been
    # done.
    prefixes = []
    for fn in os.listdir(os.path.join(tgt_wav_dir, "WORDLIST")):
        if fn.endswith(".wav"):
            pre, _ = os.path.splitext(fn)
            prefixes.append(os.path.join("WORDLIST", pre))
    for fn in os.listdir(os.path.join(tgt_wav_dir, "TEXT")):
        if fn.endswith(".wav"):
            pre, _ = os.path.splitext(fn)
            prefixes.append(os.path.join("TEXT", pre))

    if feat_type=="phonemes_onehot":
        import numpy as np
        #prepare_labels("phonemes")
        for prefix in prefixes:
            label_fn = os.path.join(label_dir, "%s.phonemes" % prefix)
            out_fn = os.path.join(feat_dir, "%s.phonemes_onehot" %  prefix)
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
        for prefix in prefixes:
            # Convert the wave to 16k mono.
            wav_fn = os.path.join(tgt_wav_dir, "%s.wav" % prefix)
            mono16k_wav_fn = os.path.join(feat_dir, "%s.wav" % prefix)
            if not os.path.isfile(mono16k_wav_fn):
                feat_extract.convert_wav(wav_fn, mono16k_wav_fn)

        # Extract features from the wavs.
        feat_extract.from_dir(Path(os.path.join(feat_dir, "WORDLIST")), feat_type=feat_type)
        feat_extract.from_dir(Path(os.path.join(feat_dir, "TEXT")), feat_type=feat_type)

def get_story_prefixes(label_type, label_dir=LABEL_DIR):
    """ Gets the Na text prefixes. """
    prefixes = [prefix for prefix in os.listdir(os.path.join(label_dir, "TEXT"))
                if prefix.endswith(".%s" % label_type)]
    prefixes = [os.path.splitext(os.path.join("TEXT", prefix))[0]
                for prefix in prefixes]
    return prefixes

def make_data_splits(label_type, train_rec_type="text_and_wordlist", max_samples=1000,
                     seed=0, tgt_dir=TGT_DIR):
    """ Creates a file with a list of prefixes (identifiers) of utterances to
    include in the test set. Test utterances must never be wordlists. Assumes
    preprocessing of label dir has already been done."""

    feat_dir = os.path.join(tgt_dir, "feat")
    test_prefix_fn=os.path.join(tgt_dir, "test_prefixes.txt")
    valid_prefix_fn=os.path.join(tgt_dir, "valid_prefixes.txt")
    with open(test_prefix_fn) as f:
        prefixes = f.readlines()
        test_prefixes = [("TEXT/" + prefix).strip() for prefix in prefixes]
    with open(valid_prefix_fn) as f:
        prefixes = f.readlines()
        valid_prefixes = [("TEXT/" + prefix).strip() for prefix in prefixes]

    label_dir = os.path.join(tgt_dir, "label")
    prefixes = get_story_prefixes(label_type, label_dir=label_dir)
    prefixes = list(set(prefixes) - set(valid_prefixes))
    prefixes = list(set(prefixes) - set(test_prefixes))
    prefixes = utils.filter_by_size(
        feat_dir, prefixes, "fbank", max_samples)

    if train_rec_type == "text":
        train_prefixes = prefixes
    else:
        wordlist_prefixes = [prefix for prefix in os.listdir(os.path.join(label_dir, "WORDLIST"))
                             if prefix.endswith("phonemes")]
        wordlist_prefixes = [os.path.splitext(os.path.join("WORDLIST", prefix))[0]
                             for prefix in wordlist_prefixes]
        wordlist_prefixes = utils.filter_by_size(
                feat_dir, wordlist_prefixes, "fbank", max_samples)
        if train_rec_type == "wordlist":
            prefixes = wordlist_prefixes
        elif train_rec_type == "text_and_wordlist":
            prefixes.extend(wordlist_prefixes)
        else:
            raise PersephoneException("train_rec_type='%s' not supported." % train_rec_type)
        train_prefixes = prefixes
    random.seed(seed)
    random.shuffle(train_prefixes)

    return train_prefixes, valid_prefixes, test_prefixes

def get_stories(label_type):
    """ Returns a list of the stories in the Na corpus. """

    prefixes = get_story_prefixes(label_type)
    texts = list(set([prefix.split(".")[0].split("/")[1] for prefix in prefixes]))
    return texts

def make_story_splits(valid_story, test_story, max_samples, label_type, tgt_dir=TGT_DIR):

    feat_dir=os.path.join(tgt_dir, "feat/")

    prefixes = get_story_prefixes(label_type)
    # TODO Remove assumption of fbank features
    prefixes = utils.filter_by_size(
        feat_dir, prefixes, "fbank", max_samples)

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

class Corpus(corpus.Corpus):
    """ Class to interface with the Na corpus. """

    def __init__(self,
                 feat_type="fbank_and_pitch",
                 label_type="phonemes_and_tones",
                 train_rec_type="text", max_samples=1000,
                 valid_story=None, test_story=None,
                 tgt_dir=Path(TGT_DIR)):

        self.tgt_dir = tgt_dir
        self.get_wav_dir().mkdir(parents=True, exist_ok=True) # pylint: disable=no-member
        self.get_label_dir().mkdir(exist_ok=True) # pylint: disable=no-member

        self.valid_story = valid_story
        self.test_story = test_story

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
            raise PersephoneException("label_type %s not implemented." % label_type)

        tgt_label_dir = str(tgt_dir / "label")
        tgt_wav_dir = str(tgt_dir / "wav")
        tgt_feat_dir = str(tgt_dir / "feat")
        prepare_labels(label_type, label_dir=tgt_label_dir)
        prepare_feats(feat_type, tgt_wav_dir=tgt_wav_dir,
                                 feat_dir=tgt_feat_dir,
                                 label_dir=tgt_label_dir)

        super().__init__(feat_type, label_type, tgt_dir, self.labels, max_samples=max_samples)

    def make_data_splits(self, max_samples, valid_story=None, test_story=None):
        # TODO Make this also work with wordlists.
        if valid_story or test_story:
            if not (valid_story and test_story):
                raise PersephoneException(
                    "We need a valid story if we specify a test story "
                    "and vice versa. This shouldn't be required but for "
                    "now it is.")

            train, valid, test = make_story_splits(valid_story, test_story,
                                                   max_samples,
                                                   self.label_type,
                                                   tgt_dir=str(self.tgt_dir))
        else:
            train, valid, test = make_data_splits(self.label_type,
                                                  train_rec_type=self.train_rec_type,
                                                  max_samples=max_samples,
                                                  tgt_dir=str(self.tgt_dir))
        self.train_prefixes = train
        self.valid_prefixes = valid
        self.test_prefixes = test

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

    def __repr__(self):
        return ("%s(" % self.__class__.__name__ +
                "feat_type=\"%s\",\n" % self.feat_type +
                "\tlabel_type=\"%s\",\n" % self.label_type +
                "\ttrain_rec_type=\"%s\",\n" % self.train_rec_type +
                "\tmax_samples=%s,\n" % self.max_samples +
                "\tvalid_story=%s,\n" % repr(self.valid_story) +
                "\ttest_story=%s)\n" % repr(self.test_story))

