""" Serves as an interface to the TIMIT data. """

import os
import random

from nltk.metrics import distance

from .. import corpus
from .. import utils
from .. import config

random.seed(0)

# Hardcoded numbers
NUM_LABELS = 61
# The number of training sentences with SA utterances removed.
TOTAL_SIZE = 3696

TIMIT_TGT_DIR = os.path.join(config.TGT_DIR, "timit")

def phone_classes(path=os.path.join(TIMIT_TGT_DIR, "train"),
                  feat_type="mfcc13_d"):
    """ Returns a sorted list of phone classes observed in the TIMIT corpus."""

    train_paths = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(feat_type + ".npy"):
                # Add to filename list.
                path = os.path.join(root, filename)
                train_paths.append(os.path.join(root, filename))
    phn_paths = [path.split(".")[0] + ".phn" for path in train_paths]
    phn_set = set()
    for phn_path in phn_paths:
        with open(phn_path) as phn_f:
            for phone in phn_f.readline().split():
                phn_set.add(phone)

    assert len(phn_set) == NUM_LABELS
    return sorted(list(phn_set))

PHONE_SET = phone_classes()
INDEX_TO_PHONE_MAP = {index: phone for index, phone in enumerate(PHONE_SET)}
PHONE_TO_INDEX_MAP = {phone: index for index, phone in enumerate(PHONE_SET)}

def phones2indices(phones):
    """ Converts a list of phones to a list of indices. Increments the index by
    1 to avoid issues to do with dynamic padding in Tensorflow. """
    return [PHONE_TO_INDEX_MAP[phone]+1 for phone in phones]

def indices2phones(indices):
    """ Converts integer representations of phones to human-readable characters. """

    return [(INDEX_TO_PHONE_MAP[index-1] if index > 0 else "pad") for index in indices]


def collapse_phones(utterance):
    """ Converts an utterance with labels of 61 possible phones to 39. This is
    done as per Kai-fu Lee & Hsiao-Wuen Hon 1989."""

    # Define some groupings of similar phones
    allophone_map = {"ux":"uw", "axr":"er", "ax-h":"ah", "em":"m", "nx":"n",
                     "hv": "hh", "eng": "ng",
                     "q":"t", # Glottal stop -> t
                     "pau":"sil", "h#": "sil", "#h": "sil", # Silences
                     "bcl":"vcl", "dcl":"vcl", "gcl":"vcl", # Voiced closures
                     "pcl":"cl", "tcl":"cl", "kcl":"cl", "qcl":"cl", # Unvoiced closures
                    }

    class_map = {"el": "l", "en": "n", "zh": "sh", "ao": "aa", "ix":"ih",
                 "ax":"ah", "sil":"sil", "cl":"sil", "vcl":"sil", "epi":"sil"}

    allo_collapse = [(allophone_map[phn] if phn in allophone_map else phn) for phn in utterance]
    class_collapse = [(class_map[phn] if phn in class_map else phn) for phn in allo_collapse]

    return class_collapse
    #return allo_collapse

CORE_SPEAKERS = ["dr1/mdab0", "dr1/mwbt0", "dr1/felc0",
                 "dr2/mtas1", "dr2/mwew0", "dr2/fpas0",
                 "dr3/mjmp0", "dr3/mlnt0", "dr3/fpkt0",
                 "dr4/mlll0", "dr4/mtls0", "dr4/fjlm0",
                 "dr5/mbpm0", "dr5/mklt0", "dr5/fnlp0",
                 "dr6/mcmj0", "dr6/mjdh0", "dr6/fmgd0",
                 "dr7/mgrt0", "dr7/mnjm0", "dr7/fdhc0",
                 "dr8/mjln0", "dr8/mpam0", "dr8/fmld0"]

def load_batch_y(path_batch):
    """ Loads the target gold labelling for a given batch."""

    batch_y = []
    phn_paths = [path.split(".")[0]+".phn" for path in path_batch]
    for phn_path in phn_paths:
        with open(phn_path) as phn_f:
            phone_indices = phones2indices(phn_f.readline().split())
            batch_y.append(phone_indices)
    return batch_y

def test_set(feat_type, path=os.path.join(TIMIT_TGT_DIR, "test"), flatten=True):
    """ Retrieves the core test set of 24 speakers. """

    test_paths = []
    for speaker in CORE_SPEAKERS:
        speaker_path = os.path.join(path, speaker)
        fns = os.listdir(speaker_path)
        for filename in fns:
            if filename.endswith(feat_type + ".npy") and not filename.startswith("sa"):
                test_paths.append(os.path.join(speaker_path, filename))
    batch_x, utter_lens = utils.load_batch_x(test_paths, flatten=flatten)
    batch_y = load_batch_y(test_paths)

    return batch_x, utter_lens, batch_y

def phoneme_error_rate(batch_y, decoded):
    """ Calculates the phoneme error rate between decoder output and the gold
    reference by first collapsing the TIMIT labels into the standard 39
    phonemes."""

    # Use an intermediate human-readable form for debugging. Perhaps can be
    # moved into a separate function down the road.
    ref = batch_y[1]
    phn_ref = collapse_phones(indices2phones(ref))
    phn_hyp = collapse_phones(indices2phones(decoded[0].values))
    return distance.edit_distance(phn_ref, phn_hyp)/len(phn_ref)

def get_valid_fns(feat_type, target_type):
    """ Retrieves a 50 speaker validation set. """

    chosen_paths = []
    for dialect in ["dr1", "dr2", "dr3", "dr4", "dr5", "dr6", "dr7", "dr8"]:
        dr_path = os.path.join(TIMIT_TGT_DIR, "test", dialect)
        all_test_speakers = [os.path.join(dr_path, speaker) for speaker in os.listdir(dr_path)]
        valid_speakers = [path for path in all_test_speakers if not
                          path.split("test/")[-1] in CORE_SPEAKERS]
        male_valid_speakers = [path for path in valid_speakers
                               if path.split("/")[-1].startswith("m")]
        female_valid_speakers = [path for path in valid_speakers
                                 if path.split("/")[-1].startswith("f")]
        # Select the first two male speakers and first female speaker
        chosen_paths.extend(male_valid_speakers[:2])
        chosen_paths.extend(female_valid_speakers[:1])

    valid_input_paths = []
    valid_target_paths = []
    for speaker_path in chosen_paths:
        fns = os.listdir(speaker_path)
        for filename in fns:
            if filename.endswith(feat_type + ".npy") and not filename.startswith("sa"):
                valid_input_paths.append(os.path.join(speaker_path, filename))
                target_fn = "%s.%s" % (filename.split(".")[0], target_type)
                assert target_fn in fns
                valid_target_paths.append(os.path.join(speaker_path, target_fn))

    return valid_input_paths, valid_target_paths

class Corpus(corpus.AbstractCorpus):
    """ Class to interface with the TIMIT corpus."""

    vocab_size = NUM_LABELS

    def __init__(self, feat_type, target_type):
        super().__init__(feat_type, target_type)
        if target_type != "phn":
            raise Exception("target_type %s not implemented." % target_type)

    def prepare(self):
        """ Preprocesses the TIMIT data. """

        raise Exception("""Not implemented. Refactor preprocess_timit.py into
                        this module""")

        #tgt_dir = os.path.join(
        #    timit_dir, "feat_type=%s-target_type=%s" % (self.feat_type, self.target_type))
        #if os.isdir(tgt_dir):
        #    # Then this preprocessing has already been done.
        #    return tgt_dir

        #prepare_train_set(tgt_dir, self.feat_type, self.target_type)
        #prepare_valid_set(tgt_dir, self.feat_type, self.target_type)
        #prepare_test_set(tgt_dir, feat_type, target_type)

    @staticmethod
    def indices_to_phonemes(indices):
        return collapse_phones(indices2phones(indices))

    @staticmethod
    def phonemes_to_indices(phonemes):
        return phones2indices(phonemes)

    def get_train_fns(self):
        train_path = os.path.join(TIMIT_TGT_DIR, "train")

        prefixes = utils.get_prefixes(
            train_path, extension=(".%s.npy" % self.feat_type))
        prefixes = [prefix for prefix in prefixes
                    if not os.path.basename(prefix).startswith("sa")]
        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in prefixes]
        target_fns = ["%s.%s" % (prefix, self.target_type)
                      for prefix in prefixes]

        return feat_fns, target_fns

    def get_valid_fns(self):
        valid_fns = get_valid_fns(self.feat_type, self.target_type)
#        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
#                    for prefix in prefixes]
#        target_fns = ["%s.%s" % (prefix, self.target_type)
#                      for prefix in prefixes]
        feat_fns, target_fns = valid_fns

        return feat_fns, target_fns

    def get_test_fns(self):
        raise Exception("Not implemented.")
