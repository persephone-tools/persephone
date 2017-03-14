""" An interface with the Na data. """

import os
import random
import subprocess
import xml.etree.ElementTree as ET

import config
import utils

random.seed(0)

ORG_DIR = config.NA_DIR
TGT_DIR = "../data/na"
ORG_TXT_NORM_DIR = os.path.join(ORG_DIR, "txt_norm")
TGT_TXT_NORM_DIR = os.path.join(TGT_DIR, "txt_norm")

if not os.path.isdir(TGT_DIR):
    os.makedirs(TGT_DIR)

TO_REMOVE = {"|", "ǀ", "↑", "«", "»", "¨", "“", "”", "D", "F"}
WORDS_TO_REMOVE = {"CHEVRON", "audible", "qʰʰʰʰʰ", "qʰʰʰʰ", "D"}
TONES = ["˧˥", "˩˥", "˩˧", "˧˩", "˩", "˥", "˧"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g', 'w̃',
        'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'ṽ̩', 'ɻ̩', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
        't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ', 's',
        'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'ɻ̃', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'ṽ̩', 'o̥', 'ts',
        'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ', 'ĩ',
        'õ', 'dz'}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩"}
PHONES = UNI_PHNS.union(BI_PHNS).union(TRI_PHNS)
NUM_PHONES = len(PHONES)
PHONES2INDICES = {phn: index for index, phn in enumerate(PHONES)}
INDICES2PHONES = {index: phn for index, phn in enumerate(PHONES)}

def phones2indices(phones):
    """ Converts a list of phones to a list of indices. Increments the index by
    1 to avoid issues to do with dynamic padding in Tensorflow. """
    return [PHONES2INDICES[phone]+1 for phone in phones]

def indices2phones(indices):
    """ Converts integer representations of phones to human-readable characters. """

    return [(INDICES2PHONES[index-1] if index > 0 else "pad") for index in indices]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def remove_multi(a, ys):
    """ Removes instances of a from the list ys."""
    return list(filter(lambda x: x != a, ys))

def contains_forbidden_word(line):
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
            elif syl[i:i+2] in BI_PHNS:
                phonemes.append(syl[i:i+2])
                i += 2
                continue
            elif syl[i:i+1] in UNI_PHNS:
                phonemes.append(syl[i:i+1])
                i += 1
                continue
            else:
                raise Exception("Failed to segment syllable: %s" % syl)
    return phonemes

def trim_wav(in_fn, out_fn, start_time, end_time):
    """ Crops the wav file at in_fn so that the audio between start_time and
    end_time is output to out_fn.
    """

    args = [config.SOX_PATH, in_fn, out_fn, "trim", start_time, "=" + end_time]
    print(args[1:])
    subprocess.run(args)

def prepare_wavs_and_transcripts(filenames, segmentation, remove_tones=True):
    """ Trims available wavs into the sentence or utterance-level."""

    if not os.path.exists(TGT_TXT_NORM_DIR):
        os.makedirs(TGT_TXT_NORM_DIR)

    WAV_DIR = os.path.join(TGT_DIR, "wav")
    if not os.path.exists(WAV_DIR):
        os.makedirs(WAV_DIR)

    syl_inv = set()
    for fn in filenames:
        with open(os.path.join(ORG_TXT_NORM_DIR, fn)) as f:
            i = 0
            for line in f:

                # Remove lines with certain words in it.
                if contains_forbidden_word(line):
                    continue

                # Remove certain symbols from lines.
                for symbol in TO_REMOVE:
                    line = line.replace(symbol, "")
                if remove_tones:
                    for tone in TONES:
                        line = line.replace(tone, "")

                sp = line.split()
                start_time = sp[0]
                end_time = sp[1]
                #Ensure the line has utterance time markers.
                assert is_number(start_time)
                assert is_number(end_time)

                syls = sp[2:]
                syl_inv = syl_inv.union(syls)

                assert fn.endswith(".txt")
                prefix = fn.strip(".txt")
                i += 1
                if segmentation == "syllables":
                    out_fn = prefix + "." + str(i) + ".syl"
                    labels = syls
                elif segmentation == "phonemes":
                    out_fn = prefix + "." + str(i) + ".phn"
                    labels = segment_phonemes(syls)

                with open(os.path.join(TGT_TXT_NORM_DIR, out_fn), "w") as out_f:
                    out_f.write(" ".join(labels))

                in_wav_fn = os.path.join(ORG_DIR, "wav", "%s.wav" % prefix)
                out_wav_fn = os.path.join(WAV_DIR, "%s.%d.wav" % (prefix, i))
                trim_wav(in_wav_fn, out_wav_fn, start_time, end_time)

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

def feat_extract():
    feat_extract.from_dir(os.path.join(TGT_DIR, "wav"), feat_type="log_mel_filterbank")

class CorpusBatches:
    """ An interface to batches of Na audio/transcriptions."""

    def __init__(self, feat_type, seg_type):
        self.feat_type = feat_type
        self.seg_type = seg_type
        if seg_type == "phonemes":
            self.vocab_size = NUM_PHONES

    def batch_gen(self, batch_size, total_size, rand=True):
        self.batch_size = batch_size
        self.total_size = total_size
        input_dir = os.path.join(TGT_DIR, "wav")
        target_dir = os.path.join(TGT_DIR, "txt_norm")
        prefixes = [fn.strip(".wav") for fn in os.listdir(input_dir) if fn.endswith(".wav")]

        if rand:
            random.shuffle(prefixes)

        mod = total_size % batch_size
        if mod != 0:
            print("WARNING Total train_size %d not divisible by"
                    "batch_size %d. Ignoring remaining %d utterances." % (
                    total_size, batch_size, mod))
        prefixes = prefixes[:total_size-mod]

        prefix_batches = [prefixes[i:i+batch_size]
                for i in range(0, len(prefixes), batch_size)]

        for prefix_batch in prefix_batches:
            input_paths = [os.path.join(input_dir, "%s.%s.npy" % (
                    prefix, self.feat_type))
                    for prefix in prefix_batch]
            if self.seg_type == "phonemes":
                target_paths = [os.path.join(target_dir, prefix+".phn")
                        for prefix in prefix_batch]

            batch_x, batch_x_lens = utils.load_data.load_batch_x(input_paths,
                                                           flatten=True)

            batch_y = []
            for target_path in target_paths:
                with open(target_path) as phn_f:
                    phones = phn_f.readline().split()
                    indices = phones2indices(phones)
                    batch_y.append(indices)
            batch_y = utils.target_list_to_sparse_tensor(batch_y)

            yield batch_x, batch_x_lens, batch_y
