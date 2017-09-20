""" A driver script that runs experiments. """

import os
import shutil

import config
import rnn_ctc
import datasets.na
#import datasets.griko
import datasets.chatino
#import datasets.timit
#import datasets.japhug
import datasets.babel
from corpus_reader import CorpusReader

EXP_DIR = config.EXP_DIR

def get_exp_dir_num():
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0])
                for fn in os.listdir(EXP_DIR) if fn.split(".")[0].isdigit()])

def prep_exp_dir():
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory.
    """

    exp_num = get_exp_dir_num()
    exp_num = exp_num + 1
    code_dir = os.path.join(EXP_DIR, str(exp_num), "code")
    shutil.copytree(os.getcwd(), code_dir)

    return os.path.join(EXP_DIR, str(exp_num))

def multi_train():
    train("fbank_and_pitch", "phonemes")
    train("fbank_and_pitch", "phonemes_and_tones")
    train("fbank_and_pitch", "tones")

def train(feat_type, label_type):
    """ Run an experiment. """

    #feat_type = "fbank"
    #label_type = "tones"
    language = "chatino"
    num_layers = 3
    hidden_size = 250
    num_trains = [128,256,512,1024,2048]
    #num_trains = [128]

    if language == "chatino":
        corpus = datasets.chatino.Corpus(feat_type, label_type)
    elif language == "na":
        corpus = datasets.na.Corpus(feat_type, label_type)
    else:
        raise Exception("Language '%s' not supported." % language)

    exp_dirs = []
    for i in num_trains:
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()
        exp_dirs.append(exp_dir)
        corpus_reader = CorpusReader(corpus, num_train=i)
        model = rnn_ctc.Model(exp_dir, corpus_reader,
                              num_layers=num_layers,
                              hidden_size=hidden_size,
                              decoding_merge_repeated=(False if
                                                       label_type=="tones"
                                                       else True))
        model.train()

    print("language: %s" % language)
    print("feat_type: %s" % feat_type)
    print("label_type: %s" % label_type)
    print("num_layers: %d" % num_layers)
    print("hidden_size: %d" % hidden_size)
    print("Exp dirs:", exp_dirs)

def train_babel():
    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()
    corpus = datasets.babel.Corpus(["turkish"])
    corpus_reader = CorpusReader(corpus, num_train=len(corpus.get_train_fns()), batch_size=128)
    model = rnn_ctc.Model(exp_dir, corpus_reader, num_layers=3)
    model.train()


def calc_time():
    """ Calculates the total spoken time a given number of utterances
    corresponds to. """

    import numpy as np

    for i in [128,256,512,1024,2048]:
        corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                         target_type="phonemes")
        corpus_reader = CorpusReader(corpus, num_train=i)

        print(len(corpus_reader.train_fns))

        total_frames = 0
        #for feat_fn, _, _ in corpus.get_train_fns():
        for feat_fn in corpus_reader.corpus.get_test_fns()[0]:
            #print(feat_fn)
            frames = len(np.load(feat_fn))
            total_frames += frames

        total_time = ((total_frames*10)/1000)/60
        print(total_time)
        print("%0.3f minutes." % total_time)

def train_japhug():
    """ Run an experiment. """

    #for i in [128,256,512,1024, 2048]:
    for i in [800]:
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus = datasets.japhug.Corpus(feat_type="log_mel_filterbank",
                                    target_type="phn", normalize=True)
        corpus_reader = CorpusReader(corpus, num_train=i)
        model = rnn_ctc.Model(exp_dir, corpus_reader, num_layers=3)
        model.train()

def test():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "131", "model", "model_best.ckpt")
    model.eval(restore_model_path)

def produce_chatino_lattices():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.chatino.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=False)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "194", "model", "model_best.ckpt")
    model.output_lattices(corpus_reader.valid_batch(), restore_model_path)

def produce_na_lattices():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "131", "model", "model_best.ckpt")
    model.output_lattices(corpus_reader.valid_batch(), restore_model_path)

def transcribe():
    """ Applies a trained model to the untranscribed Na data for Alexis. """

    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    #print(corpus_reader.untranscribed_batch())

    # Model 155 is the first Na ASR model used to give transcriptions to
    # Alexis Michaud
    restore_model_path = os.path.join(
        EXP_DIR, "155", "model", "model_best.ckpt")

    #model.eval(restore_model_path, corpus_reader.)
    model.transcribe(restore_model_path)

def train_griko():

    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()

    corpus = datasets.griko.Corpus(feat_type="log_mel_filterbank",
                                   target_type="char")
    corpus_reader = CorpusReader(corpus, num_train=256)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    model.train()

def test_griko():
    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()

    corpus = datasets.griko.Corpus(feat_type="log_mel_filterbank",
                                   target_type="char")
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "164", "model", "model_best.ckpt")
    model.eval(restore_model_path)
