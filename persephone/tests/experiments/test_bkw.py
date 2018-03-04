""" Testing Persephone on Alex/Steven's Kunwinjku data. """

import logging
import os
from os.path import splitext
from pathlib import Path
import pprint
import subprocess
from typing import List

import pint
import pytest
from pympi.Elan import Eaf

from persephone import config
from persephone import corpus
from persephone import utterance
from persephone.utterance import Utterance
from persephone.datasets import bkw
from persephone.preprocess import elan
from persephone.corpus_reader import CorpusReader
from persephone.run import prep_exp_dir
from persephone import rnn_ctc
from persephone import utils

ureg = pint.UnitRegistry()

logging.config.fileConfig(config.LOGGING_INI_PATH)

@pytest.mark.experiment
class TestBKWExperiment:

    @pytest.fixture
    def clean_git(self):
        utils.is_git_directory_clean(".")

    def test_tf_gpu(self):
        import tensorflow as tf
        # Creates a graph.
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # Runs the op.
        print(sess.run(c))

    @staticmethod
    def train_bkw(num_layers: int) -> None:
        exp_dir = prep_exp_dir(directory=config.TEST_EXP_PATH)
        corp = bkw.create_corpus(tgt_dir=Path(config.TEST_DATA_PATH) / "bkw")
        cr = CorpusReader(corp)
        model = rnn_ctc.Model(exp_dir, cr, num_layers=num_layers, hidden_size=250)
        model.train(min_epochs=40)

    def test_bkw_2_layers(self, clean_git):
        """ Trains a multispeaker BKW system using default settings. """
        self.train_bkw(num_layers=2)

    def test_bkw_3_layers(self, clean_git):
        """ Trains a multispeaker BKW system using default settings. """
        self.train_bkw(num_layers=3)

@pytest.mark.experiment
class TestBKW:

    tgt_dir = Path(config.TEST_DATA_PATH) / "bkw"
    en_words_path = Path(config.EN_WORDS_PATH)
    NUM_UTTERS = 1004 # Or 1006?
    NUM_SPEAKERS = 19

    @pytest.fixture(scope="class")
    def prep_org_data(self):
        """ Ensure the un-preprocessed data is available. """

        # Ensure the BKW data is all there
        bkw_path = Path(config.BKW_PATH)
        if not bkw_path.is_dir():
            raise NotImplementedError(
                "Data isn't available in {} and I haven't figured out how authentication".format(bkw_path) +
                " should best work for datasets that aren't yet public.")
        assert bkw_path.is_dir()

        # Ensure english-words/words.txt is there.
        assert self.en_words_path.is_file()

        return bkw_path

    @pytest.fixture
    def clean_tgt_dir(self):
        """ Clears the target testing directory. """

        if self.tgt_dir.is_dir():
            import shutil
            shutil.rmtree(str(self.tgt_dir))

        assert not self.tgt_dir.is_dir()

    @pytest.fixture
    def preprocessed_corpus(self, prep_org_data):
        """ Ensure's corpus preprocessing happens before any of the tests
        run that rely on it"""
        return bkw.create_corpus(tgt_dir=self.tgt_dir)

    def check_corpus(self, corp):

        assert len(corp.utterances) == self.NUM_UTTERS

        # Below tests might not work since filtering of utterances by size 
        #assert len(corp.get_train_fns()[0] +
        #           corp.get_valid_fns()[0] +
        #           corp.get_test_fns()[0]) == self.NUM_UTTERS
        #assert len(corp.determine_prefixes()) == self.NUM_UTTERS
        #assert (self.tgt_dir / "wav").is_dir()
        #assert len(list(corp.wav_dir.iterdir())) == self.NUM_UTTERS

    @pytest.mark.slow
    def test_bkw_preprocess(self, prep_org_data, clean_tgt_dir, preprocessed_corpus):
        self.check_corpus(preprocessed_corpus)

    def test_bkw_after_preprocessing(self, preprocessed_corpus):
        self.check_corpus(preprocessed_corpus)

    @staticmethod
    def count_empty(utterances: List[Utterance]) -> int:
        empty_count = 0
        for utter in utterances:
            if utter.text.strip() == "":
                empty_count += 1
        return empty_count

    def test_utterances_from_dir(self, prep_org_data):
        bkw_org_path = prep_org_data

        utterances = elan.utterances_from_dir(bkw_org_path, ["xv"])
        assert len(utterances) == 1036
        assert len(utterance.remove_empty_text(utterances)) == 1035
        assert len(utterance.remove_duplicates(utterances)) == 1029
        assert len(utterance.remove_duplicates(
                                   utterance.remove_empty_text(utterances))) == 1028

        utterances = elan.utterances_from_dir(bkw_org_path, ["rf"])
        assert len(utterances) == 1242
        assert len(utterance.remove_empty_text(utterances)) == 631
        assert len(utterance.remove_duplicates(utterances)) == 1239
        assert len(utterance.remove_duplicates(
                                   utterance.remove_empty_text(utterances))) == 631

        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        assert len(utterances) == 2278
        assert len(utterance.remove_empty_text(utterances)) == 1666
        assert len(utterance.remove_duplicates(utterances)) == 1899
        assert len(utterance.remove_duplicates(
                                   utterance.remove_empty_text(utterances))) == 1291

    @staticmethod
    def check_text_in_utters(text: str, utters: List[Utterance]) -> bool:
        """ Checks that the target text is found in utterances. """
        for utter in utters:
            if utter.text == text:
                return True
        return False

    def test_mark_on_rock_rf_xv_duplicate(self, prep_org_data):
        mark_on_rock_path = prep_org_data / "Mark on Rock.eaf"
        anbuyika_text = (" Anbuyika rudno karudyo mani arriwa::::m"
                         " arribebmeng Madjinbardi")

        xv_utters = elan.utterances_from_eaf(mark_on_rock_path, ["xv"])
        rf_utters = elan.utterances_from_eaf(mark_on_rock_path, ["rf"])
        xv_rf_utters = elan.utterances_from_eaf(mark_on_rock_path, ["xv", "rf"])

        assert self.check_text_in_utters(anbuyika_text, xv_utters)
        assert self.check_text_in_utters(anbuyika_text, rf_utters)
        assert self.check_text_in_utters(anbuyika_text, xv_rf_utters)
        assert not self.check_text_in_utters("some random text", xv_rf_utters)

        assert len(xv_utters) == 425
        assert len(rf_utters) == 420
        assert len(xv_rf_utters) == 845
        assert len(utterance.remove_duplicates(xv_rf_utters)) == 476
        assert len(utterance.remove_empty_text(
                   utterance.remove_duplicates(xv_rf_utters))) == 473

    def test_corpus_duration(self, preprocessed_corpus):
        corp = preprocessed_corpus
        cr = CorpusReader(corp, batch_size=1)
        cr.calc_time()
        print("Number of corpus utterances: {}".format(len(corp.get_train_fns()[0])))

    def test_explore_code_switching(self, prep_org_data):
        bkw_org_path = prep_org_data
        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        utterances = utterance.remove_empty_text(
                     utterance.remove_duplicates(utterances))
        codeswitched_path = self.tgt_dir / "codeswitched.txt"
        bkw.explore_code_switching(utterances, codeswitched_path)

    def test_speaker_id(self, prep_org_data):
        bkw_org_path = prep_org_data
        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        no_speaker_tiers = set()
        speaker_tiers = set()
        speakers = set()
        for utter in utterances:
            tier_id = splitext(utter.prefix)[0]
            if utter.speaker == None:
                no_speaker_tiers.add(tier_id)
            else:
                speaker_tiers.add((tier_id, utter.speaker))
                speakers.add(utter.speaker)

        assert len(no_speaker_tiers) == 0
        assert len(speakers) == self.NUM_SPEAKERS

    def test_overlapping_utters(self, prep_org_data):
        tier1 = "rf"
        tier2 = "rf@MN"
        eaf_path = prep_org_data / "Marys_Yirlinkirrkirr.eaf"
        eaf = Eaf(str(eaf_path))
        #import pprint
        #pprint.pprint(list(eaf.get_gaps_and_overlaps(tier1, tier2)))

    def test_speaker_durations(self, prep_org_data):
        bkw_org_path = prep_org_data
        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        print(len(utterances))
        utterances = utterance.remove_empty_text(utterances)
        print(len(utterances))
        utterances = utterance.remove_duplicates(utterances)
        print(len(utterances))
        utterances = [utter for utter in utterances if bkw.bkw_filter(utter)]
        print(len(utterances))
        utterances = [utter for utter in utterances if utterance.duration(utter) < 10000]
        total = 0
        for speaker, duration in utterance.speaker_durations(utterances):
            dur_mins = (duration * ureg.milliseconds).to(ureg.minutes)
            total += dur_mins
            print("Speaker: {}\nDuration: {:0.2f}".format(speaker, dur_mins))
            print()
        print("Total duration: {:0.2f}".format(total))

    @pytest.mark.skip
    def test_poly_durations(self, prep_org_data):
        bkw_org_path = prep_org_data
        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        print("Total duration of utterances is {}".format(
            utterance.duration(utterances)))
        print("Total duration of the first utterance is {}".format(
            utterance.duration(utterances[0])))

    def test_train_data_isnt_test_data(self, preprocessed_corpus):

        corp = preprocessed_corpus

        # Assert test fns are distinct from train fns.
        train = set(corp.get_train_fns()[0])
        valid = set(corp.get_valid_fns()[0])
        test = set(corp.get_test_fns()[0])
        print(len(train))
        print(len(valid))
        print(len(test))
        assert train - valid == train
        assert train - test == train
        assert valid - train == valid
        assert valid - test == valid
        assert test - train == test
        assert test - valid == test

        # First assert that test corpus utterances aren't in the training set
        # by loading them.
        #train = []
        #for fn in corp.get_train_fns():
        #    with open(fn) as f:
        #        train.append(read().strip())
        #valid = []
        #for fn in corp.get_valid_fns():
        #    with open(fn) as f:
        #        valid.append(read().strip())
        #test = []
        #for fn in corp.get_test_fns():
        #    with open(fn) as f:
        #        test.append(read().strip())
        #validtest = valid + test
        #print(train)
        #print(validtest)

        # Could try this at the corpus_reader level, though I need to figure
        # out how that code works again.
        #cr = CorpusReader(corp)
        #for batch in cr.train_batch_gen():
        #    print(batch)
        #    print(cr.human_readable(batch))

        # Then do the more important test of checking for duplicates again. For
        # each utterance in the test set, look for the most similar one
        # edit-distance-wise from the training set. Do the same for the
        # validation set.

        # Do a code review to ensure I'm doing nothing silly.

        # Run another model for unbounded epochs to see if training error
        # diverges from test error.

    @pytest.mark.slow
    def test_multispeaker(self, preprocessed_corpus):
        """ Trains a multispeaker BKW system using default settings. """

        exp_dir = prep_exp_dir(directory=config.TEST_EXP_PATH)
        # TODO bkw.Corpus and elan.Corpus should take an org_dir argument.
        corp = preprocessed_corpus
        cr = CorpusReader(corp)
        model = rnn_ctc.Model(exp_dir, cr, num_layers=2, hidden_size=250)
        model.train(min_epochs=30)

    @pytest.mark.skip
    def test_utt2spk(self, prep_org_data):
        corp = bkw.create_corpus(tgt_dir=self.tgt_dir, speakers=["Mark Djandiomerr"])
        assert len(corp.speakers) == 1
        assert len(corp.get_train_fns()) < self.NUM_UTTERS / 2
        corp = bkw.create_corpus(tgt_dir=self.tgt_dir)
        assert len(corp.speakers) == self.NUM_SPEAKERS
        assert len(corp.get_train_fns()) == self.NUM_UTTERS

    def test_deterministic(self, prep_org_data):
        """ Ensures loading and processing utterences from ELAN files is
        deterministic.
        """
        bkw_org_path = prep_org_data
        utterances_1 = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        utterances_2 = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])
        assert utterances_1 == utterances_2
        utterances_1 = [utter for utter in utterances_1 if bkw.bkw_filter(utter)]
        utterances_2 = [utter for utter in utterances_2 if bkw.bkw_filter(utter)]
        assert utterances_1 == utterances_2
        utterances_1 = utterance.remove_duplicates(utterances_1)
        utterances_2 = utterance.remove_duplicates(utterances_2)
        assert utterances_1 == utterances_2
        utterances_1 = [bkw.bkw_label_segmenter.segment_labels(utter) for utter in utterances_1]
        utterances_2 = [bkw.bkw_label_segmenter.segment_labels(utter) for utter in utterances_2]
        assert utterances_1 == utterances_2
        utterances_1 = utterance.remove_empty_text(utterances_1)
        utterances_2 = utterance.remove_empty_text(utterances_2)
        assert utterances_1 == utterances_2

    def test_deterministic_2(self, prep_org_data):
        corp_1 = bkw.create_corpus(tgt_dir=self.tgt_dir)
        # Remove the prefix files.
        os.remove(str(corp_1.train_prefix_fn))
        os.remove(str(corp_1.valid_prefix_fn))
        os.remove(str(corp_1.test_prefix_fn))
        corp_2 = bkw.create_corpus(tgt_dir=self.tgt_dir)
        assert corp_1.utterances != None
        assert corp_1.utterances == corp_2.utterances
        assert len(corp_1.utterances) == self.NUM_UTTERS
        assert set(corp_1.get_train_fns()[0]) == set(corp_2.get_train_fns()[0])
        assert set(corp_1.get_valid_fns()[0]) == set(corp_2.get_valid_fns()[0])
        assert set(corp_1.get_test_fns()[0]) == set(corp_2.get_test_fns()[0])

    def test_empty_wav(self, prep_org_data):
        # Checking the origin of the empty wav.

        bkw_org_path = prep_org_data
        utterances = elan.utterances_from_dir(bkw_org_path, ["rf", "xv"])

        filtered = utterance.remove_too_short(utterances)
        if filtered != utterances:
            diff = set(utterances) - set(filtered)
            print("set(utterances) - set(filtered): {}:\n".format(
                pprint.pformat(diff)))
            assert False

    # TODO This sort of test, and others don't really rely on the BKW data
    # specifically so could be tested elsewhere (in Travis!)
    def test_pickle_corpus(self, preprocessed_corpus):
        corp = preprocessed_corpus
        corp.pickle()
        retrieved_corp = corpus.Corpus.from_pickle(corp.tgt_dir)
        assert corp.utterances == retrieved_corp.utterances
        print(len(retrieved_corp.utterances))
