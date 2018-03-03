from pathlib import Path

import pytest

from persephone import utils
from persephone.run import prep_exp_dir
import persephone.datasets.bkw as bkw
from persephone.corpus_reader import CorpusReader
from persephone import rnn_ctc
from persephone import config

@pytest.mark.experiment
class TestBKWExperiment:

    @pytest.fixture
    def clean_git(self):
        utils.is_git_directory_clean(".")

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
