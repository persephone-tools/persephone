""" Testing Persephone on Alex/Steven's Kunwinjku data. """

from pathlib import Path
import subprocess

import pytest

from persephone import config
from persephone.datasets import bkw

@pytest.mark.notravis
class TestBKW:

    tgt_dir = Path(config.TEST_TGT_DATA_ROOT) / "bkw"
    NUM_UTTERS = 1250

    @pytest.fixture(scope="class")
    def prep_org_data(self):
        """ Ensure the un-preprocessed data is available. """

        # Ensure the BKW data is all there
        bkw_path = Path(config.BKW_PATH)
        if not bkw_path.is_dir():
            raise NotImplementedError(
                "Data isn't available and I haven't figured out how authentication"
                " should best work for datasets that aren't yet public.")
        assert bkw_path.is_dir()

        # Ensure english-words/words.txt is there.
        assert en_words_path.is_file()

    @pytest.fixture
    def clean_tgt_dir(self):
        """ Clears the target testing directory. """

        if self.tgt_dir.is_dir():
            import shutil
            shutil.rmtree(str(self.tgt_dir))

        assert not self.tgt_dir.is_dir()

    @pytest.mark.slow
    def test_bkw_preprocess(self, prep_org_data, clean_tgt_dir):

        corp = bkw.Corpus(tgt_dir=self.tgt_dir)

        assert len(corp.determine_prefixes()) == self.NUM_UTTERS
        assert (tgt_dir / "wav").is_dir()
        assert len(list(corp.wav_dir.iterdir())) == self.NUM_UTTERS

    def test_bkw_after_preprocessing(self):

        corp = bkw.Corpus(tgt_dir=self.tgt_dir)
        assert len(corp.determine_prefixes()) == self.NUM_UTTERS
