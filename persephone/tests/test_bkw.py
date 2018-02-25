""" Testing Persephone on Alex/Steven's Kunwinjku data. """

from pathlib import Path
import subprocess

import pytest

from persephone import config

@pytest.mark.notravis
class TestBKW:

    @pytest.fixture(scope="class")
    def prepare_bkw_data(self):
        # Ensure the BKW data is all there
        bkw_path = Path(config.BKW_PATH)
        if not bkw_path.is_dir():
            raise NotImplementedError(
                "Data isn't available and I haven't figured out how authentication"
                " should best work for datasets that aren't yet public.")
        assert bkw_path.is_dir()

        # Ensure english-words/words.txt is there.
        ENGLISH_WORDS_URL = "https://github.com/dwyl/english-words.git"
        en_words_path = Path(config.EN_WORDS_PATH)
        if not en_words_path.is_file():
            subprocess.run(["git", "clone",
                            ENGLISH_WORDS_URL, str(en_words_path.parent)])
        assert en_words_path.is_file()

        return en_words_path


    def test_en_words(self, prepare_bkw_data):
        with prepare_bkw_data.open() as f:
            print(f.read())
