import os
from os.path import join
import pytest
import subprocess

from persephone import corpus
from persephone import run
from persephone.context_manager import cd

EXP_BASE_DIR = "testing/exp/"
DATA_BASE_DIR = "testing/data/"

# TODO This needs to be uniform throughout the package and have a single point
# of control, otherwise the test will break when I change it elswhere. Perhaps
# it should have a txt extension.
TEST_PER_FN = "test/test_per" 

def set_up_base_testing_dir():
    """ Creates a directory to store corpora and experimental directories used
    in testing. """

    if not os.path.isdir(EXP_BASE_DIR):
        os.makedirs(EXP_BASE_DIR)
    if not os.path.isdir(DATA_BASE_DIR):
        os.makedirs(DATA_BASE_DIR)

def download_example_data(example_dir, example_link):
    """
    Collects the zip archive from example_link and unpacks it into
    example_dir. It should contain num_utters WAV files.
    """

    set_up_base_testing_dir()

    zip_fn = join(DATA_BASE_DIR, "data.zip")

    # Remove data previously collected
    import shutil
    if os.path.isdir(example_dir):
        shutil.rmtree(example_dir)
    if os.path.isfile(zip_fn):
        os.remove(zip_fn)

    # Fetch the zip archive
    import urllib.request
    urllib.request.urlretrieve(example_link, filename=zip_fn)

    # Unzip the data
    import subprocess
    args = ["unzip", zip_fn, "-d", example_dir]
    subprocess.run(args, check=True)

def get_test_ler(exp_dir):
    """ Gets the test LER from the experiment directory."""

    test_per_fn = join(exp_dir, TEST_PER_FN)
    with open(test_per_fn) as f:
        ler = float(f.readlines()[0].split()[-1])

    return ler

@pytest.mark.slow
def test_tutorial():
    """ Tests running the example described in the tutorial in README.md """

    # 1024 utterance sample set.
    NA_EXAMPLE_LINK = "https://cloudstor.aarnet.edu.au/plus/s/YJXTLHkYvpG85kX/download"
    na_example_dir = join(DATA_BASE_DIR, "na_example/")

    download_example_data(na_example_dir, NA_EXAMPLE_LINK)

    # Test the first setup encouraged in the tutorial
    corp = corpus.ReadyCorpus(na_example_dir)
    exp_dir = run.train_ready(corp, directory=EXP_BASE_DIR)

    # Assert the convergence of the model at the end by reading the test scores
    ler = get_test_ler(exp_dir)
    assert ler < 0.3

def test_fast():
    """
    A fast integration test that runs 1 training epoch over a tiny
    dataset. Note that this does not run ffmpeg to normalize the WAVs since
    Travis doesn't have that installed. So the normalized wavs are included in
    the feat/ directory so that the normalization isn't run.
    """

    # 4 utterance toy set
    TINY_EXAMPLE_LINK = "https://cloudstor.aarnet.edu.au/plus/s/g2GreDNlDKUq9rz/download"
    tiny_example_dir = join(DATA_BASE_DIR, "tiny_example/")

    download_example_data(tiny_example_dir, TINY_EXAMPLE_LINK)

    corp = corpus.ReadyCorpus(tiny_example_dir)

    exp_dir = run.prep_exp_dir(directory=EXP_BASE_DIR)
    model = run.get_simple_model(exp_dir, corp)
    model.train(min_epochs=2, max_epochs=5)

    # Assert the convergence of the model at the end by reading the test scores
    ler = get_test_ler(exp_dir)
    # Can't expect a decent test score but just check that there's something.
    assert ler < 1.0

@pytest.mark.slow
def test_full_na():
    """ A full Na integration test. """

    # Pulls Na wavs from cloudstor.
    # TODO uncomment after test function is complete
    NA_WAVS_LINK = "https://cloudstor.aarnet.edu.au/plus/s/LnNyNa20GQ8qsPC/download"
    na_wav_dir = join(DATA_BASE_DIR, "na/wav/")
    #download_example_data(na_wav_dir, NA_WAVS_LINK)

    #NA_REPO_URL = "https://github.com/alexis-michaud/na-data.git"
    #with cd(DATA_BASE_DIR):
    #    subprocess.run(["git", "clone", NA_REPO_URL, "na/xml/"], check=True)
    na_xml_dir = join(DATA_BASE_DIR, "na/xml/TEXT/F4")

    # Tox is called from persephone base dir, so this data directory is relative to that.
    # Note also that this subdirectory only containts TEXTs, so this integration
    # test will include only Na narratives, not wordlists.

    # Fbank+pitch feature extraction, which relies on Kaldi being installed.
    from persephone.datasets import na

    label_dir = join(DATA_BASE_DIR, "na/label")
    na.prepare_labels("phonemes_and_tones",
        org_xml_dir=na_xml_dir, label_dir=label_dir)

    tgt_feat_dir = join(DATA_BASE_DIR, "na/feat")
    na.trim_wavs(org_wav_dir=na_wav_dir,
                                     tgt_wav_dir=tgt_feat_dir,
                                     org_xml_dir=na_xml_dir)

    na.prepare_feats("fbank_and_pitch",
                                         feat_dir=tgt_feat_dir,
                                         tgt_wav_dir=feat_dir,
                                         label_dir=label_dir)
    #persephone.datasets.na.make_data_splits(train_rec_type="text")

    # Training with texts

    # Ensure LER < 0.20


# Other tests:
    # TODO assert the contents of the prefix files

    # TODO Assert that we've filtered files with too many frames

    # TODO Test out corpus reader on the same data. (Make a function for that
    # within this functions scope)
