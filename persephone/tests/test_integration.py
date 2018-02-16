import os
from os.path import join
import pytest

from persephone import corpus
from persephone import run

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

def prepare_example_data(example_dir, example_link, num_utters):
    """
    Collects the zip archive from example_link and unpacks it into
    example_dir. It should contain num_utters WAV files.
    """

    set_up_base_testing_dir()

    zip_fn = join(DATA_BASE_DIR, "example_data.zip")

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
    args = ["unzip", zip_fn, "-d", DATA_BASE_DIR]
    subprocess.run(args)

    # Check we have the right number of utterances
    wav_dir = join(example_dir, "wav/")
    assert len(os.listdir(wav_dir)) == num_utters

def get_test_ler(exp_dir):
    """ Gets the test LER from the experiment directory."""

    test_per_fn = join(exp_dir, TEST_PER_FN)
    with open(test_per_fn) as f:
        ler = float(f.readlines()[0].split()[-1])

    return ler

@pytest.mark.slow
def test_tutorial():
    """ Tests running the example described in the tutorial in README.md """

    # Prepare paths
    NA_EXAMPLE_LINK = "https://cloudstor.aarnet.edu.au/plus/s/YJXTLHkYvpG85kX/download"
    NUM_UTTERS = 1024
    na_example_dir = join(DATA_BASE_DIR, "na_example/")

    prepare_example_data(na_example_dir, NA_EXAMPLE_LINK, NUM_UTTERS)

    # Test the first setup encouraged in the tutorial
    corp = corpus.ReadyCorpus(na_example_dir)
    exp_dir = run.train_ready(corp, directory=EXP_BASE_DIR)

    # Assert the convergence of the model at the end by reading the test scores
    ler = get_test_ler(exp_dir)
    assert ler < 0.3

def test_fast():
    """
    A fast integration test that runs 1 training epoch over a tiny
    dataset.
    """

    TINY_EXAMPLE_LINK = "https://cloudstor.aarnet.edu.au/plus/s/WIJW9uS713ZdUS8/download"
    NUM_UTTERS = 4
    tiny_example_dir = join(DATA_BASE_DIR, "tiny_example/")

    prepare_example_data(tiny_example_dir, TINY_EXAMPLE_LINK, NUM_UTTERS)

    corp = corpus.ReadyCorpus(tiny_example_dir)

    exp_dir = run.train_ready(corp, directory=EXP_BASE_DIR)

    # Assert the convergence of the model at the end by reading the test scores
    ler = get_test_ler(exp_dir)
    # Can't expect a decent test score but just check that there's something.
    assert ler < 1.0

# Other tests:
    # TODO assert the contents of the prefix files

    # TODO Assert that we've filtered files with too many frames

    # TODO Test out corpus reader on the same data. (Make a function for that
    # within this functions scope)
