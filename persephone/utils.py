""" Miscellaneous utility functions. """
import logging
import logging.config
import os
from pathlib import Path
import subprocess
from subprocess import PIPE
from typing import List, Sequence, Tuple, TypeVar

from git import Repo # type: ignore
import numpy as np # type: ignore
from nltk.metrics import distance

from . import config
from .exceptions import DirtyRepoException

logger = logging.getLogger(__name__) # type: ignore

T = TypeVar("T")

def is_git_directory_clean(path_to_repo: Path,
                           search_parent_dirs: bool = True,
                           check_untracked: bool = False) -> None:
    """
    Check that the git working directory is in a clean state
    and raise exceptions if not.
    :path_to_repo: The path of the git repo
    """
    repo = Repo(str(path_to_repo), search_parent_directories=search_parent_dirs)
    logger.debug("is_git_directory_clean check for repo in path={} from "\
                  "cwd={} with search_parent_directories={}".format(
                        path_to_repo, os.getcwd(), search_parent_dirs))

    # If there are changes to already tracked files
    if repo.is_dirty():
        raise DirtyRepoException("Changes to the index or working tree."
                                 "Commit them first .")
    if check_untracked:
        if repo.untracked_files:
            raise DirtyRepoException("Untracked files. Commit them first.")

def target_list_to_sparse_tensor(target_list):
    """ Make tensorflow SparseTensor from list of targets, with each element in
    the list being a list or array with the values of the target sequence
    (e.g., the integer values of a character map for an ASR target string) See
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    contrib/ctc/ctc_loss_op_test.py for example of SparseTensor format
    """
    indices = []
    vals = []
    for t_i, target in enumerate(target_list):
        for seq_i, val in enumerate(target):
            indices.append([t_i, seq_i])
            vals.append(val)
    shape = [len(target_list), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def zero_pad(matrix, to_length):
    """ Zero pads along the 0th dimension to make sure the utterance array
    x is of length to_length."""

    assert matrix.shape[0] <= to_length
    if not matrix.shape[0] <= to_length:
        logger.error("zero_pad cannot be performed on matrix with shape {}"
                     " to length {}".format(matrix.shape[0], to_length))
        raise ValueError
    result = np.zeros((to_length,) + matrix.shape[1:])
    result[:matrix.shape[0]] = matrix
    return result

def collapse(batch_x, time_major=False):
    """ Converts timit into an array of format (batch_size, freq x num_deltas,
    time). Essentially, multiple channels are collapsed to one. """

    new_batch_x = []
    for utterance in batch_x:
        swapped = np.swapaxes(utterance, 0, 1)
        concatenated = np.concatenate(swapped, axis=1)
        new_batch_x.append(concatenated)
    new_batch_x = np.array(new_batch_x)
    if time_major:
        new_batch_x = np.transpose(new_batch_x, (1, 0, 2))
    return new_batch_x

#def load_batch_x(path_batch: Sequence[Path],
#                 flatten: bool = False,
#                 time_major: bool = False):
def load_batch_x(path_batch,
                 flatten = False,
                 time_major = False):
    """ Loads a batch of input features given a list of paths to numpy
    arrays in that batch."""

    utterances = [np.load(str(path)) for path in path_batch]
    utter_lens = [utterance.shape[0] for utterance in utterances]
    max_len = max(utter_lens)
    batch_size = len(path_batch)
    shape = (batch_size, max_len) + tuple(utterances[0].shape[1:])
    batch = np.zeros(shape)
    for i, utt in enumerate(utterances):
        batch[i] = zero_pad(utt, max_len)
    if flatten:
        batch = collapse(batch, time_major=time_major)
    return batch, np.array(utter_lens)

def batch_per(hyps: Sequence[Sequence[T]],
              refs: Sequence[Sequence[T]]) -> float:
    """ Calculates the phoneme error rate of a batch."""

    macro_per = 0.0
    for i in range(len(hyps)):
        ref = [phn_i for phn_i in refs[i] if phn_i != 0]
        hyp = [phn_i for phn_i in hyps[i] if phn_i != 0]
        macro_per += distance.edit_distance(ref, hyp)/len(ref)
    return macro_per/len(hyps)

def get_prefixes(dirname, extension):
    """ Returns a list of prefixes to files in the directory (which might be a whole
    corpus, or a train/valid/test subset. The prefixes include the path leading
    up to it, but only the filename up until the first observed period '.'
    """

    prefixes = []
    for root, _, filenames in os.walk(dirname):
        for filename in filenames:
            if filename.endswith(extension):
                # Then it's an input feature file and its prefix will
                # correspond to a training example
                prefixes.append(os.path.join(root, filename.split(".")[0]))
    return sorted(prefixes)

def get_prefix_lens(feat_dir: Path, prefixes: List[str],
                    feat_type: str) -> List[Tuple[str,int]]:
    prefix_lens = []
    for prefix in prefixes:
        path = feat_dir / ("%s.%s.npy" % (prefix, feat_type))
        _, batch_x_lens = load_batch_x([str(path)], flatten=False)
        prefix_lens.append((prefix, batch_x_lens[0]))
    return prefix_lens

def filter_by_size(feat_dir: Path, prefixes: List[str], feat_type: str,
                   max_samples: int) -> List[str]:
    """ Sorts the files by their length and returns those with less
    than or equal to max_samples length. Returns the filename prefixes of
    those files. The main job of the method is to filter, but the sorting
    may give better efficiency when doing dynamic batching unless it gets
    shuffled downstream.
    """

    # TODO Tell the user what utterances we are removing.
    prefix_lens = get_prefix_lens(Path(feat_dir), prefixes, feat_type)
    prefixes = [prefix for prefix, length in prefix_lens
                if length <= max_samples]
    return prefixes

def sort_by_size(feat_dir, prefixes, feat_type) -> List[str]:
    prefix_lens = get_prefix_lens(feat_dir, prefixes, feat_type)
    prefix_lens.sort(key=lambda prefix_len: prefix_len[1])
    prefixes = [prefix for prefix, _ in prefix_lens]
    return prefixes

def is_number(string):
    """ Tests if a string is valid float. """
    try:
        float(string)
        return True
    except ValueError:
        return False

def wav_length(fn):
    """ Returns the length of the WAV file in seconds."""

    args = [config.SOX_PATH, fn, "-n", "stat"]
    p = subprocess.Popen(
        args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    length_line = str(p.communicate()[1]).split("\\n")[1].split()
    print(length_line)
    assert length_line[0] == "Length"
    return float(length_line[-1])


def make_batches(paths: Sequence[Path], batch_size: int) -> List[Sequence[Path]]:
    """ Group utterances into batches for decoding.  """

    return [paths[i:i+batch_size]
            for i in range(0, len(paths), batch_size)]
