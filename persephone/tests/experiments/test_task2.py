""" Experimenting and design relating to Task 2 of the BKW work: finding the
part of audio containing some transcription.
"""

import logging
from pathlib import Path
import regex
import time

logging.basicConfig(filename="task2_log.txt", level=logging.DEBUG)

def clock(func):
    def decorated(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        return t2-t1, res
    return decorated

@clock
def fuzzy_align(data: str, substring: str):
    """ Finds likely instances of substring in data. """

    #logging.info("Searching in\n\t{}\nfor\n\t{}".format(data, substring))
    #logging.info("len(data): {}".format(len(data)))
    #logging.info("len(substring): {}".format(len(substring)))
    # (?b) gives the BESTMATCH
    search_str = "(?b)(?:{})".format(substring)
    match = regex.search(search_str+"{e}", data)
    return match

def get_hyps_refs():
    exp_dir = Path("testing/exp/41/0")
    refs = []
    with (exp_dir / "decoded" / "refs").open() as f:
        for line in f:
            refs.append(line.strip())
    hyps = []
    with (exp_dir / "decoded" / "best_hyps").open() as f:
        for line in f:
            hyps.append(line.strip())

    return hyps, refs

def test_fuzzy_align():

    hyps, refs = get_hyps_refs()
    reference = " ".join(refs)
    print(fuzzy_align(reference, " ".join(hyps[4:6])))

def test_fuzzy_align_time_complexity():

    # Three variables: len(data), len(substring), the edit distance of the two.

    # Keeping len(data) and edit distance fixed (approximately), by using the
    # references.
    hyps, refs = get_hyps_refs()
    reference = " ".join(refs)
    hyps.sort(key=lambda hyp: len(hyp))
    for hyp in hyps:
        time, res = fuzzy_align(reference, hyp)
        logging.info("Took {} seconds for len({}) input.".format(
            time, len(hyp)))
