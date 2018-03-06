from collections import defaultdict
from pathlib import Path
import sys

import pytest

from persephone import results
from persephone import utils
from persephone.corpus import Corpus
from persephone.corpus_reader import CorpusReader

@pytest.fixture(scope="module",
                params=["test", "valid"])
def prepared_data(request):
    data_path = Path("testing/data/bkw")
    exp_path = Path("testing/exp/19/")

    # TODO I shouldn't really be using "decoded" for the validation set dir
    # anymore.
    if request.param == "test":
        hyps_path = exp_path / "test" / "hyps"
        refs_path = exp_path / "test" / "refs"
    else:
        hyps_path = exp_path / "decoded" / "best_hyps"
        refs_path = exp_path / "decoded" / "refs"

    with hyps_path.open() as f:
        hyps = [hyp.split() for hyp in f.readlines()]
    with refs_path.open() as f:
        refs = [hyp.split() for hyp in f.readlines()]

    corp = Corpus.from_pickle(data_path)
    if request.param == "test":
        eval_prefixes = corp.test_prefixes
    else:
        eval_prefixes = corp.valid_prefixes

    return request.param, corp, eval_prefixes, hyps, refs

# TODO This should become testable on Travis using mock data.
@pytest.mark.experiment
def test_speaker_results(prepared_data):

    # TODO Make by-speaker results formatting part of results.py

    eval_type, corp, eval_prefixes, hyps, refs = prepared_data

    print()
    print("Eval type: ", eval_type)

    import pprint

    speakers = set()
    for utter in corp.utterances:
        speakers.add(utter.speaker)
    #print("{} speakers found.".format(len(speakers)))

    speaker_prefixes = defaultdict(list)
    assert corp.utterances
    for utter in corp.utterances:
        if utter.prefix in eval_prefixes:
            speaker_prefixes[utter.speaker].append(utter.prefix)

    prefix2speaker = dict()
    for utter in corp.utterances:
        prefix2speaker[utter.prefix] = utter.speaker

    speaker_hyps_refs = defaultdict(list)
    prefix_hyps_refs = zip(eval_prefixes, hyps, refs)
    for prefix, hyp, ref in prefix_hyps_refs:
        speaker_hyps_refs[prefix2speaker[prefix]].append((hyp, ref))


    print("{:20} {:10} {}".format("Speaker", "PER", "Num utters"))
    fmt = "{:20} {:<10.3f} {}"
    total_per = 0
    speaker_stats = []
    for speaker in speakers:
        if speaker_hyps_refs[speaker]:
            speaker_per = utils.batch_per(*zip(*speaker_hyps_refs[speaker]))
            speaker_stats.append((speaker, speaker_per,
                                 len(speaker_hyps_refs[speaker])))
            total_per += speaker_per*len(speaker_hyps_refs[speaker])
    speaker_stats.sort(key=lambda x: x[2], reverse=True)
    for speaker_stat in speaker_stats:
        print(fmt.format(*speaker_stat))
    print(fmt.format("Average",
                     total_per/len(eval_prefixes),
                     len(eval_prefixes)
                     )
         )

    print(results.fmt_confusion_matrix(hyps, refs,
                                       label_set=set(corp.labels),
                                       max_width=25))
    print(results.fmt_error_types(hyps, refs))
    results.fmt_latex_output(hyps, refs,
                             eval_prefixes,
                             Path("./hyps_refs.txt"))

    return

@pytest.fixture
def na_hyps_refs():
    na_test_data_path = Path("persephone/tests/test_data/na_nowordlist_tgm")
    hyps_path = na_test_data_path / "test" / "hyps"
    refs_path = na_test_data_path / "test" / "refs"
    prefixes_path = na_test_data_path / "test_prefixes.txt"

    with hyps_path.open() as f:
        hyps = [hyp.split() for hyp in f.readlines()]
    with refs_path.open() as f:
        refs = [ref.split() for ref in f.readlines()]
    with prefixes_path.open() as f:
        prefixes = [prefix.strip() for prefix in f.readlines()]

    return hyps, refs, prefixes

def test_fmt_confusion_matrix(na_hyps_refs):
    hyps, refs, _ = na_hyps_refs
    print()
    print(results.fmt_confusion_matrix(hyps, refs, max_width=12))

def test_fmt_error_types(na_hyps_refs):
    hyps, refs, _ = na_hyps_refs
    print()
    print(results.fmt_error_types(hyps, refs))

def test_fmt_latex_output(na_hyps_refs):
    hyps, refs, prefixes = na_hyps_refs
    print(prefixes)
    print()
    results.fmt_latex_output(hyps, refs, prefixes,
                                   Path("testing/fmt_latex_output.tex"))
