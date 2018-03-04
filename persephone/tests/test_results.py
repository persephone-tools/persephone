from collections import defaultdict
from pathlib import Path

import pytest

from persephone import results
from persephone import utils
from persephone.corpus import Corpus

#@pytest.fixture
#def bkw_exp_dir():
#    path = Path("persephone/tests/test_sets/bkw_slug_exp_13")

def test_speaker_results():
    data_path = Path("testing/data/bkw")
    exp_path = Path("testing/exp/19/")
    # TODO I shouldn't really be using "decoded" for the validation set dir
    # anymore.
    hyps_path = exp_path / "decoded" / "best_hyps"
    refs_path = exp_path / "decoded" / "refs"

    with hyps_path.open() as f:
        hyps = [hyp.split() for hyp in f.readlines()]
    with refs_path.open() as f:
        refs = [hyp.split() for hyp in f.readlines()]

    corp = Corpus.from_pickle(data_path)

    import pprint

    speaker = "Mark Djandiomerr"
    speaker_prefixes = defaultdict(list)
    assert corp.utterances
    for utter in corp.utterances:
        if utter.speaker == speaker and utter.prefix in corp.valid_prefixes:
            speaker_prefixes[speaker].append(utter.prefix)

    pprint.pprint(speaker_prefixes)

    speaker_hyps_refs = defaultdict(list)
    prefix_hyps_refs = zip(corp.valid_prefixes, hyps, refs)
    for prefix, hyp, ref in prefix_hyps_refs:
        if prefix in speaker_prefixes[speaker]:
            speaker_hyps_refs[speaker].append((hyp, ref))
    print(len(speaker_hyps_refs[speaker]))
    print(utils.batch_per(*zip(*speaker_hyps_refs[speaker])))

    return

    # Create a Dict[speaker, List[prefix]]

    # Create a Dict[speaker, Tuple[List[hyp], List[ref]]]

    def make_spk2hypsrefs(hyps, refs, prefixes, prefix2speaker):
        spk2hypsrefs = defaultdict(lambda: [[],[]])
        for prefix, hyp, ref in zip(prefixes, hyps, refs):
            spk2hypsrefs[prefix2speaker[prefix]][0].append(hyp)
            spk2hypsrefs[prefix2speaker[prefix]][1].append(ref)
        return spk2hypsrefs

    per = utils.batch_per(hyps, refs)
    print(per)
    spk2hypsrefs = make_spk2hypsrefs(hyps, refs, prefixes, prefix2speaker)
    #alignments = []
    #for ref, hyp in zip(refs, hyps):
    #    alignment = distance.min_edit_distance_align(ref, hyp)
    #    alignments.append(alignment)
    #print(alignments)

    #res = results.by_speaker(hyps, refs, prefixes, prefix2speaker)
