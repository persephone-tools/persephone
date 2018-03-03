from pathlib import Path

import pytest

from persephone import results
from persephone.corpus import Corpus

#@pytest.fixture
#def bkw_exp_dir():
#    path = Path("persephone/tests/test_sets/bkw_slug_exp_13")

def test_speaker_results():
    data_path = Path("testing/data/bkw")
    exp_path = Path("testing/exp/20/")
    # TODO I shouldn't really be using "decoded" for the validation set dir
    # anymore.
    hyps_path = exp_path / "decoded" / "best_hyps"
    refs_path = exp_path / "decoded" / "refs"

    with hyps_path.open() as f:
        hyps = [hyp.split() for hyp in f.readlines()]
    with refs_path.open() as f:
        refs = [hyp.split() for hyp in f.readlines()]

    corp = Corpus.from_pickle(data_path)

    # Create a Dict[speaker, List[prefix]]

    # Create a Dict[speaker, Tuple[List[hyp], List[ref]]]

    from persephone import distance
    from persephone import utils

    from collections import defaultdict

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
