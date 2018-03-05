""" Script for formatting results of experiments """

from . import config
from .datasets import na
from pathlib import Path
from typing import Set, Dict, Tuple, Sequence

from collections import Counter
from collections import defaultdict
import os
from . import utils

from .distance import min_edit_distance_align
from .distance import cluster_alignment_errors

def round_items(floats):
    return ["%0.3f" % fl for fl in floats]

def average(exp_nums, show_tgm_f1=False, phones=na.PHONEMES, tones=na.TONES):
    """ Averages the results across a few experimental runs. """

    ler_total = 0
    per_total = 0
    ter_total = 0
    tgm_f1_total = 0
    for i, exp_num in enumerate(exp_nums):
        path = os.path.join(config.EXP_DIR, str(exp_num))
        ler, per, ter = test_results(path, phones, tones)
        tgm_f1 = 0
        if show_tgm_f1:
            tgm_f1 = symbol_f1(exp_num, "|")
        print("Exp #{}:".format(i))
        print("\tPER & TER & LER & TGM-F1")
        print("\t{} & {} & {} & {}".format(per, ter, ler, tgm_f1))
        ler_total += ler
        per_total += per
        ter_total += ter
        tgm_f1_total +=  tgm_f1
    print("Average:")
    print("\tPER & TER & LER & TGM-F1")
    print("\t{} & {} & {} & {}\\\\".format(
        per_total/(i+1), ter_total/(i+1), ler_total/(i+1), tgm_f1_total/(i+1)))

def format(exp_paths,
                   phones=na.PHONEMES,
                   tones=na.TONES):
    """ Takes a list of experimental paths such as mam/exp/<number> and outputs
    the results. """

    valid_lers = []
    valid_pers = []
    test_lers = []
    test_pers = []
    test_ters = []

    for path in exp_paths:

        test_ler, test_per, test_ter = test_results(path, phones, tones)
        test_lers.append(test_ler)
        test_pers.append(test_per)
        test_ters.append(test_ter)

        with open(os.path.join(path, "best_scores.txt")) as best_f:
            sp = best_f.readline().replace(",", "").split()
            training_ler, valid_ler, valid_per = float(sp[4]), float(sp[7]), float(sp[10])
            valid_lers.append(valid_ler)

    print("Valid LER", round_items(valid_lers))
    print("Test LER", round_items(test_lers))
    print("Test PER", round_items(test_pers))
    print("Test TER", round_items(test_ters))

    print("PERS:")
    for item in zip([128,256,512,1024,2048], test_pers):
        print("(%d, %f)" % item)

    print("TERS:")
    for item in zip([128,256,512,1024,2048], test_ters):
        print("(%d, %f)" % item)

def filter_labels(sent, labels=None):
    """ Returns only the tokens present in the sentence that are in labels."""
    if labels:
        return [tok for tok in sent if tok in labels]
    return sent

def test_results(exp_path, phones, tones):
    """ Gets results of the model on the test set. """

    test_path = os.path.join(exp_path, "test")
    print(test_path)
    with open(os.path.join(test_path, "test_per")) as test_f:
        line = test_f.readlines()[0]
        test_ler = float(line.split()[2].strip(","))

    test_per = filtered_error_rate(os.path.join(test_path, "hyps"),
                                      os.path.join(test_path, "refs"),
                                      phones)

    test_ter = filtered_error_rate(os.path.join(test_path, "hyps"),
                                      os.path.join(test_path, "refs"),
                                      tones)

    return test_ler, test_per, test_ter

def filtered_error_rate(hyps_path, refs_path, labels):

    with open(hyps_path) as hyps_f:
        lines = hyps_f.readlines()
        hyps = [filter_labels(line.split(), labels) for line in lines]
    with open(refs_path) as refs_f:
        lines = refs_f.readlines()
        refs = [filter_labels(line.split(), labels) for line in lines]

    # For the case where there are no tokens left after filtering.
    only_empty = True
    for entry in hyps:
        if entry != []:
            only_empty = False
    if only_empty:
        return -1

    return utils.batch_per(hyps, refs)

def ed_alignments(exp_path):

    test_path = os.path.join(exp_path, "test")
    hyps_path = os.path.join(test_path, "hyps")
    refs_path = os.path.join(test_path, "refs")

    with open(hyps_path) as hyps_f:
        lines = hyps_f.readlines()
        hyps = [line.split() for line in lines]
    with open(refs_path) as refs_f:
        lines = refs_f.readlines()
        refs = [line.split() for line in lines]

    alignments = []
    for ref, hyp in zip(refs, hyps):
        alignment = min_edit_distance_align(ref, hyp)
        alignments.append(alignment)

    return alignments

def symbol_f1(exp_num, symbol):

    exp_path = os.path.join(config.EXP_DIR, str(exp_num))
    alignments = ed_alignments(exp_path)

    correct = 0
    del_ = 0
    ins = 0
    del_sub = 0
    ins_sub = 0
    total_hyp = 0
    total_ref = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for alignment in alignments:
        for arrow in alignment:
            if arrow[0] == symbol:
                if arrow[1] == symbol:
                    tp +=1
                else:
                    fn += 1
            elif arrow[0] != symbol:
                if arrow[1] == symbol:
                    fp += 1
                else:
                    tn += 1

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp+fp)
    fdr = fp/(tp+fp)
    f1 = 2*tp/(2*tp + fp + fn) # precision and sensitivity harmonic mean.
    #print("Sensitivity: {}".format(sensitivity))
    #print("(False negative rate: {})".format(1-sensitivity))
    #print("Specificity: {}".format(specificity))
    #print("(False positive rate: {})".format(1-specificity))
    #print("Accuracy: {}".format(accuracy))
    #print("(Error rate: {})".format(1-accuracy))
    #print("Precision: {}".format(precision))
    #print("(False discovery rate: {})".format(fdr))
    #print("F1: {}".format(f1))
    #print("TP: {}, FN: {}".format(tp, fn))
    #print("TN: {}, FP: {}".format(tn, fp))

    errors = []
    for alignment in alignments:
        for arrow in alignment:
            if arrow[0] != arrow[1]:
                errors.append(arrow)

    err_hist = {}
    for error in errors:
        if error in err_hist:
            err_hist[error] += 1
        else:
            err_hist[error] = 1

    return f1

def fmt_latex_output(hyps: Sequence[Sequence[str]],
                     refs: Sequence[Sequence[str]],
                     prefixes: Sequence[str],
                     out_fn: Path,
                    ) -> None:
    """ Output the hypotheses and references to a LaTeX source file for
    pretty printing.
    """

    alignments_ = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    with out_fn.open("w") as out_f:
        print("\documentclass[10pt]{article}\n"
              "\\usepackage[a4paper,margin=0.5in,landscape]{geometry}\n"
              "\\usepackage[utf8]{inputenc}\n"
              "\\usepackage{xcolor}\n"
              "\\usepackage{polyglossia}\n"
              "\\usepackage{booktabs}\n"
              "\\usepackage{longtable}\n"
              "\setmainfont[Mapping=tex-text,Ligatures=Common,Scale=MatchLowercase]{Doulos SIL}\n"
              "\DeclareRobustCommand{\hl}[1]{{\\textcolor{red}{#1}}}\n"
              #"% Hyps path: " + hyps_path +
              "\\begin{document}\n"
              "\\begin{longtable}{ll}", file=out_f)

        print("\\toprule", file=out_f)
        for sent in zip(prefixes, alignments_):
            prefix = sent[0]
            alignments = sent[1:]
            print("Utterance ID: &", prefix.strip().replace("_", "\_"), "\\\\", file=out_f)
            for i, alignment in enumerate(alignments):
                ref_list = []
                hyp_list = []
                for arrow in alignment:
                    if arrow[0] == arrow[1]:
                        # Then don't highlight it; it's correct.
                        ref_list.append(arrow[0])
                        hyp_list.append(arrow[1])
                    else:
                        # Then highlight the errors.
                        ref_list.append("\hl{%s}" % arrow[0])
                        hyp_list.append("\hl{%s}" % arrow[1])
                print("Ref: &", "".join(ref_list), "\\\\", file=out_f)
                print("Hyp: &", "".join(hyp_list), "\\\\", file=out_f)
            print("\\midrule", file=out_f)

        print("\end{longtable}", file=out_f)
        print("\end{document}", file=out_f)

def fmt_error_types(hyps: Sequence[Sequence[str]],
                    refs: Sequence[Sequence[str]]
                   ) -> str:

    alignments = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    arrow_counter = Counter() # type: Dict[Tuple[str, str], int]
    for alignment in alignments:
        arrow_counter.update(alignment) 
    sub_count = sum([count for arrow, count in arrow_counter.items()
                if arrow[0] != arrow[1] and arrow[0] != "" and arrow[1] != ""])
    dels = [(arrow[0], count) for arrow, count in arrow_counter.items()
            if arrow[0] != arrow[1] and arrow[0] != "" and arrow[1] == ""]
    del_count = sum([count for arrow, count in dels])
    ins_count = sum([count for arrow, count in arrow_counter.items()
                if arrow[0] != arrow[1] and arrow[0] == "" and arrow[1] != ""])
    total = sum([count for _, count in arrow_counter.items()])

    fmt_pieces = []
    fmt = "{:15}{:<4}\n"
    fmt_pieces.append(fmt.format("Substitutions", sub_count))
    fmt_pieces.append(fmt.format("Deletions", del_count))
    fmt_pieces.append(fmt.format("Insertions", ins_count))
    fmt_pieces.append("\n")
    fmt_pieces.append("Deletions:\n")
    fmt_pieces.extend(["{:4}{:<4}\n".format(label, count)
                       for label, count in
                       sorted(dels, reverse=True, key=lambda x: x[1])])

    return "".join(fmt_pieces)


def fmt_confusion_matrix(hyps: Sequence[Sequence[str]],
                         refs: Sequence[Sequence[str]],
                         label_set: Set[str] = None,
                         max_width: int = 25) -> str:
    """ Formats a confusion matrix over substitutions, ignoring insertions
    and deletions. """

    if not label_set:
        # Then determine the label set by reading
        raise NotImplementedError()

    alignments = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    arrow_counter = Counter() # type: Dict[Tuple[str, str], int]
    for alignment in alignments:
        arrow_counter.update(alignment)

    ref_total = Counter() # type: Dict[str, int]
    for alignment in alignments:
        ref_total.update([arrow[0] for arrow in alignment])

    labels = [label for label, count
              in sorted(ref_total.items(), key=lambda x: x[1], reverse=True)
              if label != ""][:max_width]

    format_pieces = []
    fmt = "{:3} "*(len(labels)+1)
    format_pieces.append(fmt.format(" ", *labels))
    fmt = "{:3} " + ("{:<3} " * (len(labels)))
    for ref in labels:
        # TODO
        ref_results = [arrow_counter[(ref, hyp)] for hyp in labels]
        format_pieces.append(fmt.format(ref, *ref_results))

    return "\n".join(format_pieces)
