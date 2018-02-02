""" Ultimately a module to produce lattices from the multinomials of CTC-based
    neural networks. Includes other methods to do things such as sample and
    collapse paths.
"""

import os
import subprocess

import tensorflow as tf

from .corpus_reader import CorpusReader

from . import config

def test_sampling():
    """ As a proof of concept, first we want to sample paths from a trained CTC
    network's multinomials. After collapsing them, we want to assess their
    phoneme error rate to ensure we frequently get decent PERs."""
    pass

def compile_fst(prefix, syms_fn):
    """ Compiles the given text-based FST into a binary using OpenFST."""

    # Compile the fst
    args = [os.path.join(config.OPENFST_BIN_PATH, "fstcompile"),
            "--arc_type=log",
            "--isymbols=%s" % syms_fn,
            "--osymbols=%s" % syms_fn,
            "%s.txt" % prefix, "%s.bin" % prefix]
    subprocess.run(args)

def draw_fst(prefix, syms_fn):
    """ Draws the given fst using dot."""

    args = ["fstdraw", "--isymbols=%s" % syms_fn,
            "--osymbols=%s" % syms_fn,
            "%s.bin" % prefix, "%s.dot" % prefix]
    subprocess.run(args)

#    args = ["dot", "-Tpdf", "%s.dot" % prefix]
#    with open("%s.pdf" % prefix, "w") as out_f:
#        subprocess.run(args, stdout=out_f)

def create_symbol_table(index_to_token, filename):
    """ Creates a symbol table for the given vocab."""

    with open(filename, "w") as out_f:
        print("<eps> 0", file=out_f)
        for phone_id, phone in index_to_token.items():
            print("%s %d" % (phone, phone_id+1), file=out_f)
        print("<bl> %d" % (len(index_to_token)+1), file=out_f)

def logsoftmax2confusion(logsoftmax, index_to_token, prefix, beam_size):
    """ Converts a sequence of softmax outputs into a confusion network."""

    with open(prefix + ".confusion.txt", "w") as out_f:
        for node_id, timestep_softmax in enumerate(logsoftmax):
            phone_probs = list(zip(timestep_softmax, range(len(timestep_softmax))))
            # Only take the top entries within the beam.
            beam = sorted(phone_probs, reverse=True)[:beam_size]
            for prob, phone_id in beam:
                if phone_id == len(index_to_token):
                    # Then it's just outside the range of the mapping and is
                    # thus a blank symbol
                    print("%d %d <bl> <bl> %f" % (node_id, node_id+1, -prob),
                          file=out_f)
                else:
                    print("%d %d %s %s %f" % (
                          node_id, node_id+1,
                          index_to_token[phone_id], index_to_token[phone_id],
                          -prob),
                          file=out_f)
        # If I'm writing -logsoftmax values where they are arc costs, then
        # I believe 0 means a cost of zero.
        print("%d 0" % (node_id+1), file=out_f)

def create_collapse_fst(index_to_token, fst_fn):
    with open(fst_fn, "w") as out_f:
        for i, phone in sorted(index_to_token.items()):
            print("0 %d %s %s" % (i+1, phone, phone), file=out_f)
            print("%d %d %s <eps>" % (i+1, i+1, phone), file=out_f)
            for j, phone in sorted(index_to_token.items()):
                if j != i:
                    print("%d %d %s %s" % (i+1, j+1, phone, phone), file=out_f)
            print("%d 0 <bl> <eps>" % (i+1), file=out_f)
        print("0 0 <bl> <eps>", file=out_f)
        for i, phone in sorted(index_to_token.items()):
            print("%d 0" % (i+1), file=out_f)
        print("0 0", file=out_f)

def logsoftmax2lattice(logsoftmax, index_to_token, prefix, beam_size):
    """ Takes a numpy array of shape [time_steps, vocab] along with a function
    to produce tokens from their indices and an output filename prefix. A
    lattice is written to the output filename.
    """

    # Create the confusion network based on logsoftmax
    logsoftmax2confusion(logsoftmax, index_to_token, prefix, beam_size)
