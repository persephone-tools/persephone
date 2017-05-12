import numpy as np
import subprocess

def compile_fst(prefix, syms_fn):

    # Compile the fst
    args = ["fstcompile", "--isymbols=%s" % syms_fn,
            "--osymbols=%s" % syms_fn,
            "%s.txt" % prefix, "%s.bin" % prefix]
    subprocess.run(args)

def draw_fst(prefix, syms_fn):
    args = ["fstdraw", "--isymbols=%s" % syms_fn,
            "--osymbols=%s" % syms_fn,
            "%s.bin" % prefix, "%s.dot" % prefix]
    subprocess.run(args)

#    args = ["dot", "-Tpdf", "%s.dot" % prefix]
#    with open("%s.pdf" % prefix, "w") as out_f:
#        subprocess.run(args, stdout=out_f)

def create_symbol_tables(vocab, fn):

    # Store the symbol tables
    with open(fn, "w") as out_f:
        print("<eps> 0", file=out_f)
        for phone_id, phone in enumerate(vocab):
            print("%s %d" % (phone, phone_id+1), file=out_f)

def softmax2confusion(prefix, vocab=["a","b","c"]):
    """ Converts a sequence of softmax outputs into a confusion network."""

    a = np.load(prefix + ".npy")
    with open(prefix + ".confusion.txt", "w") as out_f:
        for node_id, softmax in enumerate(a):
            for phone_id, prob in enumerate(softmax):
                print("%d %d %s %s %f" % (
                    node_id, node_id+1, vocab[phone_id], vocab[phone_id], prob),
                    file=out_f)
        print("%d 1" % (node_id+1), file=out_f)

def confusion2lattice_fst(vocab):
    with open("confusion2lattice_fst.txt", "w") as out_f:
        for i, phone in enumerate(vocab[:-1]):
            print("0 %d %s %s" % (i+1, phone, phone), file=out_f)
            print("%d %d %s <eps>" % (i+1, i+1, phone), file=out_f)
            for j, phone in enumerate(vocab[:-1]):
                if j != i:
                    print("%d %d %s %s" % (i+1, j+1, phone, phone), file=out_f)
            print("%d 0 <bl> <eps>" % (i+1), file=out_f)
        print("0 0 <bl> <eps>", file=out_f)
        for i, phone in enumerate(vocab):
            print("%d 0" % (i+1), file=out_f)
        print("0 0", file=out_f)
