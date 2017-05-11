import numpy as np
import subprocess

def softmax2confusion(prefix):
    """ Converts a sequence of softmax outputs into a confusion network."""

    vocab = ["a", "b", "c"]
    a = np.load(prefix + ".npy")
    with open(prefix + ".confusion.txt", "w") as out_f:
        for node_id, softmax in enumerate(a):
            for phone_id, prob in enumerate(softmax):
                print("%d %d %s %s %f" % (
                    node_id, node_id+1, vocab[phone_id], vocab[phone_id], prob),
                    file=out_f)

    # Store the symbol tables
    with open(prefix + ".isymbols.txt", "w") as out_f:
        print("<eps> 0", file=out_f)
        for phone_id, phone in enumerate(vocab):
            print("%s %d" % (phone, phone_id), file=out_f)
    with open(prefix + ".osymbols.txt", "w") as out_f:
        print("<eps> 0", file=out_f)
        for phone_id, phone in enumerate(vocab):
            print("%s %d" % (phone, phone_id), file=out_f)

    # Compile the fst
    args = ["fstcompile", "--isymbols=%s.isymbols.txt" % prefix,
            "--osymbols=%s.isymbols.txt" % prefix,
            "%s.confusion.txt" % prefix, "%s.confusion.bin" % prefix]
    subprocess.run(args)

    args = ["fstdraw", "--isymbols=%s.isymbols.txt" % prefix,
            "--osymbols=%s.isymbols.txt" % prefix,
            "%s.confusion.bin" % prefix, "%s.confusion.dot" % prefix]
    subprocess.run(args)

    args = ["dot", "-Tpdf", "%s.confusion.dot" % prefix]
    with open("%s.confusion.pdf" % prefix, "w") as out_f:
        subprocess.run(args, stdout=out_f)
