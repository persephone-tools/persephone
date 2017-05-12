import os

import logits2lattices

vocab = ['pad', 'On', 'a', 'an', 'ch', 'd', 'dy', 'dz', 'e', 'en', 'i', 'in',
         'j', 'k', 'kw', 'l', 'ly', 'm', 'n', 'ny', 'o', 'p', 'q', 'r', 's',
         't', 'ts', 'ty', 'u', 'w', 'x', 'y', "<bl>"]

# Create the symbol list
logits2lattices.create_symbol_tables(vocab, "chatino_visual/symbols.txt")
logits2lattices.confusion2lattice_fst(vocab)
#numpy_dir = "chatino_visual/201/lattice"
#for fn in os.listdir(numpy_dir):
#    prefix = os.path.join(numpy_dir, fn[:-4])
#    logits2lattices.softmax2confusion(prefix, vocab)
#    print(prefix)
#    logits2lattices.compile_fst(prefix + "confusion.", "chatino_visual/symbols.txt")
#    logits2lattices.draw_fst(prefix + "confusion.", "chatino_visual/symbols.txt")
    # Compose fst
#    args = ["fstcompose", prefix + ".confusion.txt",
#    "confusion2lattice_fst.bin


# For each of the numpy lattices

