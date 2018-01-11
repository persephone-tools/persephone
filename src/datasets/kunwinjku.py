import os
from os.path import join

from nltk_contrib.textgrid import TextGrid

import config

def prepare_butcher_labels(label_dir):
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    for fn in os.listdir(config.BUTCHER_DIR):
        path = join(config.BUTCHER_DIR, fn)
        if path.endswith(".TextGrid"):
            prefix = os.path.basename(os.path.splitext(path)[0])
            out_path = join(label_dir, prefix+".phonemes")
            with open(path) as f, open(out_path, "w") as out_f:
                tg = TextGrid(f.read())
                for tier in tg:
                    if tier.nameid == "etic":
                        transcript = [text for _, _, text in tier.simple_transcript]
                        print(" ".join(transcript).strip(), file=out_f)

label_dir = join(config.TGT_DIR, "butcher/kun/")

prepare_butcher_labels(label_dir)

#os.listdir(config.BUTCHER_DIR)

#print(join(config.BUTCHER_DIR,
#    "487_551_493_165_Oscar_Kalarriya.TextGrid"))
