import os

KALDI_ROOT = "/home/oadams/code/kaldi/"

def make_scps(wav_dir):
    prefixes = []
    for fn in os.listdir(wav_dir):
        prefix, ext = os.path.splitext(fn)
        if ext == ".wav":
            prefixes.append(os.path.join(wav_dir, prefix))

    with open("wavs.scp", "w") as wav_scp:
        for prefix in prefixes:
            print(prefix, prefix + ".wav", file=wav_scp)

    with open("pitch_feats.scp", "w") as pitch_scp:
        for prefix in prefixes:
            print(prefix, prefix + ".pitch.txt", file=pitch_scp)

make_scps("/home/oadams/code/mam/data/na/wav")

# Then something like:
# kaldi/src/featbin/compute-kaldi-pitch-feats scp:wavs.scp scp,t:pitch_feats.scp
