""" Preprocesses the TIMIT data into a format for end-to-end phoneme recognition."""

import os
from os.path import join
import shutil
import subprocess

from .. import config
from .. import feat_extract

TIMIT_TGT_DIR = os.path.join(config.TGT_DIR, "timit")

def all_feature_extraction(feat_type):
    """ Walk over all the wav files in the TIMIT dir and extract features. """

    for root, _, fns in os.walk(TIMIT_TGT_DIR):
        print("Processing speaker %s" % root)
        for filename in fns:
            if filename.endswith(".wav"):
                if feat_type == "log_mel_filterbank":
                    feat_extract.logfbank_feature_extraction(join(root, filename))
                elif feat_type == "mfcc13_d":
                    feat_extract.feature_extraction(join(root, filename))
                else:
                    raise Exception("Invalid feature type selection.")

def create_raw_data():
    """ Copies the original TIMIT data to a working directory and does basic
    preprocessing such as removing phone timestamps and converting NIST files
    to WAV."""

    def preprocess_phones(org_path, tgt_path):
        """ Takes the org_path .phn file, preprocesses and writes to tgt_path."""
        with open(org_path) as in_f, open(tgt_path, "w") as out_f:
            phones = []
            for line in in_f:
                phones.append(line.strip().split()[-1])
            print(" ".join(phones), file=out_f)

    def sph2wav(sphere_path, wav_path):
        """ Calls sox to convert the sphere file to wav."""
        args = [config.SOX_PATH, sphere_path, wav_path]
        subprocess.run(args)

    for root, _, fns in os.walk(config.TIMIT_ORG_DIR):
        for filename in fns:
            org_path = join(root, filename)
            sub_path = join(root[len(config.TIMIT_ORG_DIR):], filename)
            tgt_path = join(TIMIT_TGT_DIR, sub_path)

            # Create parent directory
            parent_path = os.path.dirname(tgt_path)
            if not os.path.exists(parent_path):
                os.makedirs(parent_path)

            print("Creating raw data for %s" % tgt_path)

            if filename.endswith("phn"):
                preprocess_phones(org_path, tgt_path)
            elif filename.endswith("wav"):
                # It's actually in sphere format
                sphere_path = tgt_path[:-3]+"sph"
                shutil.copyfile(org_path, sphere_path)
                sph2wav(sphere_path, tgt_path)

if __name__ == "__main__":
    #create_raw_data()
    all_feature_extraction(feat_type="log_mel_filterbank")
