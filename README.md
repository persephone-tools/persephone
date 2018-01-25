# Persephone

Persephone (/pərˈsɛfəni/) is an automatic phoneme transcription tool for
low-resource languages. It is designed for situations where the traditional
speech recognition pipeline, which depends on a large pronunciation lexicon and
substantial training data, is inapplicable because of the lack of such a
lexicon.

The goal is to make state-of-the-art phonemic transcription accessible.

The tool is implemented in Python/Tensorflow. Currently implemented is a
CTC-based model.

The user interface and APIs are a work in progress. I'm happy to offer direct
help to anyone who wants to use it. Contact me at oliver.adams@gmail.com.

## Quickstart

This tutorial will ensure the code is working on your machine. The code be run on computers without GPUs since I've made the settings less computationally demanding than they normally would be. Ideally you'd have access to a server with more memory and a GPU.

### 1. Installation

The code has been tested on Mac and Linux systems. It should work on Windows too, but that hasn't yet been tested.

From here on I assume Python 3, ffmpeg and git have been installed. These should be available for your operating system.

For now you must open up a terminal to enter commands at the command line. (The commands below are prefixed with a "$". Don't enter the "$", just whatever comes afterwards).

Fetch the latest code:

```
$ git clone git@github.com:oadams/mam.git
$ cd mam
```

We now need to set up some dependencies in a virtual environment. Run:
```
$ virtualenv -p python3 venv3
$ source ~/venv3/bin/activate
$ pip install -r requirements.txt
```

### 2. Get the example data

Currently the tool assumes each utterance is in its own audio file, and that there is a corresponding transcription file where each token is a phoneme or tone. I've uploaded an example dataset that includes some Na data that has already been preprocessed. We'll use this example dataset in this tutorial. Once we confirm that the software itself is working on your computer, we can discuss preprocessing of your own data.

Get the data [here](https://cloudstor.aarnet.edu.au/sender/?s=download&token=b6789ee3-bbcb-7f92-2f38-18ffc1086817)

Unzip `na_example.zip`. There should now be a directory `na_example/`, with subdirectories `feat/` and `label/`. You can put `na_example` anywhere, but for the rest of this tutorial I assume it is in `mam/data/na_example/`.

### 3. Running an experiment

One way to conduct experiments is to run the code from the iPython interpreter. Back to the terminal:

```
$ cd src
$ ipython
> import corpus
> corp = corpus.ReadyCorpus("../data/na_example")
> import run
> run.train_ready(corp)
```

You'll should now see something like:

```
3280
2018-01-18 10:30:22.290964: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    exp_dir ../exp/1, epoch 0
        Batch 0
        Batch 1
        ...
```
 
The message may vary a bit depending on your CPU, but if it says "Batch 0" at the bottom without an error, then training is very likely working. Contact me if you have any trouble getting to this point.

On the current settings it will train through batches 1...205 for at least 30 "epochs", potentially more. If you don't have a GPU then this will take quite a while, though you should notice it converging in performance within a couple hours on most personal computers.

After a few epochs you can see how its going by going to opening up `mam/exp/<experiment_number>/train_log.txt`. This will show you the error rates on the training set and the held-out validation set. In the `mam/exp/<experiment_number>/decoded` subdirectory, you'll see the validation set reference in `refs` and the model hypotheses for each epoch in `epoch<epoch_num>_hyps`.
