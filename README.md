# Persephone v0.0.0

Persephone (/pərˈsɛfəni/) is an automatic phoneme transcription tool. Traditional speech recognition tools require a large pronunciation lexicon (describing how words are pronounced) and much training data so that the system can learn to output orthographic transcriptions. In contrast, Persephone is designed for situations where training data is limited, perhaps as little as 30 minutes of transcribed speech. Such limitations on data are common in the documentation of low-resource languages. It is possible to use such small amounts of data to train a transcription model that can help aid transcription, yet such technology has not been widely adopted.

The goal of Persephone is to make state-of-the-art phonemic transcription accessible to people involved in language documentation. Creating an easy-to-use user interface is central to this. The user interface and APIs are currently a work in progress.

The tool is implemented in Python/Tensorflow with extensibility in mind. Currently just one model is implemented, which uses bidirectional LSTMs and the connectionist temporal classification (CTC) loss function.

We are happy to offer direct help to anyone who wants to use it. If you're having trouble, contact Oliver Adams at oliver.adams@gmail.com. We are also very welcome to thoughts, constructive criticism, help and pull requests.

## Quickstart

This guide is written to help you get the tool working on your machine. We will use a small example setup that involves training a phoneme transcription tool for [Yongning Na](http://lacito.vjf.cnrs.fr/pangloss/languages/Na_en.php). The example that we will run can be run on most personal computers without a graphics processing unit (GPU), since I've made the settings less computationally demanding than is optimal. Ideally you'd have access to a server with more memory and a GPU, but this isn't necessary.

### 1. Installation

The code has been tested on Mac and Linux systems. It hasn't yet been tested on Windows.

Ensure Python 3, ffmpeg and git have been installed. These should be available for your operating system and can be installed via your package manager.

For now you must open up a terminal to enter commands at the command line. (The commands below are prefixed with a "$". Don't enter the "$", just whatever comes afterwards).

Fetch the latest code:

```
$ git clone git@github.com:oadams/mam.git
$ cd mam
```

We now need to set up some dependencies in a virtual environment.
```
$ virtualenv -p python3 venv3
$ source ~/venv3/bin/activate
$ pip install -r requirements.txt
```

### 2. Get the example data

Currently the tool assumes each utterance is in its own audio file, and that for each utterance in the training set there is a corresponding transcription file with phonemes (or perhaps characters) delimited by spaces. I've uploaded an example dataset that includes some Yongning Na data that has already been preprocessed. We'll use this example dataset in this tutorial. Once we confirm that the software itself is working on your computer, we can discuss preprocessing of your own data.

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
 
The message may vary a bit depending on your CPU, but if it says "Batch 0" at the bottom without an error, then training is very likely working. Contact me if you have any trouble getting to this point, or if you had to deviate from the above instructions to get to this point.

On the current settings it will train through batches 1 to 200 or so for at least 30 "epochs", potentially more. If you don't have a GPU then this will take quite a while, though you should notice it converging in performance within a couple hours on most personal computers.

After a few epochs you can see how its going by going to opening up `mam/exp/<experiment_number>/train_log.txt`. This will show you the error rates on the training set and the held-out validation set. In the `mam/exp/<experiment_number>/decoded` subdirectory, you'll see the validation set reference in `refs` and the model hypotheses for each epoch in `epoch<epoch_num>_hyps`.
