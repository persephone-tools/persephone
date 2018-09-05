import keras

class RNN_CTC_model:
    """Defines a RNN CTC model with Keras"""

    def __init__(self, exp_dir: str, corpus_reader, num_layers: int = 3,
            hidden_size: int=250, beam_width: int = 100,
            decoding_merge_repeated: bool = True) -> None:
        """Initialize a new model

        Arguments:
            exp_dir: Path that the experiment directory is located at
            corpus_reader: `CorpusReader` object that provides access to the corpus
                            this model is being trained on.
            num_layers: number of layers in the network
            hidden_size: the size, in nodes, of the hidden layers
            beam_width: size of the beam width (used for the decoding)
            decoding_merge_repeated: A flag to toggle behavior of repeating characters
                                     if true "a b b b b c" becomes "a b c"
        """
        raise NotImplementedError