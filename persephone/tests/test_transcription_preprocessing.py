def test_segment_into_chars():
    from persephone.preprocess.labels import segment_into_chars

    input_1 = "hello"
    output_1 = "h e l l o"

    input_2 = "hello world"
    output_2 = "h e l l o w o r l d"

    input_3 = "hello wo     rld"
    output_3 = "h e l l o w o r l d"

    input_4 = "hello wo     rld\r\n"
    output_4 = "h e l l o w o r l d"

    assert segment_into_chars(input_1) == output_1
    assert segment_into_chars(input_2) == output_2
    assert segment_into_chars(input_3) == output_3

def test_segment_into_tokens():
    from persephone.preprocess.labels import segment_into_tokens
    from persephone.datasets.na import PHONEMES
    from persephone.datasets.na import TONES
    from persephone.datasets.na import SYMBOLS_TO_PREDICT

    input_1 = "ə˧ʝi˧-ʂɯ˥ʝi˩ | -dʑo˩ … | ə˩-gi˩!"
    output_1 = "ə ˧ ʝ i ˧ ʂ ɯ ˥ ʝ i ˩ | dʑ o ˩ | ə ˩ g i ˩"


    input_2 = "ʈʂʰɯ˧ne˧ ʝi˥-kv̩˩-tsɯ˩ | -mv̩˩."
    output_2 = "ʈʂʰ ɯ ˧ n e ˧ ʝ i ˥ k v̩ ˩ ts ɯ ˩ | m v̩ ˩"

    input_3 = "   ʈʂʰɯ˧ne˧ ʝi˥-kv̩˩-tsɯ˩ | -mv̩˩.\r\n"
    output_3 = "ʈʂʰ ɯ ˧ n e ˧ ʝ i ˥ k v̩ ˩ ts ɯ ˩ | m v̩ ˩"

    token_inv = PHONEMES.union(TONES).union(SYMBOLS_TO_PREDICT)

    assert segment_into_tokens(input_1, token_inv) == output_1
    assert segment_into_tokens(input_2, token_inv) == output_2
    assert segment_into_tokens(input_3, token_inv) == output_3
