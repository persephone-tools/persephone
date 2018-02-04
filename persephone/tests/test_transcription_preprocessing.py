def test_segment_into_chars():
    from persephone.transcription_preprocessing import segment_into_chars

    input_1 = "hello"
    output_1 = "h e l l o"

    input_2 = "hello world"
    output_2 = "h e l l o w o r l d"

    input_3 = "hello wo     rld"
    output_3 = "h e l l o w o r l d"

    assert segment_into_chars(input_1) == output_1
    assert segment_into_chars(input_2) == output_2
    assert segment_into_chars(input_3) == output_3
