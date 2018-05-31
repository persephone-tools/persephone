from pathlib import Path


def test_too_short():
    from persephone.utterance import Utterance
    from persephone import utterance
    utterance_too_short = Utterance(
        org_media_path=Path(
            'data/org/BKW-speaker-ids/Mark on rock with Timecode.mp4'),
        org_transcription_path=Path(
            'data/org/BKW-speaker-ids/Mark on Rock.eaf'),
        prefix='Mark on Rock.rf@MARK.401',
        start_time=1673900, end_time=1673923,
        text=' kunkare bu yoh', speaker='Mark Djandiomerr')

    utterance_ok = Utterance(
        org_media_path=Path(
            'data/org/BKW-speaker-ids/Mandak/20161102_mandak.wav'),
        org_transcription_path=Path(
            'data/org/BKW-speaker-ids/Mandak/Mandak_MN.eaf'),
        prefix='Mandak_MN.xv@MN.5',
        start_time=23155, end_time=25965,
        text='Mani mandak karrulkngeyyo.', speaker='Margaret')

    utterances = [utterance_too_short, utterance_ok]

    assert utterance.remove_too_short(utterances) == [utterance_ok]

def test_remove_duplicates_same_time():
    from persephone.utterance import Utterance, remove_duplicates
    utter_a1 = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='test text', speaker='Unit tester'
    )

    utter_a2 = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='test text', speaker='Unit tester'
    )

    utter_b = Utterance(
        org_media_path=Path(
            'testb.wav'),
        org_transcription_path=Path(
            'testb.txt'),
        prefix='testb',
        start_time=1,
        end_time=2,
        text='testb text', speaker='Unit tester'
    )

    all_utterances = [utter_a1, utter_a2, utter_b]
    result = remove_duplicates(all_utterances)
    assert result
    assert len(result) == 2
    assert utter_b in result
    assert (utter_a1 in result or utter_a2 in result)

def test_utterance_durations():
    from persephone.utterance import Utterance, duration
    utter_a1 = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='test text', speaker='Unit tester'
    )

    duration_a1 = duration(utter_a1)
    assert duration_a1 == 2 - 1

    utter_a2 = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=15,
        text='test text', speaker='Unit tester'
    )
    duration_a2 = duration(utter_a2)
    assert duration_a2 == 15 - 1

    from persephone.utterance import total_duration
    utterance_group = [utter_a1, utter_a2]
    group_duration = total_duration(utterance_group)
    assert group_duration == duration_a1 + duration_a2

def test_make_speaker_utters():
    """Test that we can make an associative mapping between speak names
    and the utterances that they made"""

    from persephone.utterance import Utterance, make_speaker_utters
    utter_a = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='a text', speaker='a'
    )

    utter_b = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='b text', speaker='b'
    )

    utter_c = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=1,
        end_time=2,
        text='the first thing c said', speaker='c'
    )

    utter_c1 = Utterance(
        org_media_path=Path(
            'test.wav'),
        org_transcription_path=Path(
            'test.txt'),
        prefix='test',
        start_time=6,
        end_time=10,
        text='the second thing c said', speaker='c'
    )

    all_utterances = [utter_a, utter_b, utter_c, utter_c1]

    speakers_to_utterances = make_speaker_utters(all_utterances)
    assert 'a' in speakers_to_utterances
    assert 'b' in speakers_to_utterances
    assert 'c' in speakers_to_utterances
    assert 'NotValidSpeaker' not in speakers_to_utterances
    assert len(speakers_to_utterances['a']) == 1
    assert len(speakers_to_utterances['b']) == 1
    assert len(speakers_to_utterances['c']) == 2