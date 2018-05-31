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