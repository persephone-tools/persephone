from pathlib import Path

from persephone import utterance
from persephone.utterance import Utterance

def test_too_short():
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
