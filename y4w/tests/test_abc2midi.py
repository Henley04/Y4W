"""Tests for the zero-dependency ABC -> MIDI converter.

No GPU / no optional deps required: these run everywhere including Windows CI.
"""
from __future__ import annotations

import struct

from y4w.abc2midi import (
    abc_to_midi,
    abc_to_midi_files,
    parse_abc,
    render_voice_midi,
)


SAMPLE = """X:1
T:Test Song
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal "C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|
V: Ins Z4|
% chorus
V: Vocal "C"G2A2c2B2A2G2E4|
V: Ins C2E2G2c2|G2E2C2G,2|
"""


def _is_smf(data: bytes) -> bool:
    # SMF header chunk is 8-byte header + 6-byte body = 14 bytes; MTrk
    # therefore starts at offset 14.
    return data[:4] == b"MThd" and data[14:18] == b"MTrk"


def _note_on_count(data: bytes) -> int:
    """Count note-on events (status 0x9n) ignoring running status/velocity bytes."""
    count = 0
    i = 22  # 14 header + 8 track chunk header
    end = 8 + 8 + struct.unpack(">I", data[18:22])[0]
    while i < end:
        # delta time (VLQ)
        while i < end and data[i] & 0x80:
            i += 1
        i += 1
        if i >= end:
            break
        status = data[i]
        if status == 0xFF:  # meta
            i += 2
            ln = 0
            while i < end and data[i] & 0x80:
                ln = (ln << 7) | (data[i] & 0x7F)
                i += 1
            ln = (ln << 7) | (data[i] & 0x7F)
            i += 1 + ln
            continue
        if status & 0xF0 == 0x90:
            if data[i + 2] > 0:
                count += 1
            i += 3
        elif status & 0xF0 == 0x80:
            i += 3
        elif status & 0xF0 == 0xC0:
            i += 2
        else:
            i += 1
    return count


def test_parse_header():
    s = parse_abc(SAMPLE)
    assert s.title == "Test Song"
    assert s.meter == (4, 4)
    assert (s.unit_num, s.unit_den) == (1, 16)
    assert s.tempo_bpm == 88.0
    assert s.key == "C"


def test_parse_two_voices():
    s = parse_abc(SAMPLE)
    assert set(s.voices) == {"Vocal", "Ins"}
    assert len(s.voices["Vocal"]) > 0
    assert len(s.voices["Ins"]) > 0


def test_note_pitches_are_midi():
    s = parse_abc(SAMPLE)
    assert all(0 <= n.pitch <= 127 for n in s.voices["Vocal"])
    assert all(n.duration > 0 for n in s.voices["Vocal"])
    # First vocal note in K:C with L:1/16 should be E4 = MIDI 64.
    assert s.voices["Vocal"][0].pitch == 64


def test_chord_symbols_are_ignored():
    s = parse_abc('X:1\nK:C\nV: Vocal "Cmaj7"E2 "Am"G2|\n')
    assert [n.pitch for n in s.voices["Vocal"]] == [64, 67]


def test_accidentals_and_octaves():
    s = parse_abc("X:1\nK:C\nV: Vocal ^C2 _E2 =F2 c2 C,2|\n")
    pitches = [n.pitch for n in s.voices["Vocal"]]
    assert pitches == [61, 63, 65, 72, 48]


def test_key_signature_applies():
    # In G major, F is sharped: F -> 66 (F#4)
    s = parse_abc("X:1\nK:G\nV: Vocal F2 G2|\n")
    assert s.voices["Vocal"][0].pitch == 66


def test_abc_to_midi_two_files():
    r = abc_to_midi(SAMPLE)
    assert _is_smf(r.vocal_midi)
    assert _is_smf(r.accompaniment_midi)
    assert r.voices == ["Vocal", "Ins"]
    assert r.vocal_note_count > 0
    assert r.accompaniment_note_count > 0
    assert r.vocal_note_count == _note_on_count(r.vocal_midi)
    assert r.accompaniment_note_count == _note_on_count(r.accompaniment_midi)
    assert 60.0 / 88.0 * 60 > r.duration_seconds > 0


def test_to_dict_base64():
    r = abc_to_midi(SAMPLE)
    d = r.to_dict(base64=True)
    assert d["vocal_midi_b64"]
    assert isinstance(d["vocal_midi"], str)
    assert d["meter"] == "4/4"
    assert d["key"] == "C"


def test_empty_voice_still_valid_midi():
    r = abc_to_midi("X:1\nK:C\nV: Vocal C2D2|\n")
    assert _is_smf(r.vocal_midi)
    assert _is_smf(r.accompaniment_midi)
    assert r.accompaniment_note_count == 0
    assert _note_on_count(r.accompaniment_midi) == 0


def test_files_written(tmp_path):
    data = abc_to_midi_files(SAMPLE, str(tmp_path))
    assert data["vocal_path"].endswith("vocal.mid")
    with open(data["vocal_path"], "rb") as fh:
        assert _is_smf(fh.read())
    with open(data["accompaniment_path"], "rb") as fh:
        assert _is_smf(fh.read())


def test_render_empty_midi():
    midi = render_voice_midi([], track_name="Empty")
    assert _is_smf(midi)
    assert _note_on_count(midi) == 0


def test_multiple_tunes_warns():
    s = parse_abc(SAMPLE + "\n\nX:2\nT:Second\nK:C\nV: Vocal C2|\n")
    assert any("多首" in w for w in s.warnings)
