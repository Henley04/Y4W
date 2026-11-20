"""Tests for the zero-dependency ABC -> MIDI converter.

No GPU / no optional deps required: these run everywhere including Windows CI.
"""
from __future__ import annotations

import struct

import pytest

from y4w.abc2midi import (
    _key_fifths,
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


def _track_payload_bounds(data: bytes) -> tuple[int, int]:
    """Return (track_start, track_end) for a single-track SMF."""
    assert data[14:18] == b"MTrk"
    length = struct.unpack(">I", data[18:22])[0]
    return 22, 22 + length


def _note_on_count(data: bytes) -> int:
    """Count sounding note-on events (status 0x9n, velocity > 0)."""
    count = 0
    i, end = _track_payload_bounds(data)
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


def _meta_payloads(data: bytes) -> list[tuple[int, bytes]]:
    """Return (meta_type, payload) pairs from the track."""
    out: list[tuple[int, bytes]] = []
    i, end = _track_payload_bounds(data)
    while i < end:
        while i < end and data[i] & 0x80:
            i += 1
        i += 1
        if i >= end:
            break
        if data[i] != 0xFF:
            status = data[i]
            if status & 0xF0 in (0x90, 0x80):
                i += 3
            elif status & 0xF0 == 0xC0:
                i += 2
            else:
                i += 1
            continue
        mtype = data[i + 1]
        i += 2
        ln = 0
        while i < end and data[i] & 0x80:
            ln = (ln << 7) | (data[i] & 0x7F)
            i += 1
        ln = (ln << 7) | (data[i] & 0x7F)
        i += 1
        out.append((mtype, data[i:i + ln]))
        i += ln
    return out


def _voice_notes(abc: str, voice: str = "Vocal"):
    return parse_abc(abc).voices[voice]


def _durations(abc: str, voice: str = "Vocal"):
    return [n.duration for n in _voice_notes(abc, voice)]


# --- header parsing ---------------------------------------------------------

def test_parse_header():
    s = parse_abc(SAMPLE)
    assert s.title == "Test Song"
    assert s.meter == (4, 4)
    assert (s.unit_num, s.unit_den) == (1, 16)
    assert s.tempo_bpm == 88.0
    assert s.key == "C"


def test_meter_symbols_and_quoted_tempo():
    assert parse_abc("X:1\nM:C\nK:C\nV: V C2|\n").meter == (4, 4)
    assert parse_abc("X:1\nM:C|\nK:C\nV: V C2|\n").meter == (2, 2)
    assert parse_abc('X:1\nQ:"1/4=132"\nK:C\nV: V C2|\n').tempo_bpm == 132


# --- voices -----------------------------------------------------------------

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


def test_inline_music_on_voice_definition_line():
    abc = 'X:1\nL:1/8\nK:C\nV: Vocal clef=treble name="V" "C"E2F2|\n'
    notes = _voice_notes(abc)
    assert [n.pitch for n in notes] == [64, 65]


def test_continuation_line_starting_with_bar_symbol():
    abc = "X:1\nL:1/8\nK:C\nV: Vocal C2D2|\n|:E2F2|\n"
    assert [n.pitch for n in _voice_notes(abc)] == [60, 62, 64, 65]


# --- pitch, accidentals, key signature --------------------------------------

def test_accidentals_and_octaves():
    s = parse_abc("X:1\nK:C\nV: Vocal ^C2 _E2 =F2 c2 C,2|\n")
    pitches = [n.pitch for n in s.voices["Vocal"]]
    assert pitches == [61, 63, 65, 72, 48]


def test_double_accidentals():
    assert parse_abc("X:1\nK:C\nV: Vocal ^^C2|\n").voices["Vocal"][0].pitch == 62
    assert parse_abc("X:1\nK:C\nV: Vocal __E2|\n").voices["Vocal"][0].pitch == 62


def test_accidental_persists_within_measure():
    pitches = [n.pitch for n in _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal ^F F F|\n")]
    assert pitches == [66, 66, 66]


def test_accidental_resets_at_bar_line():
    pitches = [n.pitch for n in _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal ^F|F|\n")]
    assert pitches == [66, 65]


def test_key_signature_applies():
    # In G major, F is sharped: F -> 66 (F#4)
    s = parse_abc("X:1\nK:G\nV: Vocal F2 G2|\n")
    assert s.voices["Vocal"][0].pitch == 66


def test_key_fifths_with_trailing_parameters():
    assert _key_fifths("Am clef=treble") == (0, True)
    assert _key_fifths("F#m") == (3, True)
    assert _key_fifths("Bb") == (-2, False)


def test_mid_tune_key_change():
    abc = "X:1\nL:1/8\nK:C\nV: Vocal F2|\nK:G\nV: Vocal F2|\n"
    assert [n.pitch for n in _voice_notes(abc)] == [65, 66]


# --- rhythm -----------------------------------------------------------------

def test_dotted_notes():
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C2. D|\n") == [720, 240]
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C..|\n") == [420]


def test_duration_forms():
    # L:1/8 -> 240 ticks per unit
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C C2 C/2 C3/2 C//|\n") == [
        240, 480, 120, 360, 60
    ]


def test_broken_rhythm_both_sides():
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C>D|\n") == [360, 120]
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C<D|\n") == [120, 360]
    assert _durations("X:1\nL:1/8\nK:C\nV: Vocal C>>D|\n") == [420, 60]


def test_tie_same_line():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal C4-C4|\n")
    assert [(n.pitch, n.duration) for n in notes] == [(60, 1920)]


def test_tie_across_lines():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal C4-\nC4|\n")
    assert [(n.pitch, n.duration) for n in notes] == [(60, 1920)]


def test_tie_across_bar_line():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal C4-|C4|\n")
    assert [(n.pitch, n.duration) for n in notes] == [(60, 1920)]


def test_tie_different_pitch_is_not_merged():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal C4-D4|\n")
    assert [(n.pitch, n.duration) for n in notes] == [(60, 960), (62, 960)]


def test_whole_measure_rest_follows_meter():
    # 4/4 with L:1/16 -> one bar = 1920 ticks
    assert _voice_notes("X:1\nM:4/4\nL:1/16\nK:C\nV: Vocal ZC2|\n")[0].start == 1920
    # 3/4 with L:1/8 -> one bar = 1440 ticks
    assert _voice_notes("X:1\nM:3/4\nL:1/8\nK:C\nV: Vocal ZC2|\n")[0].start == 1440
    # Z4 = four bars
    assert _voice_notes("X:1\nM:4/4\nL:1/16\nK:C\nV: Vocal Z4C2|\n")[0].start == 7680


def test_chord_notes_share_start():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal [CEG]2|\n")
    assert [(n.pitch, n.start, n.duration) for n in notes] == [
        (60, 0, 480), (64, 0, 480), (67, 0, 480)
    ]


def test_tuplet_scales_duration():
    notes = _voice_notes("X:1\nL:1/8\nK:C\nV: Vocal (3CDE D2|\n")
    assert [n.duration for n in notes] == [160, 160, 160, 480]
    assert notes[3].start == 480


# --- invalid input ----------------------------------------------------------

def test_zero_denominator_unit_does_not_crash():
    s = parse_abc("X:1\nL:1/0\nK:C\nV: Vocal C2|\n")
    assert any("单位音符" in w for w in s.warnings)
    assert s.unit_den == 16


def test_multiple_tunes_warns():
    s = parse_abc(SAMPLE + "\n\nX:2\nT:Second\nK:C\nV: Vocal C2|\n")
    assert any("多首" in w for w in s.warnings)


# --- MIDI rendering ---------------------------------------------------------

def test_abc_to_midi_two_files():
    r = abc_to_midi(SAMPLE)
    assert _is_smf(r.vocal_midi)
    assert _is_smf(r.accompaniment_midi)
    assert r.voices == ["Vocal", "Ins"]
    assert r.vocal_note_count > 0
    assert r.accompaniment_note_count > 0
    assert r.vocal_note_count == _note_on_count(r.vocal_midi)
    assert r.accompaniment_note_count == _note_on_count(r.accompaniment_midi)


def test_midi_meta_events():
    r = abc_to_midi(SAMPLE)
    metas = _meta_payloads(r.vocal_midi)
    types = [t for t, _ in metas]
    assert 0x51 in types  # tempo
    assert 0x58 in types  # time signature
    assert 0x59 in types  # key signature
    assert types[-1] == 0x2F  # end of track
    tempo = next(p for t, p in metas if t == 0x51)
    us = int.from_bytes(tempo, "big")
    assert abs(60_000_000 / us - 88) < 0.5
    ts = next(p for t, p in metas if t == 0x58)
    assert ts[0] == 4 and ts[1] == 2


def test_midi_key_signature_meta_for_minor():
    r = abc_to_midi("X:1\nK:Am\nV: Vocal A2|\n")
    ks = next(p for t, p in _meta_payloads(r.vocal_midi) if t == 0x59)
    assert ks[0] == 0 and ks[1] == 1


def test_long_delta_uses_multibyte_vlq():
    # A 4-bar rest pushes the first note past 127 ticks -> multi-byte delta.
    r = abc_to_midi("X:1\nM:4/4\nL:1/16\nK:C\nV: Vocal Z4C2|\n")
    assert _is_smf(r.vocal_midi)
    assert r.vocal_note_count == 1


def test_to_dict_plain_bytes():
    r = abc_to_midi(SAMPLE)
    d = r.to_dict(base64=False)
    assert isinstance(d["vocal_midi"], list)
    assert isinstance(d["accompaniment_midi"], list)
    assert d["vocal_midi"][:4] == [0x4D, 0x54, 0x68, 0x64]


def test_to_dict_base64():
    r = abc_to_midi(SAMPLE)
    d = r.to_dict(base64=True)
    assert d["vocal_midi_b64"].startswith("TVRo")
    assert isinstance(d["vocal_midi"], str)
    assert d["meter"] == "4/4"


def test_empty_voice_still_valid_midi():
    r = abc_to_midi("X:1\nK:C\nV: Vocal C2D2|\n")
    assert _is_smf(r.vocal_midi)
    assert _is_smf(r.accompaniment_midi)
    assert r.accompaniment_note_count == 0
    assert _note_on_count(r.accompaniment_midi) == 0


def test_only_ins_voice_is_not_promoted_to_vocal():
    r = abc_to_midi("X:1\nK:C\nV: Ins C2D2|\n")
    assert r.vocal_note_count == 0
    assert r.accompaniment_note_count == 2
    assert any("未找到歌声" in w for w in r.warnings)


def test_multiple_accompaniment_voices_warn():
    r = abc_to_midi("X:1\nK:C\nV: Vocal C2|\nV: Ins C2|\nV: Drums C2|\n")
    assert any("多个伴奏" in w for w in r.warnings)


def test_render_empty_midi():
    midi = render_voice_midi([], track_name="Empty")
    assert _is_smf(midi)
    assert _note_on_count(midi) == 0


# --- file output / path safety ---------------------------------------------

def test_files_written(tmp_path):
    data = abc_to_midi_files(SAMPLE, str(tmp_path))
    assert data["vocal_path"].endswith("vocal.mid")
    with open(data["vocal_path"], "rb") as fh:
        assert _is_smf(fh.read())
    with open(data["accompaniment_path"], "rb") as fh:
        assert _is_smf(fh.read())


@pytest.mark.parametrize("bad", ["../escape.mid", "/tmp/abs.mid", "sub/dir.mid"])
def test_path_traversal_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        abc_to_midi_files(SAMPLE, str(tmp_path), vocal_filename=bad)
    # Nothing escaped the output directory.
    assert not (tmp_path.parent / "escape.mid").exists()
