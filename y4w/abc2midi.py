"""Zero-dependency ABC notation -> MIDI (SMF) conversion for YuE2 scores.

YuE2 symbolic planning emits a small two-voice ABC dialect::

    X:1
    M:4/4
    L:1/16
    Q:1/4=88
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:C
    % verse
    V: Vocal "C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|
    V: Ins Z4|

``V: Vocal`` carries the singing melody (with optional quoted chord symbols),
``V: Ins`` carries the instrumental/accompaniment line.  This module parses
that dialect with the Python standard library only (no ``music21`` / mido /
external ``abc2midi`` binary) so it works inside the Windows portable bundle
and on CI runners, and writes **one standard MIDI file per voice**:

    * ``vocal.mid``         - the singing melody track
    * ``accompaniment.mid`` - the instrumental track

The writer produces a compliant SMF (format 1, one track) with tempo, time
signature, key signature, program change and note on/off events.

Public API
----------
``parse_abc(text)``
    Parse ABC text into a :class:`ParsedScore` (header + per-voice notes).
``abc_to_midi(text)``
    Convert ABC text into :class:`AbcConversionResult` holding raw MIDI bytes
    for both voices plus metadata (note counts, duration, warnings).
``abc_to_midi_files(text, out_dir)``
    Same, but write ``vocal.mid`` / ``accompaniment.mid`` to ``out_dir``.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field

__all__ = [
    "MidiNote",
    "ParsedScore",
    "AbcConversionResult",
    "parse_abc",
    "render_voice_midi",
    "abc_to_midi",
    "abc_to_midi_files",
]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class MidiNote:
    """A single note with absolute tick positions (PPQ ticks)."""

    pitch: int
    start: int
    duration: int
    velocity: int = 100

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass
class ParsedScore:
    """Parsed ABC score: header fields plus notes grouped by voice name."""

    title: str = ""
    meter: tuple[int, int] = (4, 4)
    unit_num: int = 1
    unit_den: int = 16
    tempo_bpm: float = 120.0
    key: str = "C"
    voices: dict[str, list[MidiNote]] = field(default_factory=dict)
    voice_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def unit_ticks(self, ppq: int = 480) -> float:  # pragma: no cover - helper
        return 4.0 * ppq * self.unit_num / self.unit_den

    def total_ticks(self, ppq: int = 480) -> int:
        last = 0
        for notes in self.voices.values():
            for n in notes:
                last = max(last, n.end)
        return last


@dataclass
class AbcConversionResult:
    """Result of an ABC -> two-MIDI conversion.

    ``vocal_midi`` / ``accompaniment_midi`` are raw SMF bytes; callers that
    need text (REST API) can base64-encode them.
    """

    vocal_midi: bytes
    accompaniment_midi: bytes
    voices: list[str]
    vocal_note_count: int
    accompaniment_note_count: int
    duration_seconds: float
    title: str
    tempo_bpm: float
    meter: str
    key: str
    warnings: list[str]

    def to_dict(self, *, base64: bool = False) -> dict:
        """Serialize to a JSON-friendly dict (optionally base64 MIDI bytes)."""
        import base64

        def _enc(data: bytes):
            return base64.b64encode(data).decode("ascii") if base64 else list(data)

        return {
            "voices": self.voices,
            "vocal_note_count": self.vocal_note_count,
            "accompaniment_note_count": self.accompaniment_note_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "title": self.title,
            "tempo_bpm": self.tempo_bpm,
            "meter": self.meter,
            "key": self.key,
            "warnings": self.warnings,
            "vocal_midi": _enc(self.vocal_midi),
            "accompaniment_midi": _enc(self.accompaniment_midi),
            "vocal_midi_b64": base64.b64encode(self.vocal_midi).decode("ascii"),
            "accompaniment_midi_b64": base64.b64encode(
                self.accompaniment_midi
            ).decode("ascii"),
        }


# --------------------------------------------------------------------------- #
# Key signature helpers (circle of fifths)
# --------------------------------------------------------------------------- #
_MAJOR_FIFTHS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
_MINOR_FIFTHS = {
    "A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
    "D": -1, "G": -2, "C": -3, "F": -4, "Bb": -5, "Eb": -6, "Ab": -7,
}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"
_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _key_fifths(key: str) -> tuple[int, bool]:
    """Return (fifths, is_minor) for an ABC key string such as ``Cm``/``F#m``."""
    k = key.strip()
    if not k:
        return 0, False
    is_minor = False
    lower = k.lower()
    if lower.endswith("min") or lower.endswith("m") or "minor" in lower:
        is_minor = True
    # Extract root letter + optional accidental.
    root = ""
    for ch in k:
        if ch.isalpha() and ch.upper() in _PITCH_CLASS:
            root = ch.upper()
            break
    rest = k[k.index(root) + 1:] if root else ""
    acc = ""
    for ch in rest:
        if ch in "#b":
            acc += ch
            if acc == "bb":
                break
        elif ch.isalpha():
            break
    # Normalise "b" flats into the Bb/Eb/.. spelling used by the map.
    root_acc = root + ("b" if acc == "b" or acc == "bb" else "#" if acc == "#" else "")
    table = _MINOR_FIFTHS if is_minor else _MAJOR_FIFTHS
    if root_acc in table:
        return table[root_acc], is_minor
    if root in table:
        return table[root], is_minor
    return 0, is_minor


def _key_accidentals(fifths: int) -> dict[str, int]:
    acc: dict[str, int] = {}
    if fifths > 0:
        for i in range(min(fifths, 7)):
            acc[_SHARP_ORDER[i]] = 1
    elif fifths < 0:
        for i in range(min(-fifths, 7)):
            acc[_FLAT_ORDER[i]] = -1
    return acc


# --------------------------------------------------------------------------- #
# ABC body scanner
# --------------------------------------------------------------------------- #
def _strip_comment(line: str) -> str:
    """Strip ABC ``%`` comments (but not a percent inside a quoted chord)."""
    out: list[str] = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        if ch == "%" and not in_quote:
            break
        out.append(ch)
    return "".join(out)


def _parse_duration(s: str, i: int) -> tuple[float, int]:
    """Parse a note duration starting at ``i``.

    Returns (duration_in_L_units, new_index).  ``E2`` -> 2 units,
    ``E`` -> 1, ``E/2`` -> 0.5, ``E3/2`` -> 1.5, ``E//`` -> 0.25.
    """
    start = i
    numerator = ""
    while i < len(s) and s[i].isdigit():
        numerator += s[i]
        i += 1
    n = int(numerator) if numerator else 1
    if i < len(s) and s[i] == "/":
        i += 1
        denom = ""
        while i < len(s) and s[i].isdigit():
            denom += s[i]
            i += 1
        if denom:
            return n / int(denom), i
        # Each additional slash halves again: E/ = 1/2, E// = 1/4.
        d = 2
        while i < len(s) and s[i] == "/":
            d *= 2
            i += 1
        return n / d, i
    if i == start:  # no digits / slash consumed -> implicit single unit
        return 1.0, i
    return float(n), i


def _parse_music(
    text: str,
    notes: list[MidiNote],
    cursor: int,
    unit_ticks: float,
    key_acc: dict[str, int],
    warnings: list[str],
) -> int:
    """Parse one voice's music string; append notes; return new tick cursor."""
    i = 0
    tie_pending = False
    broken = 1.0  # broken-rhythm factor from '<' / '>'
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':  # chord symbol - ignore for the note track
            i += 1
            while i < n and text[i] != '"':
                i += 1
            if i < n:
                i += 1
            continue
        if ch == "!":  # decoration !trill!
            i += 1
            while i < n and text[i] != "!":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == "{":  # grace notes - skip the group
            i += 1
            while i < n and text[i] != "}":
                i += 1
            if i < n:
                i += 1
            continue
        if ch in "[|]":  # chord bracket / barline - structural only
            i += 1
            continue
        if ch == "(":  # tuplet opening - skip "(3"
            i += 1
            while i < n and text[i].isdigit():
                i += 1
            continue
        if ch == ")":
            i += 1
            continue
        if ch in "<>":
            broken = 1.5 if ch == ">" else 0.5
            i += 1
            continue
        if ch == "-":  # tie: extend the previous note of the same pitch
            tie_pending = True
            i += 1
            continue

        # Explicit accidental (^ #, _ flat, = natural) - may be doubled.
        explicit: int | None = None
        if ch in "^_=":
            if ch == "^":
                explicit = 1
            elif ch == "_":
                explicit = -1
            else:
                explicit = 0
            i += 1
            while i < n and text[i] in "^_=":
                t = text[i]
                delta = 1 if t == "^" else -1 if t == "_" else 0
                explicit = (explicit or 0) + delta
                i += 1

        # Rests.
        if i < n and text[i] in "zZ":
            r = text[i]
            i += 1
            units, i = _parse_duration(text, i)
            ticks = int(round(units * unit_ticks * broken))
            if r == "Z":  # whole-measure rest (YuE2 writes Z4 for full bars)
                ticks = int(round(units * 4 * unit_ticks))
            cursor += ticks
            broken = 1.0
            tie_pending = False
            continue

        # A note letter.
        if i < n and text[i].upper() in _PITCH_CLASS and text[i].isalpha():
            letter = text[i].upper()
            is_lower = text[i].islower()
            i += 1
            octave = 5 if is_lower else 4
            while i < n and text[i] in ",'":
                if text[i] == ",":
                    octave -= 1
                else:
                    octave += 1
                i += 1
            units, i = _parse_duration(text, i)
            alter = explicit if explicit is not None else key_acc.get(letter, 0)
            pitch = _PITCH_CLASS[letter] + 12 * (octave + 1) + alter
            pitch = max(0, min(127, pitch))
            ticks = max(1, int(round(units * unit_ticks * broken)))
            if tie_pending and notes and notes[-1].pitch == pitch:
                notes[-1].duration += ticks
            else:
                notes.append(MidiNote(pitch=pitch, start=cursor, duration=ticks))
            cursor += ticks
            tie_pending = False
            broken = 1.0
            continue

        if explicit is not None:
            warnings.append(f"未识别的临时记号后字符: {text[i:i+6]!r}")
        i += 1  # unknown char -> skip
    return cursor


# --------------------------------------------------------------------------- #
# Score parser
# --------------------------------------------------------------------------- #
def _split_first_tune(text: str) -> tuple[str, bool]:
    """Return the first tune and whether more tunes were present."""
    blocks: list[list[str]] = []
    current: list[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if line.lstrip().startswith("X:"):
            if started:
                blocks.append(current)
                current = []
            started = True
        current.append(line)
    if started:
        blocks.append(current)
        more = len(blocks) > 1
        return "\n".join(blocks[0]), more
    return text, False


_HEADER_FIELDS = {"X", "T", "M", "L", "Q", "K", "N", "S", "B", "Z", "P", "U"}


def parse_abc(text: str, *, ppq: int = 480) -> ParsedScore:
    """Parse ABC text into a :class:`ParsedScore`.

    Only the first tune is used if the text contains several.  Voices are
    discovered from ``V:`` lines; note positions are accumulated per voice.
    """
    text, more = _split_first_tune(text)
    score = ParsedScore()
    if more:
        score.warnings.append("检测到多首曲子，仅转换第一首 (X:1)")

    # First pass: header fields.
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in _HEADER_FIELDS and key != "V":
            continue
        if key == "T" and not score.title:
            score.title = value
        elif key == "M":
            try:
                num, den = value.split("/", 1)
                score.meter = (int(num), int(den))
            except (ValueError, IndexError):
                score.warnings.append(f"无法解析拍号 M:{value}")
        elif key == "L":
            try:
                num, den = value.split("/", 1)
                score.unit_num, score.unit_den = int(num), int(den)
            except (ValueError, IndexError):
                score.warnings.append(f"无法解析单位音符 L:{value}")
        elif key == "Q":
            # Forms: 1/4=88, 88, "1/4=120"
            try:
                bpm = float(value.split("=")[-1].strip())
                if 0 < bpm < 1000:
                    score.tempo_bpm = bpm
            except ValueError:
                score.warnings.append(f"无法解析速度 Q:{value}")
        elif key == "K":
            score.key = value
    fifths, is_minor = _key_fifths(score.key)

    # Second pass: voices / music.  Header V: definitions contain clef=/name=;
    # body V: lines carry actual music.
    score.voices = {}
    score.voice_order = []
    cursor_by_voice: dict[str, int] = {}
    current_voice: str | None = None

    def ensure_voice(name: str) -> None:
        if name not in score.voices:
            score.voices[name] = []
            score.voice_order.append(name)
            cursor_by_voice[name] = 0

    in_header = True
    for raw in text.splitlines():
        line = _strip_comment(raw)
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            fld = stripped.split(":", 1)[0].strip()
        else:
            fld = ""

        if fld == "K":
            in_header = False
            continue
        if in_header:
            continue

        if fld == "V":
            _, rest = stripped.split(":", 1)
            rest = rest.strip()
            if not rest:
                continue
            name = rest.split()[0]
            tail = rest[len(name):]
            is_definition = ("clef=" in tail or "snm=" in tail or "name=" in tail)
            if is_definition and not tail.strip().startswith(('"', "z", "Z")):
                # Voice declaration, not music.
                ensure_voice(name)
                current_voice = name
                continue
            ensure_voice(name)
            current_voice = name
            music = tail.strip()
            if music:
                cursor_by_voice[name] = _parse_music(
                    music, score.voices[name], cursor_by_voice[name],
                    4.0 * ppq * score.unit_num / score.unit_den,
                    _key_accidentals(fifths), score.warnings,
                )
            continue

        # Music continuation line (no V: prefix) belongs to current voice.
        if current_voice and fld == "":
            cursor_by_voice[current_voice] = _parse_music(
                stripped, score.voices[current_voice], cursor_by_voice[current_voice],
                4.0 * ppq * score.unit_num / score.unit_den,
                _key_accidentals(fifths), score.warnings,
            )

    if not score.voices:
        score.warnings.append("未找到任何 V: 声部；无法生成 MIDI")
    return score


# --------------------------------------------------------------------------- #
# MIDI (SMF) writer - standard library only
# --------------------------------------------------------------------------- #
def _vlq(value: int) -> bytes:
    """Encode a non-negative int as a MIDI variable-length quantity."""
    if value < 0:
        value = 0
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.reverse()
    return bytes(out)


def _meta(mtype: int, payload: bytes) -> bytes:
    return bytes([0xFF, mtype]) + _vlq(len(payload)) + payload


def render_voice_midi(
    notes: list[MidiNote],
    *,
    ppq: int = 480,
    tempo_bpm: float = 120.0,
    meter: tuple[int, int] = (4, 4),
    key: str = "C",
    track_name: str = "Voice",
    program: int = 0,
) -> bytes:
    """Render one voice's notes to a standard MIDI file (format 1, 1 track)."""
    fifths, is_minor = _key_fifths(key)
    events: list[tuple[int, int, bytes]] = []  # (tick, priority, bytes)

    # --- meta events at tick 0 ---
    events.append((0, 0, _meta(0x03, track_name.encode("utf-8", "replace"))))
    us_per_quarter = int(round(60_000_000 / max(1.0, tempo_bpm)))
    events.append((0, 0, _meta(0x51, struct.pack(">I", us_per_quarter)[1:])))
    num, den = meter
    dd = int(round(math.log2(den))) if den > 0 else 2
    events.append((0, 0, _meta(0x58, bytes([num & 0xFF, max(0, dd), 24, 8]))))
    events.append((0, 0, _meta(0x59, struct.pack(">bB", fifths, 1 if is_minor else 0))))
    events.append((0, 1, bytes([0xC0, program & 0x7F])))  # program change

    # --- notes ---
    for note in notes:
        vel = max(1, min(127, note.velocity))
        pitch = max(0, min(127, note.pitch))
        events.append((note.start, 2, bytes([0x90, pitch, vel])))     # note on
        events.append((note.end, 1, bytes([0x80, pitch, 0])))         # note off

    end_tick = max((e[0] for e in events), default=0)
    events.append((end_tick, 3, _meta(0x2F, b"")))  # end of track

    # Note-off (priority 1) before note-on (priority 2) at the same tick.
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    prev_tick = 0
    for tick, _prio, payload in events:
        track += _vlq(tick - prev_tick) + payload
        prev_tick = tick
    track_bytes = bytes(track)

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 1, ppq)
    track_chunk = b"MTrk" + struct.pack(">I", len(track_bytes)) + track_bytes
    return header + track_chunk


# --------------------------------------------------------------------------- #
# High-level conversion
# --------------------------------------------------------------------------- #
_VOCAL_ALIASES = ("vocal", "voice", "vocal melody", "singer", "人声", "主旋律", "melody")


def _classify_voices(score: ParsedScore) -> tuple[list[str], list[str]]:
    """Split discovered voices into (vocal, accompaniment) name lists."""
    vocal: list[str] = []
    accomp: list[str] = []
    for name in score.voice_order:
        low = name.lower()
        if any(alias in low for alias in _VOCAL_ALIASES):
            vocal.append(name)
        else:
            accomp.append(name)
    if not vocal and score.voice_order:
        # No explicit vocal name: treat the first voice as the melody.
        vocal.append(score.voice_order[0])
        accomp = [n for n in score.voice_order[1:]]
    return vocal, accomp


def abc_to_midi(
    text: str,
    *,
    ppq: int = 480,
    vocal_track_name: str = "Vocal",
    accomp_track_name: str = "Accompaniment",
) -> AbcConversionResult:
    """Convert ABC ``text`` into one MIDI file per voice.

    Returns :class:`AbcConversionResult` with raw SMF bytes for the vocal and
    accompaniment tracks.  Always produces both files (an empty voice yields a
    valid MIDI containing only metadata), so callers get a stable 2-file API.
    """
    score = parse_abc(text, ppq=ppq)
    vocal_names, accomp_names = _classify_voices(score)

    def collect(names: list[str]) -> list[MidiNote]:
        merged: list[MidiNote] = []
        for nm in names:
            merged.extend(score.voices.get(nm, []))
        merged.sort(key=lambda x: x.start)
        return merged

    v_notes = collect(vocal_names)
    a_notes = collect(accomp_names)
    common = dict(
        ppq=ppq,
        tempo_bpm=score.tempo_bpm,
        meter=score.meter,
        key=score.key,
    )
    v_midi = render_voice_midi(
        v_notes, track_name=vocal_track_name, program=54, **common
    )
    a_midi = render_voice_midi(
        a_notes, track_name=accomp_track_name, program=0, **common
    )
    total_ticks = max(score.total_ticks(ppq), ppq * 4)
    duration_seconds = total_ticks / ppq * (60.0 / max(1.0, score.tempo_bpm))
    num, den = score.meter
    return AbcConversionResult(
        vocal_midi=v_midi,
        accompaniment_midi=a_midi,
        voices=list(score.voice_order),
        vocal_note_count=len(v_notes),
        accompaniment_note_count=len(a_notes),
        duration_seconds=duration_seconds,
        title=score.title,
        tempo_bpm=score.tempo_bpm,
        meter=f"{num}/{den}",
        key=score.key,
        warnings=score.warnings,
    )


def abc_to_midi_files(
    text: str,
    out_dir: str,
    *,
    ppq: int = 480,
    vocal_filename: str = "vocal.mid",
    accompaniment_filename: str = "accompaniment.mid",
) -> dict:
    """Convert ABC text and write both MIDI files into ``out_dir``.

    Returns a dict with absolute paths and metadata.  ``out_dir`` is created
    if it does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = abc_to_midi(text, ppq=ppq)
    v_path = os.path.abspath(os.path.join(out_dir, vocal_filename))
    a_path = os.path.abspath(os.path.join(out_dir, accompaniment_filename))
    with open(v_path, "wb") as fh:
        fh.write(result.vocal_midi)
    with open(a_path, "wb") as fh:
        fh.write(result.accompaniment_midi)
    data = result.to_dict()
    data.update({
        "vocal_path": v_path,
        "accompaniment_path": a_path,
    })
    return data
