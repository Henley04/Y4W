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

Supported ABC subset: notes (``C``-``B`` / ``c``-``b``), octave marks
(``,`` / ``'``), accidentals (``^ _ =`` and doubled forms), durations
(``C2``, ``C/2``, ``C3/2``, ``C//``), dotted notes (``C2.``), ties (``-``,
including across line breaks and bar lines), rests (``z`` and ``Z``
whole-measure rests), broken rhythm (``<`` / ``>`` and doubled forms),
chords (``[CEG]``), tuplets (``(3``), bar lines, quoted chord symbols
(ignored), ``%`` comments, ``M:``/``L:``/``Q:``/``K:`` fields (including
mid-tune changes), multi-voice ``V:`` declarations, and multiple tunes
(first tune only).

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

import base64 as _base64
import math
import os
import re
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
        """Serialize to a JSON-friendly dict.

        With ``base64=True`` the MIDI bytes are base64 strings; otherwise they
        are plain ``list[int]`` byte arrays.
        """

        def _enc(data: bytes):
            return _base64.b64encode(data).decode("ascii") if base64 else list(data)

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
            "vocal_midi_b64": _base64.b64encode(self.vocal_midi).decode("ascii"),
            "accompaniment_midi_b64": _base64.b64encode(
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
    """Return (fifths, is_minor) for an ABC key field value.

    Only the leading key token is considered, so decorated fields such as
    ``Am clef=treble`` are handled correctly.
    """
    k = key.strip().strip('"')
    if not k:
        return 0, False
    token = k.split()[0]  # drop trailing parameters (clef=..., etc.)
    low = token.lower()
    is_minor = low.endswith("min") or (low.endswith("m") and not low.endswith("dim"))

    root = ""
    root_idx = -1
    for idx, ch in enumerate(token):
        if ch.isalpha() and ch.upper() in _PITCH_CLASS:
            root = ch.upper()
            root_idx = idx
            break
    if not root:
        return 0, False

    # Accidental immediately after the root letter.
    acc = ""
    for ch in token[root_idx + 1:]:
        if ch in "#b":
            acc += ch
        elif ch.isalpha():
            break
    root_acc = root + ("b" if acc.startswith("b") else "#" if acc.startswith("#") else "")

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
# Header field helpers
# --------------------------------------------------------------------------- #
_ATTR_RE = re.compile(r'[A-Za-z_]+\s*=\s*(?:"[^"]*"|\S+)')
_MUSIC_RE = re.compile(r'[A-Ga-gzZ\[\]()]|[\^_=]')


def _parse_meter(value: str) -> tuple[int, int] | None:
    v = value.strip().strip('"').split()[0] if value.strip() else ""
    if v == "C":
        return (4, 4)
    if v in ("C|", "C|:"):
        return (2, 2)
    if "/" in v:
        a, b = v.split("/", 1)
        try:
            n, d = int(a), int(b)
            if n > 0 and d > 0:
                return (n, d)
        except ValueError:
            return None
    return None


def _parse_unit(value: str) -> tuple[int, int] | None:
    v = value.strip().strip('"').split()[0] if value.strip() else ""
    if "/" in v:
        a, b = v.split("/", 1)
        try:
            n, d = int(a), int(b)
            if n > 0 and d > 0:
                return (n, d)
        except ValueError:
            return None
    return None


def _parse_tempo(value: str) -> float | None:
    v = value.strip().strip('"').split()[0] if value.strip() else ""
    if not v:
        return None
    try:
        bpm = float(v.split("=")[-1].strip())
    except ValueError:
        return None
    return bpm if 0 < bpm < 1000 else None


# --------------------------------------------------------------------------- #
# ABC body scanner
# --------------------------------------------------------------------------- #
@dataclass
class _VoiceState:
    """Mutable per-voice parser state that must survive line breaks."""

    cursor: int = 0
    tie_pending: bool = False
    acc_state: dict[tuple[str, int], int] = field(default_factory=dict)
    tuplet_factor: float = 1.0
    tuplet_left: int = 0
    broken_next: float = 1.0


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
    """Parse a duration in L units starting at ``i``.

    Handles ``E2`` (2 units), ``E`` (1), ``E/2`` (0.5), ``E3/2`` (1.5),
    ``E//`` (0.25) and dotted forms (``E2.`` -> 3 units; each extra dot adds
    half of the previous increment).
    """
    start = i
    numerator = ""
    while i < len(s) and s[i].isdigit():
        numerator += s[i]
        i += 1
    n = int(numerator) if numerator else 1
    base = float(n)
    if i < len(s) and s[i] == "/":
        i += 1
        denom = ""
        while i < len(s) and s[i].isdigit():
            denom += s[i]
            i += 1
        if denom:
            base = n / int(denom)
        else:
            d = 2
            while i < len(s) and s[i] == "/":
                d *= 2
                i += 1
            base = n / d
    elif i == start:
        base = 1.0
    # Dotted notes: 1.5, 1.75, 1.875, ...
    dots = 0
    while i < len(s) and s[i] == ".":
        dots += 1
        i += 1
    if dots:
        factor = 1.0
        add = 0.5
        for _ in range(dots):
            factor += add
            add /= 2.0
        base *= factor
    return base, i


def _parse_music(
    text: str,
    notes: list[MidiNote],
    st: _VoiceState,
    ctx: dict,
    warnings: list[str],
) -> None:
    """Parse one voice's music string, appending notes and advancing ``st``."""
    i = 0
    n = len(text)
    chord_buf: list[tuple[int, int]] | None = None

    def _quantized(raw: float) -> int:
        factor = st.tuplet_factor * st.broken_next
        return max(1, int(round(raw * factor)))

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
        if ch == "|":  # bar line resets in-measure accidentals
            st.acc_state.clear()
            i += 1
            continue
        if ch == "[":
            i += 1
            if i < n and text[i].isdigit():  # [1 / [2 ending marker
                while i < n and text[i].isdigit():
                    i += 1
                continue
            chord_buf = []
            continue
        if ch == "]":
            i += 1
            if chord_buf is not None:
                dur_start = i
                units_after, i = _parse_duration(text, i)
                if i > dur_start:  # explicit duration written after "]"
                    dur = _quantized(units_after * ctx["unit_ticks"])
                    for pitch, _ in chord_buf:
                        notes.append(
                            MidiNote(pitch=pitch, start=st.cursor, duration=dur)
                        )
                    st.cursor += dur
                elif chord_buf:
                    max_dur = max(d for _, d in chord_buf)
                    for pitch, dur in chord_buf:
                        notes.append(
                            MidiNote(pitch=pitch, start=st.cursor, duration=dur)
                        )
                    st.cursor += max_dur
                chord_buf = None
            continue
        if ch == "(":  # tuplet: (3 -> 3 notes in the time of 2
            i += 1
            digits = ""
            while i < n and text[i].isdigit():
                digits += text[i]
                i += 1
            p = int(digits) if digits else 0
            q = p - 1 if p > 1 else 1
            if i < n and text[i] == ":":  # (p:q[:r] form
                i += 1
                qd = ""
                while i < n and text[i].isdigit():
                    qd += text[i]
                    i += 1
                q = int(qd) if qd else q
                while i < n and text[i] == ":":
                    i += 1
                    while i < n and text[i].isdigit():
                        i += 1
            if p > 1 and q > 0:
                st.tuplet_factor = q / p
                st.tuplet_left = p
            continue
        if ch == ")":
            i += 1
            continue
        if ch in "<>":
            # Broken rhythm: A>B lengthens A and shortens B.
            symbol = ch
            count = 0
            while i < n and text[i] == symbol:
                count += 1
                i += 1
            long_f = 2.0 - 0.5 ** count
            short_f = 0.5 ** count
            prev_f, next_f = (long_f, short_f) if symbol == ">" else (short_f, long_f)
            if notes and chord_buf is None:
                old = notes[-1].duration
                new = max(1, int(round(old * prev_f)))
                notes[-1].duration = new
                st.cursor += new - old
            st.broken_next = next_f
            continue
        if ch == "-":  # tie
            st.tie_pending = True
            i += 1
            continue

        # Explicit accidental (^ sharp, _ flat, = natural) - possibly doubled.
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
                explicit = (explicit or 0) + (1 if t == "^" else -1 if t == "_" else 0)
                i += 1

        # Rests.
        if i < n and text[i] in "zZ":
            r = text[i]
            i += 1
            units, i = _parse_duration(text, i)
            if r == "Z":
                # Whole-measure rest; a number multiplies measures (Z4 = 4 bars).
                ticks = max(1, int(round(units * ctx["bar_ticks"])))
            else:
                ticks = _quantized(units * ctx["unit_ticks"])
            st.cursor += ticks
            st.broken_next = 1.0
            st.tie_pending = False
            continue

        # A note letter.
        if i < n and text[i].isalpha() and text[i].upper() in _PITCH_CLASS:
            letter = text[i].upper()
            is_lower = text[i].islower()
            i += 1
            octave = 5 if is_lower else 4
            while i < n and text[i] in ",'":
                octave += 1 if text[i] == "'" else -1  # ' up, , down
                i += 1
            units, i = _parse_duration(text, i)
            key_alter = st.acc_state.get(
                (letter, octave), ctx["key_acc"].get(letter, 0)
            )
            alter = explicit if explicit is not None else key_alter
            if explicit is not None:
                st.acc_state[(letter, octave)] = explicit
            pitch = _PITCH_CLASS[letter] + 12 * (octave + 1) + alter
            pitch = max(0, min(127, pitch))
            dur = _quantized(units * ctx["unit_ticks"])

            if chord_buf is not None:
                chord_buf.append((pitch, dur))
            else:
                if st.tie_pending and notes and notes[-1].pitch == pitch:
                    notes[-1].duration += dur
                else:
                    notes.append(
                        MidiNote(pitch=pitch, start=st.cursor, duration=dur)
                    )
                st.cursor += dur
                st.tie_pending = False
            if st.tuplet_left:
                st.tuplet_left -= 1
                if st.tuplet_left == 0:
                    st.tuplet_factor = 1.0
            st.broken_next = 1.0
            continue

        if explicit is not None:
            warnings.append(f"未识别的临时记号后字符: {text[i:i + 6]!r}")
        i += 1  # unknown char -> skip


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


def parse_abc(text: str, *, ppq: int = 480) -> ParsedScore:
    """Parse ABC text into a :class:`ParsedScore`.

    Only the first tune is used if the text contains several.  Voices are
    discovered from ``V:`` lines; note positions are accumulated per voice and
    survive line breaks (so ties and accidentals work across lines).
    """
    text, more = _split_first_tune(text)
    score = ParsedScore()
    if more:
        score.warnings.append("检测到多首曲子，仅转换第一首 (X:1)")

    # --- pass 1: header fields ---
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if len(key) != 1 or not key.isalpha():
            continue
        if key == "T" and not score.title:
            score.title = value
        elif key == "M":
            m = _parse_meter(value)
            if m:
                score.meter = m
            else:
                score.warnings.append(f"无法解析拍号 M:{value}，使用 4/4")
        elif key == "L":
            u = _parse_unit(value)
            if u:
                score.unit_num, score.unit_den = u
            else:
                score.warnings.append(f"无法解析单位音符 L:{value}，使用 1/16")
        elif key == "Q":
            bpm = _parse_tempo(value)
            if bpm:
                score.tempo_bpm = bpm
            else:
                score.warnings.append(f"无法解析速度 Q:{value}")
        elif key == "K":
            score.key = value.split()[0] if value.split() else "C"

    if score.unit_den <= 0:
        score.warnings.append("单位音符长度非法，回退到 L:1/16")
        score.unit_num, score.unit_den = 1, 16
    if score.meter[0] <= 0 or score.meter[1] <= 0:
        score.warnings.append("拍号非法，回退到 M:4/4")
        score.meter = (4, 4)

    # --- pass 2: voices / music ---
    ctx = {
        "unit_ticks": 4.0 * ppq * score.unit_num / score.unit_den,
        "bar_ticks": 4.0 * ppq * score.meter[0] / score.meter[1],
        "key_acc": _key_accidentals(_key_fifths(score.key)[0]),
    }
    states: dict[str, _VoiceState] = {}
    current_voice: str | None = None
    in_header = True

    def ensure_voice(name: str) -> None:
        if name not in score.voices:
            score.voices[name] = []
            score.voice_order.append(name)
            states[name] = _VoiceState(cursor=0)

    for raw in text.splitlines():
        line = _strip_comment(raw)
        stripped = line.strip()
        if not stripped:
            continue

        fld = ""
        rest = ""
        if ":" in stripped:
            head, tail = stripped.split(":", 1)
            head = head.strip()
            # A field line has a single-letter name; things like "|:E2F2|"
            # must be treated as music, not as a field.
            if len(head) == 1 and head.isalpha():
                fld = head
                rest = tail.strip()

        if fld == "K":
            in_header = False
            key_token = rest.split()[0] if rest.split() else "C"
            ctx["key_acc"] = _key_accidentals(_key_fifths(key_token)[0])
            continue
        if fld == "M":
            m = _parse_meter(rest)
            if m:
                score.meter = m
                ctx["bar_ticks"] = 4.0 * ppq * m[0] / m[1]
            continue
        if fld == "L":
            u = _parse_unit(rest)
            if u:
                score.unit_num, score.unit_den = u
                ctx["unit_ticks"] = 4.0 * ppq * u[0] / u[1]
            continue
        if fld == "V":
            parts = rest.split(None, 1)
            name = parts[0] if parts else ""
            tail = parts[1] if len(parts) > 1 else ""
            if not name:
                continue
            ensure_voice(name)
            current_voice = name
            # Strip definition attributes (clef=..., name="..."), then parse
            # any music that remains on the same line.
            music = _ATTR_RE.sub(" ", tail).strip()
            if music and _MUSIC_RE.search(music):
                _parse_music(music, score.voices[name], states[name], ctx,
                             score.warnings)
            continue
        if fld:
            # Other field lines (T/Q/X/N/...) carry no music.
            continue

        # Music continuation line without a V: prefix.
        if current_voice and not in_header:
            _parse_music(stripped, score.voices[current_voice],
                         states[current_voice], ctx, score.warnings)

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
    us_per_quarter = min(0xFFFFFF, int(round(60_000_000 / max(1.0, tempo_bpm))))
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
_VOCAL_PREFIXES = (
    "vocal", "voice", "voc", "singer", "lead", "melody", "mel",
    "人声", "歌声", "主旋",
)
_ACCOMP_PREFIXES = (
    "ins", "inst", "instr", "acc", "accomp", "acom", "bass", "drum", "perc",
    "piano", "keys", "guitar", "chord", "pad", "strings", "伴奏", "器乐",
)


def _classify_voices(score: ParsedScore) -> tuple[list[str], list[str]]:
    """Split discovered voices into (vocal, accompaniment) name lists."""
    vocal: list[str] = []
    accomp: list[str] = []
    unknown: list[str] = []

    for name in score.voice_order:
        low = name.lower()
        if low.startswith(_VOCAL_PREFIXES):
            vocal.append(name)
        elif low.startswith(_ACCOMP_PREFIXES):
            accomp.append(name)
        else:
            unknown.append(name)

    if not vocal and unknown:
        # No explicitly named vocal voice: treat the first unclassified voice
        # as the melody, but never promote a known accompaniment voice.
        vocal.append(unknown.pop(0))
        score.warnings.append(
            f"未找到明确的歌声声部，已将声部 {vocal[0]!r} 作为旋律"
        )
    accomp.extend(unknown)

    if len(accomp) > 1:
        score.warnings.append(
            f"检测到多个伴奏声部 {accomp}，已合并到 accompaniment.mid"
        )
    if not vocal:
        score.warnings.append("未找到歌声声部，vocal.mid 将为空")
    return vocal, accomp


def _safe_target(out_dir: str, filename: str) -> str:
    """Resolve ``filename`` inside ``out_dir``, rejecting path traversal.

    Absolute paths and any path separator are rejected outright (rather than
    silently rewritten) so a mistyped CLI argument fails loudly.
    """
    if not filename or filename in (".", ".."):
        raise ValueError(f"非法文件名: {filename!r}")
    if os.path.isabs(filename) or "/" in filename or "\\" in filename:
        raise ValueError(f"文件名不得包含路径分隔符: {filename!r}")
    root = os.path.abspath(out_dir)
    target = os.path.abspath(os.path.join(root, filename))
    try:
        inside = os.path.commonpath([root, target]) == root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise ValueError(f"文件名越界: {filename!r}")
    return target


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
    if it does not exist; filenames are constrained to stay inside it.
    """
    os.makedirs(out_dir, exist_ok=True)
    v_path = _safe_target(out_dir, vocal_filename)
    a_path = _safe_target(out_dir, accompaniment_filename)
    result = abc_to_midi(text, ppq=ppq)
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
