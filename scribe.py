#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Meeting audio -> Russian transcript with timestamps and speakers, via OpenRouter.

The recording is cut at quiet points into ~30 min chunks (API timeouts). Per chunk:
  words    - MAI-Transcribe-2 (best Russian WER), --terms and glossary.txt go to its phrase list
  speakers - Scribe v2 diarization
Speaker chunks overlap by 5 min; labels are linked through the words they share there.
Each MAI word then gets the speaker by time overlap. A Zoom Audio Recorder file also carries who
Zoom highlighted as speaking and where the user's own voice is: then those name the speakers.

  uv run scribe.py <audio> [--name slug] [--terms "a,b"]
  uv run scribe.py names <meeting_dir> 2=Андрей 3=Мария
  uv run scribe.py join <meeting_dir> <dir_2> ...   later recordings into <meeting_dir>/2/...: one transcript.md
  uv run scribe.py share <meeting_dir> [topic]   followup.md -> «Итоги встречи ДД.ММ — тема.md» w/o transcript links
"""
import argparse, base64, hashlib, json, os, re, shutil, subprocess, sys, time, unicodedata
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import groupby
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
API = "https://openrouter.ai/api/v1/audio/transcriptions"
WORDS_MODEL = "microsoft/mai-transcribe-2"
SPK_MODEL = "elevenlabs/scribe-v2"
CHUNK = 30 * 60  # seconds per request; whole 2.7 h file -> 524 timeout
OVERLAP = 5 * 60  # speaker chunks overlap the previous one by this much (for linking labels)
RETRY = {408, 429, 500, 502, 503, 504, 524, 529}
LAG = 0.5  # Zoom moves its speaker highlight this long after a new voice (median of 85 switches, 2026-10-12)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def hms(t):
    t = int(t)
    return f"{t // 3600:02}:{t % 3600 // 60:02}:{t % 60:02}"


def api_key():
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    return os.environ.get("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY: add it to .env")


# --- audio -------------------------------------------------------------------

def scan(p):
    """Decoded duration (container metadata can be missing or wrong) and long (5 s+) silences:
    MAI returns 500 on audio that starts with one."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", p, "-vn", "-af",
                        "aresample=first_pts=0,silencedetect=noise=-35dB:d=5", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")  # tags can be in any encoding
    t = re.findall(r"time=(\d+):(\d+):([\d.]+)", r.stderr)
    if r.returncode or not t:
        sys.exit(r.stderr.strip())
    num = lambda k: map(float, re.findall(rf"{k}: (-?[\d.]+)", r.stderr))
    h, m, s = t[-1]
    return int(h) * 3600 + int(m) * 60 + float(s), list(zip(num("silence_start"), num("silence_end")))


def pcm(p, a, d, rate=16000):
    """Mono s16 PCM bytes of [a, a+d). first_pts=0: audio that starts late in a video file keeps scan()'s timeline."""
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", p, "-af",
                           "aresample=first_pts=0", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                          stdout=subprocess.PIPE, check=True).stdout


def quietest(p, t, radius=120, win=0.6):
    """Center of the quietest `win`-second window within t±radius (relative, works in noisy rooms)."""
    a = max(t - radius, 0)
    x = array("h", pcm(p, a, 2 * radius, 8000))
    n = int(8000 * win)
    energy = [sum(v * v for v in x[i:i + n]) for i in range(0, len(x) - n, n // 4)]
    return a + (energy.index(min(energy)) * n // 4 + n / 2) / 8000 if energy else t


def plan(src, dur, sil):
    """Chunks of ~CHUNK seconds cut at quiet points; a chunk never starts with a long silence."""
    cuts = [0.0]
    while dur - cuts[-1] > CHUNK * 1.2:
        cuts.append(quietest(src, cuts[-1] + CHUNK))
    starts = []
    for a, b in zip(cuts, cuts[1:] + [dur]):
        for s, e in sil:  # in order: silencedetect can split one pause at a click
            if s <= a + 0.5 and a + 5 < e < b:
                a = e - 0.3
        starts.append(a)
    return list(zip(starts, starts[1:] + [dur]))  # a skipped silence stays in the previous chunk: it may be quiet speech


def to_mp3(raw, out):
    tmp = out.with_suffix(".tmp.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
                    "-c:a", "libmp3lame", "-b:a", "64k", str(tmp)], input=raw, check=True)
    tmp.rename(out)


# --- API ---------------------------------------------------------------------

def transcribe(key, mp3, out, model, **extra):
    body = {"model": model, "language": "ru", "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
            "input_audio": {"data": base64.b64encode(mp3.read_bytes()).decode(), "format": "mp3"}, **extra}
    for attempt in range(4):
        t0 = time.time()
        try:
            r = httpx.post(API, json=body, headers={"Authorization": f"Bearer {key}"},
                           timeout=httpx.Timeout(3600, connect=30))
            if r.status_code == 200 and "words" in (j := r.json()):  # [] is a valid answer: no speech
                tmp = out.with_suffix(".tmp")
                tmp.write_text(r.text)
                tmp.rename(out)  # a cached answer is always complete
                log(f"  {out.stem}: {len(j['words'])} words in {time.time() - t0:.0f}s")
                return j
            msg = f"HTTP {r.status_code}: {r.text.strip()[:500]}"
            if r.status_code in (401, 402):  # key or balance: every chunk would fail, so no fallback to Scribe words
                sys.exit(f"{out.stem}: {msg}")
            if r.status_code != 200 and r.status_code not in RETRY:
                break
        except (httpx.HTTPError, ValueError) as e:  # ValueError: 200 with a non-JSON body
            msg = repr(e)
        log(f"  {out.stem}: {msg} (attempt {attempt + 1})")
        if attempt < 3:
            time.sleep(15 * 2 ** attempt)
    raise RuntimeError(f"{out.stem}: {msg}")


# --- speakers ----------------------------------------------------------------

def is_word(w):
    return w.get("type", "word") == "word"


def label(w):
    s = w.get("speaker_label", w.get("speaker"))
    return None if s is None else str(s)


def turns_of(words):
    turns = []
    for w in words:
        if label(w) is None:
            continue
        if turns and turns[-1][2] == label(w) and w["start"] - turns[-1][1] < 2:
            turns[-1][1] = w["end"]
        else:
            turns.append([w["start"], w["end"], label(w)])
    return turns


def link(chunks, starts):
    """Global speaker labels S1, S2, ... Speaker chunk i starts OVERLAP s before MAI cut starts[i]: labels of
    chunks i-1 and i that sit on the same words there are one person (same audio, same context -> robust to
    room/channel drift that breaks voice-embedding linking). Unlinked labels get new ids."""
    out, n = [], 0
    for i, ws in enumerate(chunks):
        m = {None: None}
        if i:
            lo, hi = starts[i] - OVERLAP, starts[i]
            prev = [w for w in out if lo <= w["start"] < hi]
            co = Counter()
            for w in ws:
                if lo <= w["start"] < hi and label(w):
                    p = min(prev, key=lambda p: abs(p["start"] - w["start"]), default=None)
                    if p and abs(p["start"] - w["start"]) < 0.5 and label(p):
                        co[label(w), label(p)] += 1
            for (loc, g), c in co.most_common():
                if loc not in m and g not in m.values() and c >= 5:
                    m[loc] = g
        for w in ws:
            if label(w) not in m:
                n += 1
                m[label(w)] = f"S{n}"
        mid = starts[i] - OVERLAP / 2 if i else -1
        out = [w for w in out if w["start"] < mid] + [w | {"speaker_label": m[label(w)]} for w in ws if w["start"] >= mid]
    return out


def assign(words, turns):
    """Speaker with max time overlap (or nearest within 1 s); weak 1-2 word blips inside a turn are smoothed."""
    for w in words:  # MAI words can step back in time, so look turns up instead of keeping a running pointer
        best, score = None, -1.0
        for t in turns[bisect_left(turns, w["start"] - 1, key=lambda t: t[1]):]:
            if t[0] > w["end"] + 1:
                break
            o = min(w["end"], t[1]) - max(w["start"], t[0])
            if o > score:
                best, score = t[2], o
        w["spk"], w["fit"] = best, max(score, 0) / max(w["end"] - w["start"], 0.01)
    prev = next((w["spk"] for w in words if w["spk"]), None)
    for w in words:  # unlabeled words follow the previous speaker
        w["spk"] = prev = w["spk"] or prev
    runs = [list(g) for _, g in groupby(words, key=lambda w: w["spk"])]
    for p, r, n in zip(runs, runs[1:], runs[2:]):
        if len(r) <= 2 and p[0]["spk"] == n[0]["spk"] and all(w["fit"] < 0.5 for w in r):
            for w in r:
                w["spk"] = p[0]["spk"]
    return words


def tags(p):
    """Container tags of the recording ({} without any)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", p],
                           capture_output=True, text=True, errors="replace")  # tags can be in any encoding
        return json.loads(r.stdout)["format"].get("tags") or {}
    except (OSError, ValueError, KeyError, TypeError):  # OSError: no ffprobe
        return {}


def recorded(tags):
    """The creation_time tag as local time; Zoom Audio Recorder and iPhone Voice Memos set it when the recording
    ends. None without a real one (1904 is the MP4 epoch)."""
    try:
        t = datetime.fromisoformat(tags["creation_time"]).astimezone()
    except (KeyError, ValueError, OverflowError, OSError):
        return None
    return t if t.year >= 2000 else None


def zoom_timeline(tags):
    """Zoom Audio Recorder's timeline in the file's description tag: (seconds, name) whenever Zoom highlighted
    another speaker (name None: its meeting window was not visible), and [start, end] spans of the user's own
    voice, found by their microphone. None for other files."""
    try:
        t = json.loads(tags["description"])
        changes = sorted(((float(s), str(n) if n else None) for s, n in t["speakers"]), key=lambda c: c[0])
        me = []
        for a, b in sorted((float(a), float(b)) for a, b in t["me"]):  # merged: a word falls in one span at most
            if me and a <= me[-1][1]:
                me[-1][1] = max(me[-1][1], b)
            else:
                me.append([a, b])
        return changes, me
    except (ValueError, KeyError, TypeError):
        return None


def by_zoom(words, zoom):
    """Speakers named by Zoom instead of Scribe's labels: the user by their microphone, everyone else by Zoom's
    highlight. Where Zoom showed no one, a Scribe label takes the person it clearly matched elsewhere.
    Sets w["spk"] to "S<id>"; returns ({id: name}, the user's id, share of words named by Zoom or the microphone)."""
    changes, me = zoom
    mine = [s for s, _ in me]
    shown = {name for _, name in changes if name is not None}
    # One name lit all call long while the user spoke: the highlight never followed a voice. It may be the user, stuck
    # (2026-10-21, a 13-minute call: the user's name went to the other speaker's words), or the one other person in
    # speaker view; nothing tells which, so it names no one.
    if len(shown) == 1 and sum(e - s for s, e in me) >= 5:
        log(f"Zoom lit only {shown.pop()} all call long: highlight ignored, speakers by microphone and Scribe")
        changes = [(t, None) for t, _ in changes]
    ids = {}
    for _, name in changes:
        if name is not None:
            ids.setdefault(name, len(ids) + 1)
    ws = sorted(words, key=lambda w: w["start"])
    starts = [w["start"] for w in ws]

    def heard(w):
        mid = (w["start"] + w["end"]) / 2
        j = bisect_right(mine, mid + 0.15) - 1
        return j >= 0 and mid <= me[j][1] + 0.15

    mic, total, loud = [heard(w) for w in ws], Counter(), Counter()
    for w, m in zip(ws, mic):
        if w["spk"]:
            total[w["spk"]] += w["end"] - w["start"]
            loud[w["spk"]] += m * (w["end"] - w["start"])
    # Turns of the user's own Scribe labels: the microphone hears those 61%+ (10 Zoom calls), mixed ones 8-44%
    theirs = {s for s in total if loud[s] >= 0.5 * total[s]}
    begins = [w["start"] for i, w in enumerate(ws) if w["spk"] in theirs
              and (i == 0 or ws[i - 1]["spk"] != w["spk"] or w["start"] - ws[i - 1]["end"] >= 1)]

    def opens(t):
        """A highlight change at t follows a voice that began just as one of the user's spans or turns did. The
        microphone may catch a quiet start late (2026-10-27: 0.9 and 2.6 s late, so their name went to the other)."""
        j = bisect_right(mine, t - LAG + 0.5) - 1
        if j >= 0 and me[j][0] - 0.5 <= t - LAG <= min(me[j][1], me[j][0] + 1.5):
            return True
        k = bisect_left(begins, t - LAG - 1.5)
        return k < len(begins) and begins[k] <= t - LAG + 0.5

    # The user's name: the one Zoom switched to as they began to speak (gallery view highlights the user too,
    # speaker view never). The window reappearing is no new voice, and the next speaker's highlight comes later.
    votes = Counter(n for (_, p), (t, n) in zip(changes, changes[1:]) if p and n and opens(t))
    own, n = (votes.most_common(1) or [(None, 0)])[0]
    if n < 2 or n < 0.6 * votes.total():
        own = None
    own_id = ids[own] if own else len(ids) + 1 if me else None

    def voice_start(t):
        """The highlight lags the voice: the Scribe label change nearest to LAG before it, else the longest pause."""
        lo, hi = max(bisect_left(starts, t - 2), 1), bisect_left(starts, t + 0.3)
        if switches := [starts[i] for i in range(lo, hi) if ws[i]["spk"] != ws[i - 1]["spk"]]:
            return min(switches, key=lambda x: abs(x - (t - LAG)))
        gap, at = max(((starts[i] - ws[i - 1]["end"], starts[i]) for i in range(lo, hi)), default=(0, 0))
        return at if gap >= 0.2 else t - LAG

    segs, prev = [], None  # (from, id or None); the window appearing or hiding is no new voice
    for t, name in changes:
        at = voice_start(t) if prev and name else t
        segs.append((max(at, segs[-1][0]) if segs else at, ids.get(name)))
        prev = name
    froms = [s for s, _ in segs]
    tally = defaultdict(Counter)
    for w, m in zip(ws, mic):
        i = bisect_right(froms, w["start"]) - 1
        lit = segs[i][1] if i >= 0 else None
        # Zoom lighting the user up while their microphone is silent: quiet words the microphone missed, if it hears
        # this Scribe label at times (8 Zoom calls: the user's labels 61%+, mixed ones 8%+). A label it hardly ever
        # hears is another voice under a stuck highlight (2026-10-21: stuck on the user for 15 minutes; that label 2%).
        if lit == own_id and w["spk"] and loud[w["spk"]] < 0.05 * total[w["spk"]]:
            lit = None
        w["zoom"] = own_id if m else lit
        if w["spk"] and w["zoom"]:
            tally[w["spk"]][w["zoom"]] += w["end"] - w["start"]
    evidenced = sum(w["zoom"] is not None for w in ws)
    # A label takes a person on real evidence only: most of its named time, and enough of it. Otherwise its id
    # follows its Scribe number, so names stay put when another label changes.
    base, who, sure = max([*ids.values(), own_id or 0]), {}, {}
    for s in total:
        named = tally[s].total()
        best, d = (tally[s].most_common(1) or [(None, 0)])[0]
        # The microphone hears the user's own labels 61%+ (9 Zoom calls), mixed ones 8-44%. A label it hears less
        # than half is not the user's, even with no one else lit: its silent words are another voice (2026-10-21:
        # Scribe merged both speakers into one label, heard 42%).
        if best == own_id and loud[s] < 0.5 * total[s]:
            best = None
        who[s] = best if best and d >= 0.6 * named and named >= max(5, 0.3 * total[s]) else base + int(s[1:])
        if who[s] == best and d >= 0.85 * named:
            sure[s] = best
    # On a Zoom recording Scribe follows voices closely (every person comes through a different channel), while the
    # highlight lags and the microphone misses quiet words. So a label that is clearly one person names all its words,
    # unless the evidence clearly gives one of its turns to someone else.
    i = 0
    while i < len(ws):
        j = i + 1
        while j < len(ws) and ws[j]["spk"] == ws[i]["spk"] and ws[j]["start"] - ws[j - 1]["end"] < 2:
            j += 1
        if person := sure.get(ws[i]["spk"]):
            votes = Counter()
            for w in ws[i:j]:
                if w["zoom"]:
                    votes[w["zoom"]] += w["end"] - w["start"]
            other, d = (votes.most_common(1) or [(None, 0)])[0]
            person = other if other != person and d >= max(1, 0.7 * votes.total()) else person
            for w in ws[i:j]:
                w["zoom"] = person
        i = j
    for w in ws:
        sid = w.pop("zoom") or who.get(w["spk"])
        w["spk"] = f"S{sid}" if sid else None
    return {str(i): n for n, i in ids.items()}, own_id, evidenced / max(len(ws), 1)


def utterances(words):
    utts = []
    for w in words:  # numbered by global label, so names survive re-recognizing the words
        spk = int(w["spk"][1:]) if w["spk"] else 0
        u = utts[-1] if utts else None
        if u and u["speaker"] == spk and not (w["start"] - u["start"] > 90 and u["text"][-1:] in ".?!…"):
            u["text"] += " " + w["word"]
            u["end"] = max(u["end"], round(w["end"], 2))  # MAI words can step back in time
        else:
            utts.append({"start": round(w["start"], 2), "end": round(w["end"], 2), "speaker": spk, "text": w["word"]})
    return utts


# --- output ------------------------------------------------------------------

def parts(d):
    """Later recordings of the same meeting, moved into its folder by `join` as 2/, 3/, ..."""
    return sorted((p for p in d.iterdir() if (p / "transcript.json").exists()), key=lambda p: (len(p.name), p.name))


def render(d):
    """transcript.md of the meeting: one recording, or several with timecodes [N·ЧЧ:ММ:СС]."""
    if (d.parent / "transcript.json").exists():  # a later recording: the meeting's transcript shows it
        return render(d.parent)
    recs = [d] + parts(d)
    many, talk, sections = len(recs) > 1, Counter(), []
    for i, r in enumerate(recs, 1):
        t = json.loads((r / "transcript.json").read_text())
        # labels are per recording: an unnamed one carries its recording's number, «Спикер 2·1»
        who = lambda s: t["names"].get(str(s), f"Спикер {i}·{s}" if many else f"Спикер {s}")
        blocks = []
        for u in t["utterances"]:  # labels mapped to one name merge into one paragraph
            talk[who(u["speaker"])] += u["end"] - u["start"]
            b = blocks[-1] if blocks else None
            if b and b["who"] == who(u["speaker"]) and u["start"] - b["start"] < 90:
                b["text"] += " " + u["text"]
            else:
                blocks.append({"start": u["start"], "who": who(u["speaker"]), "text": u["text"]})
        meta = [f"`{t['source']}`", hms(t["duration"])]
        if t.get("recorded"):  # shown in this machine's time zone
            meta.append(f"{datetime.fromisoformat(t['recorded']).astimezone():%d.%m.%Y %H:%M %Z} (creation_time)")
        # note.md: what the recording missed, e.g. the user's account of an unrecorded start
        note = [(r / "note.md").read_text().strip(), ""] if (r / "note.md").exists() else []
        tc = f"{i}·" if many else ""
        sections.append(("Источник: " + " · ".join(meta + [t["engine"]]),
                         note + [f"**[{tc}{hms(b['start'])}] {b['who']}:** {b['text']}\n" for b in blocks]))
    who_talked = "Спикеры: " + ", ".join(f"{s} ({v / 60:.0f} мин)" if v >= 60 else f"{s} ({v:.0f} с)"
                                         for s, v in talk.most_common())
    lines = [f"# {d.name}", ""]
    if many:
        lines += [f"Записей: {len(recs)}. Таймкоды [N·ЧЧ:ММ:СС] — время от начала записи N.", "", who_talked, ""]
        for i, (source, body) in enumerate(sections, 1):
            lines += [f"## Запись {i}", "", source, ""] + body
    else:
        lines += [sections[0][0], "", who_talked, ""] + sections[0][1]
    (d / "transcript.md").write_text("\n".join(lines))
    return d / "transcript.md"


def run(a):
    key = api_key()
    src = Path(a.audio).expanduser().resolve()
    st = src.stat()
    tg = tags(str(src))
    at = recorded(tg)
    date = (at or datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))).strftime("%Y-%m-%d")
    slug = re.sub(r"\W+", "-", unicodedata.normalize("NFC", src.stem).lower()).replace(date, "")  # date once
    slug = re.sub(r"-{2,}", "-", slug).strip("-") or "meeting"
    d = ROOT / "meetings" / f"{date}-{a.name or slug}"
    # the folder may have been renamed after the first run: find it by the recording, so nothing is paid twice
    mine = f'"source": {json.dumps(str(src), ensure_ascii=False)}'

    def holds(m):  # this recording: the path and the size (the path may hold a re-exported file by now)
        size = m.parent / "raw" / "size"
        return mine in m.read_text() and (not size.exists() or size.read_text() == str(st.st_size))

    known = sorted(ROOT.glob("meetings/*/transcript.json")) + sorted(ROOT.glob("meetings/*/*/transcript.json"))
    prior = next((m.parent for m in known if holds(m)), None)
    if prior and (not a.name or prior.parent != ROOT / "meetings"):  # a joined recording stays in its meeting
        if a.name and prior.parent != ROOT / "meetings":
            log(f"--name ignored: {prior.relative_to(ROOT)} is a part of a joined meeting")
        d = prior
    elif prior and prior != d:  # a new slug for a known recording renames its folder, keeping its date
        if re.match(r"\d{4}-\d\d-\d\d-", prior.name):
            d = d.with_name(f"{prior.name[:10]}-{a.name}")
        if d != prior:
            if d.exists():
                sys.exit(f"{prior.relative_to(ROOT)} already holds this recording: move it or pick another --name")
            prior.rename(d)
            log(f"{prior.relative_to(ROOT)} -> {d.relative_to(ROOT)}")
    mark = d / "raw" / "size"  # same folder + another file (e.g. Zoom's audio_only.m4a) must not reuse the cache
    if mark.exists() and mark.read_text() != str(st.st_size):
        sys.exit(f"{d.relative_to(ROOT)} already holds another recording; pass another --name")
    dur, sil = scan(str(src))
    spans = plan(str(src), dur, sil)
    work = d / "work"
    work.mkdir(parents=True, exist_ok=True)
    mark.parent.mkdir(exist_ok=True)
    mark.write_text(str(st.st_size))
    glossary = ROOT / "glossary.txt"
    terms = (a.terms or "").split(",") + (glossary.read_text().splitlines() if glossary.exists() else [])
    terms = list(dict.fromkeys(x for x in map(str.strip, terms) if x and not x.startswith("#")))  # no duplicates
    zoom = zoom_timeline(tg)
    log(f"{src.name}: {hms(dur)}, {len(spans)} chunk(s) -> {d.relative_to(ROOT)}"
        + (f", recorded {at:%d.%m.%Y %H:%M %Z}" if at else "") + (", Zoom speakers" if zoom else ""))
    fresh = set()  # chunks requested (paid) in this run

    def job(name, s, e, model, **extra):
        out, mp3 = d / "raw" / f"{name}.json", work / f"{name}.mp3"
        if out.exists():
            log(f"  {name}: cached")
            return json.loads(out.read_text())
        fresh.add(name)
        if not mp3.exists():
            to_mp3(pcm(str(src), s, e - s), mp3)
        return transcribe(key, mp3, out, model, **extra)

    # OpenRouter rejects `keyterms` for both models; MAI takes a phrase list via Azure passthrough
    w_opts = {"provider": {"options": {"azure": {"phraseList": {"phrases": terms[:50]}}}}} if terms else {}
    shift = lambda ws, s: [w | {"start": w["start"] + s, "end": w["end"] + s} for w in ws if is_word(w)]
    sspans = [(max(s - OVERLAP, 0), e) for s, e in spans]
    # cache files are named by chunk start and end, so a changed plan never reuses a stale response
    wn, sn = [f"words_{s:.0f}-{e:.0f}" for s, e in spans], [f"speakers_{s:.0f}-{e:.0f}" for s, e in sspans]
    ex, fw, fs = ThreadPoolExecutor(8), [], []  # no `with`: Ctrl-C must not hang on the running requests

    def spent():
        done = [(n, f.result()) for n, f in zip(wn + sn, fw + fs) if f.done() and not f.cancelled() and not f.exception()]
        cost = lambda rs: sum(r.get("usage", {}).get("cost", 0) for r in rs)
        log(f"cost: ${cost(r for _, r in done):.2f} total, ${cost(r for n, r in done if n in fresh):.2f} this run")

    try:
        fw = [ex.submit(job, n, s, e, WORDS_MODEL, **w_opts) for n, (s, e) in zip(wn, spans)]
        fs = [ex.submit(job, n, s, e, SPK_MODEL, diarize=True) for n, (s, e) in zip(sn, sspans)]
        try:
            spk = [shift(f.result()["words"], s) for (s, _), f in zip(sspans, fs)]
        except RuntimeError as err:  # API failure after retries (transcribe): no speakers, no transcript
            ex.shutdown()  # the other requests are paid for: let them finish into the cache
            spent()
            sys.exit(f"{err}\nrerun the same command: finished chunks are cached in raw/")
        spk_words = link(spk, [s for s, _ in spans])
        words = [w for w in spk_words if w["start"] < spans[0][0]]  # a skipped lead-in may be quiet speech: Scribe words
        engine = f"слова: {WORDS_MODEL}, спикеры: {'Zoom и ' if zoom else ''}{SPK_MODEL}"
        for (s, e), f, sw in zip(spans, fw, spk):
            try:
                words += shift(f.result()["words"], s)
            except RuntimeError as err:  # API failure after retries (transcribe)
                log(f"WARNING: words {hms(s)}-{hms(e)} failed ({err}); using {SPK_MODEL} words there")
                words += [w for w in sw if s <= w["start"] < e]
                engine += f" (кусок {hms(s)}-{hms(e)}: только {SPK_MODEL})"
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)  # not yet sent: not paid
        if sent := sum(f.running() for f in fw + fs):
            log(f"interrupted: waiting for {sent} sent (paid) request(s) to land in raw/; Ctrl-C again drops them")
            try:
                ex.shutdown()
            except KeyboardInterrupt:
                pass
        os._exit(130)
    ex.shutdown()
    spent()
    if a.terms and not any(n.startswith("words_") for n in fresh):
        log("--terms had no effect: words came from the cache (delete raw/words_*.json to re-recognize)")

    turns = turns_of(spk_words)
    labels = " ".join(t[2] for t in turns) + (json.dumps(zoom, ensure_ascii=False) if zoom else "")
    sig = hashlib.sha1(labels.encode()).hexdigest()[:12]
    words, names = assign(words, turns), {}
    if zoom:
        names, own, share = by_zoom(words, zoom)
        print(f"Zoom and the microphone named {share:.0%} of words; own voice: "
              + (f"{names.get(str(own), f'Спикер {own}')} (speaker {own})" if own else "none"))
    utts = utterances(words)
    t = {"source": str(src), "duration": dur, "recorded": at.isoformat(timespec="minutes") if at else None,
         "engine": engine, "speakers": sig, "names": names, "utterances": utts}
    old = d / "transcript.json"
    if old.exists():  # names stay valid while the speaker labels are the same
        o = json.loads(old.read_text())
        if o.get("speakers", sig) == sig:
            t["names"] = o["names"]
        elif o["names"] and o["names"] != names:
            log("names dropped: speaker labels changed, run `names` again")
    old.write_text(json.dumps(t, ensure_ascii=False, indent=1))
    shutil.rmtree(work)
    print(render(d))
    print(f"{len(words)} words, {len(utts)} utterances, {len({u['speaker'] for u in utts})} speakers")


def meeting(path):
    d = Path(path).expanduser().resolve()
    if not (d / "transcript.json").exists():
        sys.exit(f"{d}: no transcript.json")
    return d


def names(a):
    d = meeting(a.dir)
    t = json.loads((d / "transcript.json").read_text())
    pairs = dict(p.partition("=")[::2] for p in a.pairs)
    known = {str(u["speaker"]) for u in t["utterances"]}
    if bad := pairs.keys() - known:
        sys.exit(f"no such speaker(s): {', '.join(bad)}; known: {', '.join(sorted(known, key=int))}")
    t["names"] = {k: v for k, v in (t["names"] | pairs).items() if v}  # "N=" resets to «Спикер N»
    (d / "transcript.json").write_text(json.dumps(t, ensure_ascii=False, indent=1))
    print(render(d))


def join(a):
    """Later recordings of one meeting go into its folder as 2/, 3/, ... with their cache: one transcript.md."""
    d = meeting(a.dir)
    if (d.parent / "transcript.json").exists():
        sys.exit(f"{d}: a part of a joined meeting; join into {d.parent}")
    for p in map(meeting, a.parts):
        if p == d or parts(p):
            sys.exit(f"{p}: cannot join a meeting into {d}")
        to = d / str(len(parts(d)) + 2)
        (p / "transcript.md").unlink(missing_ok=True)  # the meeting's transcript.md shows it now
        p.rename(to)
        log(f"{p.name} -> {d.name}/{to.name}")
    print(render(d))


def share(a):
    """followup.md -> «Итоги встречи ДД.ММ — <тема>.md» for the participants' chat: the heading gives the date and
    the topic, lines that link to transcripts are dropped (the participants have none)."""
    d = meeting(a.dir)
    if not (d / "followup.md").exists():
        sys.exit(f"{d / 'followup.md'}: not found")
    text = (d / "followup.md").read_text()
    m = re.match(r"# (.+) — итоги встречи (\d\d)\.(\d\d)\.\d{4}( \(\?\))?\s*$", text.partition("\n")[0])
    if not m:
        sys.exit("followup.md must start with «# <Тема> — итоги встречи ДД.ММ.ГГГГ»")
    topic, dd, mm, _ = m.groups()
    kept = []
    for line in text.splitlines():
        if re.search(r"\]\([^)]*transcript\.md\)", line):
            log(f"  dropped: {line}")
        else:
            kept.append(line)
    name = (a.topic or topic).replace(": ", " — ").replace(":", ".").replace("/", "-")
    out = d / f"Итоги встречи {dd}.{mm} — {name}.md"
    for old in d.glob("Итоги встречи *.md"):  # one copy, or an outdated one could get sent
        if old != out:
            old.unlink()
            log(f"  removed {old.name}")
    out.write_text(re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).rstrip() + "\n")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if sys.argv[1:2] == ["names"]:
        p.add_argument("cmd", metavar="names")
        p.add_argument("dir")
        p.add_argument("pairs", nargs="+", metavar="N=Имя")
        names(p.parse_args())
    elif sys.argv[1:2] == ["join"]:
        p.add_argument("cmd", metavar="join")
        p.add_argument("dir", help="the meeting's folder (its first recording)")
        p.add_argument("parts", nargs="+", metavar="dir", help="folders of its later recordings, in order")
        join(p.parse_args())
    elif sys.argv[1:2] == ["share"]:
        p.add_argument("cmd", metavar="share")
        p.add_argument("dir")
        p.add_argument("topic", nargs="?", help="topic for the file name when the heading's is too long")
        share(p.parse_args())
    else:
        p.add_argument("audio")
        p.add_argument("--name", help="slug for meetings/<date>-<slug>")
        p.add_argument("--terms", help="comma-separated names/terms for MAI's phrase list (first 50 used)")
        run(p.parse_args())
