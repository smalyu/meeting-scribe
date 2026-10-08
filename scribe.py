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
Each MAI word then gets the speaker by time overlap.

  uv run scribe.py <audio> [--name slug] [--terms "a,b"]
  uv run scribe.py names <meeting_dir> 2=Андрей 3=Мария
"""
import argparse, base64, hashlib, json, os, re, shutil, subprocess, sys, time, unicodedata
from array import array
from bisect import bisect_left
from collections import Counter
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
                out.write_text(r.text)
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

def render(d):
    t = json.loads((d / "transcript.json").read_text())
    who = lambda s: t["names"].get(str(s), f"Спикер {s}")
    blocks, talk = [], Counter()
    for u in t["utterances"]:  # labels mapped to one name merge into one paragraph
        talk[who(u["speaker"])] += u["end"] - u["start"]
        b = blocks[-1] if blocks else None
        if b and b["who"] == who(u["speaker"]) and u["start"] - b["start"] < 90:
            b["text"] += " " + u["text"]
        else:
            blocks.append({"start": u["start"], "who": who(u["speaker"]), "text": u["text"]})
    lines = [f"# {d.name}", "",
             f"Источник: `{t['source']}` · {hms(t['duration'])} · {t['engine']}", "",
             "Спикеры: " + ", ".join(f"{s} ({v / 60:.0f} мин)" if v >= 60 else f"{s} ({v:.0f} с)"
                                    for s, v in talk.most_common()), ""]
    lines += [f"**[{hms(b['start'])}] {b['who']}:** {b['text']}\n" for b in blocks]
    (d / "transcript.md").write_text("\n".join(lines))
    return d / "transcript.md"


def run(a):
    key = api_key()
    src = Path(a.audio).expanduser().resolve()
    st = src.stat()
    date = datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime)).strftime("%Y-%m-%d")
    slug = re.sub(r"\W+", "-", unicodedata.normalize("NFC", src.stem).lower()).strip("-")
    d = ROOT / "meetings" / f"{date}-{a.name or slug}"
    if not a.name and not d.exists():  # the folder may have been renamed after the first run: find it by the recording
        d = next((m.parent for m in ROOT.glob(f"meetings/{date}-*/transcript.json")
                  if f'"source": {json.dumps(str(src), ensure_ascii=False)}' in m.read_text()), d)
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
    log(f"{src.name}: {hms(dur)}, {len(spans)} chunk(s) -> {d.relative_to(ROOT)}")

    def job(name, s, e, model, **extra):
        out, mp3 = d / "raw" / f"{name}.json", work / f"{name}.mp3"
        if out.exists():
            log(f"  {name}: cached")
            return json.loads(out.read_text())
        if not mp3.exists():
            to_mp3(pcm(str(src), s, e - s), mp3)
        return transcribe(key, mp3, out, model, **extra)

    # OpenRouter rejects `keyterms` for both models; MAI takes a phrase list via Azure passthrough
    w_opts = {"provider": {"options": {"azure": {"phraseList": {"phrases": terms[:50]}}}}} if terms else {}
    shift = lambda ws, s: [w | {"start": w["start"] + s, "end": w["end"] + s} for w in ws if is_word(w)]
    sspans = [(max(s - OVERLAP, 0), e) for s, e in spans]
    with ThreadPoolExecutor(8) as ex:
        # cache files are named by chunk start and end, so a changed plan never reuses a stale response
        fw = [ex.submit(job, f"words_{s:.0f}-{e:.0f}", s, e, WORDS_MODEL, **w_opts) for s, e in spans]
        fs = [ex.submit(job, f"speakers_{s:.0f}-{e:.0f}", s, e, SPK_MODEL, diarize=True) for s, e in sspans]
        spk = [shift(f.result()["words"], s) for (s, _), f in zip(sspans, fs)]
        spk_words = link(spk, [s for s, _ in spans])
        words = [w for w in spk_words if w["start"] < spans[0][0]]  # a skipped lead-in may be quiet speech: Scribe words
        engine = f"слова: {WORDS_MODEL}, спикеры: {SPK_MODEL}"
        for (s, e), f, sw in zip(spans, fw, spk):
            try:
                words += shift(f.result()["words"], s)
            except RuntimeError as err:  # API failure after retries (transcribe)
                log(f"WARNING: words {hms(s)}-{hms(e)} failed ({err}); using {SPK_MODEL} words there")
                words += [w for w in sw if s <= w["start"] < e]
                engine += f" (кусок {hms(s)}-{hms(e)}: только {SPK_MODEL})"

    turns = turns_of(spk_words)
    sig = hashlib.sha1(" ".join(t[2] for t in turns).encode()).hexdigest()[:12]
    utts = utterances(assign(words, turns))
    t = {"source": str(src), "duration": dur, "engine": engine, "speakers": sig, "names": {}, "utterances": utts}
    old = d / "transcript.json"
    if old.exists():  # names stay valid while the speaker labels are the same
        o = json.loads(old.read_text())
        if o.get("speakers", sig) == sig:
            t["names"] = o["names"]
        elif o["names"]:
            log("names dropped: speaker labels changed, run `names` again")
    old.write_text(json.dumps(t, ensure_ascii=False, indent=1))
    shutil.rmtree(work)
    print(render(d))
    print(f"{len(words)} words, {len(utts)} utterances, {len({u['speaker'] for u in utts})} speakers")


def names(a):
    d = Path(a.dir).resolve()
    t = json.loads((d / "transcript.json").read_text())
    pairs = dict(p.partition("=")[::2] for p in a.pairs)
    known = {str(u["speaker"]) for u in t["utterances"]}
    if bad := pairs.keys() - known:
        sys.exit(f"no such speaker(s): {', '.join(bad)}; known: {', '.join(sorted(known, key=int))}")
    t["names"] = {k: v for k, v in (t["names"] | pairs).items() if v}  # "N=" resets to «Спикер N»
    (d / "transcript.json").write_text(json.dumps(t, ensure_ascii=False, indent=1))
    print(render(d))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if sys.argv[1:2] == ["names"]:
        p.add_argument("cmd", metavar="names")
        p.add_argument("dir")
        p.add_argument("pairs", nargs="+", metavar="N=Имя")
        names(p.parse_args())
    else:
        p.add_argument("audio")
        p.add_argument("--name", help="slug for meetings/<date>-<slug>")
        p.add_argument("--terms", help="comma-separated names/terms for MAI's phrase list (first 50 used)")
        run(p.parse_args())
