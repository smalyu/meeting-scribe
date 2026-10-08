#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Meeting audio -> Russian transcript with timestamps and speakers, via OpenRouter.

The recording is cut at quiet points into ~30 min chunks (API timeouts). Per chunk:
  words    - MAI-Transcribe-2 (best Russian WER)
  speakers - Scribe v2 diarization
Speaker chunks overlap by 5 min; labels are linked through the words they share there.
Each MAI word then gets the speaker by time overlap.

  uv run scribe.py <audio> [--name slug] [--speakers N] [--terms "a,b"]
  uv run scribe.py names <meeting_dir> 1=Андрей 2=Мария
"""
import argparse, base64, json, os, re, shutil, subprocess, sys, time
from array import array
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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

def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def pcm(p, a, d, rate=16000):
    """Mono s16 PCM bytes of [a, a+d)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", p,
                           "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"], capture_output=True, check=True).stdout


def silences(p):
    """Long (5 s+) silences: MAI returns 500 on audio that starts with one."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", p, "-af", "silencedetect=noise=-35dB:d=5",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return list(zip(map(float, re.findall(r"silence_start: ([\d.]+)", err)),
                    map(float, re.findall(r"silence_end: ([\d.]+)", err))))


def quietest(p, t, radius=120, win=0.6):
    """Center of the quietest `win`-second window within t±radius (relative, works in noisy rooms)."""
    a = max(t - radius, 0)
    x = array("h", pcm(p, a, 2 * radius, 8000))
    n = int(8000 * win)
    energy = [sum(v * v for v in x[i:i + n]) for i in range(0, len(x) - n, n // 4)]
    return a + (energy.index(min(energy)) * n // 4 + n / 2) / 8000 if energy else t


def plan(src, dur):
    """Chunks of ~CHUNK seconds cut at quiet points; leading long silence skipped."""
    cuts = [0.0]
    while dur - cuts[-1] > CHUNK * 1.2:
        cuts.append(quietest(src, cuts[-1] + CHUNK))
    spans = []
    for a, b in zip(cuts, cuts[1:] + [dur]):
        for s, e in silences(src) if a == 0 else []:
            if s <= 0.5 and e > 5:
                a = e - 0.3
        spans.append((a, b))
    return spans


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
            if r.status_code == 200 and r.json().get("words"):
                out.write_text(r.text)
                log(f"  {out.stem}: ok in {time.time() - t0:.0f}s")
                return r.json()
            msg = f"HTTP {r.status_code}: {r.text.strip()[:500]}"
            if r.status_code != 200 and r.status_code not in RETRY:
                break
        except httpx.TransportError as e:
            msg = repr(e)
        log(f"  {out.stem}: {msg} (attempt {attempt + 1})")
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
    """Global speaker labels. Speaker chunk i starts OVERLAP s before MAI cut starts[i]: labels of chunks
    i-1 and i that sit on the same words there are one person (same audio, same context -> robust to
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
            used = set()
            for (loc, g), c in co.most_common():
                if loc not in m and g not in used and c >= 5:
                    m[loc], _ = g, used.add(g)
        for w in ws:
            if label(w) not in m:
                n += 1
                m[label(w)] = f"S{n}"
        mid = starts[i] - OVERLAP / 2 if i else -1
        out = [w for w in out if w["start"] < mid] + [w | {"speaker_label": m[label(w)]} for w in ws if w["start"] >= mid]
    return out


def assign(words, turns):
    """Speaker with max time overlap (or nearest within 1 s); weak 1-2 word blips inside a turn are smoothed."""
    j = 0
    for w in words:
        while j < len(turns) and turns[j][1] < w["start"] - 1:
            j += 1
        best, score = None, -1.0
        for t in turns[j:]:
            if t[0] > w["end"] + 1:
                break
            o = min(w["end"], t[1]) - max(w["start"], t[0])
            if o > score:
                best, score = t[2], o
        w["spk"], w["fit"] = best, max(score, 0) / max(w["end"] - w["start"], 0.01)
    prev = next((w["spk"] for w in words if w["spk"]), None)
    for w in words:  # unlabeled words follow the previous speaker
        w["spk"] = prev = w["spk"] or prev
    runs = []
    for w in words:
        if runs and runs[-1][0]["spk"] == w["spk"]:
            runs[-1].append(w)
        else:
            runs.append([w])
    for p, r, n in zip(runs, runs[1:], runs[2:]):
        if len(r) <= 2 and p[0]["spk"] == n[0]["spk"] and all(w["fit"] < 0.5 for w in r):
            for w in r:
                w["spk"] = p[0]["spk"]
    return words


def utterances(words):
    ids, utts = {}, []
    for w in words:
        spk = ids.setdefault(w["spk"], len(ids) + 1)
        u = utts[-1] if utts else None
        if u and u["speaker"] == spk and not (w["start"] - u["start"] > 90 and u["text"][-1:] in ".?!…"):
            u["text"] += " " + w["word"]
            u["end"] = w["end"]
        else:
            utts.append({"start": round(w["start"], 2), "end": w["end"], "speaker": spk, "text": w["word"]})
    return utts


# --- output ------------------------------------------------------------------

def render(d):
    t = json.loads((d / "transcript.json").read_text())
    who = lambda s: t.get("names", {}).get(str(s), f"Спикер {s}")
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
             "Спикеры: " + ", ".join(f"{s} ({v / 60:.0f} мин)" for s, v in talk.most_common()), ""]
    lines += [f"**[{hms(b['start'])}] {b['who']}:** {b['text']}\n" for b in blocks]
    (d / "transcript.md").write_text("\n".join(lines))
    return d / "transcript.md"


def run(a):
    key = api_key()
    src = Path(a.audio).expanduser().resolve()
    date = datetime.fromtimestamp(src.stat().st_birthtime).strftime("%Y-%m-%d")
    d = ROOT / "meetings" / f"{date}-{a.name or re.sub(r'\W+', '-', src.stem.lower()).strip('-')}"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    work = d / "work"
    work.mkdir(exist_ok=True)
    glossary = ROOT / "glossary.txt"
    terms = [x.strip() for x in glossary.read_text().splitlines() if x.strip() and not x.startswith("#")] \
        if glossary.exists() else []
    terms = [x.strip() for x in (a.terms or "").split(",") if x.strip()] + terms
    dur = duration(str(src))
    spans = plan(str(src), dur)
    log(f"{src.name}: {hms(dur)}, {len(spans)} chunk(s) -> {d.relative_to(ROOT)}")

    def job(name, mp3, model, **extra):
        out = d / "raw" / f"{name}.json"
        if out.exists():
            return json.loads(out.read_text())
        return transcribe(key, mp3, out, model, **extra)

    def chunk(name, s, e):
        mp3 = work / f"{name}.mp3"
        if not (d / "raw" / f"{name}.json").exists() and not mp3.exists():
            to_mp3(pcm(str(src), s, e - s), mp3)
        return mp3

    spk_opts = {"diarize": True, **({"keyterms": terms[:1000]} if terms else {})}
    if a.speakers:
        spk_opts["provider"] = {"options": {"elevenlabs": {"num_speakers": a.speakers}}}
    w_opts = {"keyterms": terms[:50]} if terms else {}
    shift = lambda ws, s: [w | {"start": w["start"] + s, "end": w["end"] + s} for w in ws if is_word(w)]
    sspans = [(max(s - OVERLAP, 0) if i else s, e) for i, (s, e) in enumerate(spans)]
    with ThreadPoolExecutor(8) as ex:
        wm = list(ex.map(lambda x: chunk(f"words_{x[0]}", *x[1]), enumerate(spans)))
        sm = list(ex.map(lambda x: chunk(f"speakers_{x[0]}", *x[1]), enumerate(sspans)))
        fw = [ex.submit(job, f"words_{i}", m, WORDS_MODEL, **w_opts) for i, m in enumerate(wm)]
        fs = [ex.submit(job, f"speakers_{i}", m, SPK_MODEL, **spk_opts) for i, m in enumerate(sm)]
        spk = [shift(f.result()["words"], s) for (s, _), f in zip(sspans, fs)]
        spk_words = link(spk, [s for s, _ in spans])
        words, engine = [], f"слова: {WORDS_MODEL}, спикеры: {SPK_MODEL}"
        for (s, e), f, sw in zip(spans, fw, spk):
            try:
                words += shift(f.result()["words"], s)
            except Exception as err:
                log(f"WARNING: words {hms(s)}-{hms(e)} failed ({err}); using {SPK_MODEL} words there")
                words += [w for w in sw if s <= w["start"] < e]
                engine += f" (кусок {hms(s)}-{hms(e)}: только {SPK_MODEL})"

    utts = utterances(assign(words, turns_of(spk_words)))
    t = {"source": str(src), "duration": dur, "engine": engine, "names": {}, "utterances": utts}
    old = d / "transcript.json"
    if old.exists():
        t["names"] = json.loads(old.read_text()).get("names", {})
    old.write_text(json.dumps(t, ensure_ascii=False, indent=1))
    shutil.rmtree(work)
    print(render(d))
    print(f"{len(words)} words, {len(utts)} utterances, {len({u['speaker'] for u in utts})} speakers")


def names(a):
    d = Path(a.dir)
    t = json.loads((d / "transcript.json").read_text())
    t["names"].update(dict(p.split("=", 1) for p in a.pairs))
    (d / "transcript.json").write_text(json.dumps(t, ensure_ascii=False, indent=1))
    print(render(d))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "names":
        p = argparse.ArgumentParser()
        p.add_argument("cmd"), p.add_argument("dir"), p.add_argument("pairs", nargs="+")
        names(p.parse_args())
    else:
        p = argparse.ArgumentParser()
        p.add_argument("audio")
        p.add_argument("--name", help="slug for meetings/<date>-<slug>")
        p.add_argument("--speakers", type=int, help="max number of speakers")
        p.add_argument("--terms", help="comma-separated names/terms to bias recognition")
        run(p.parse_args())
