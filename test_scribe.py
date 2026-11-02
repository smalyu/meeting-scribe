#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pytest"]
# ///
"""Tests without the network: synthetic audio via ffmpeg, OpenRouter played by a fake httpx.post.

  uv run test_scribe.py
"""
import argparse, json, random, re, subprocess, sys, wave
from array import array
from datetime import datetime, timezone

import httpx, pytest

import scribe

RATE = 16000


def wav(path, parts):
    """Mono 16 kHz WAV of (seconds, loudness) parts: white noise, 0 = digital silence."""
    rnd, pcm = random.Random(0), array("h")
    for sec, vol in parts:
        amp = int(vol * 32767)
        pcm.extend(rnd.randint(-amp, amp) if amp else 0 for _ in range(int(sec * RATE)))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(RATE), f.writeframes(pcm.tobytes())
    return path


def m4a(path, parts, **tags):
    """The same as AAC with container tags, like Zoom Audio Recorder and Voice Memos write them."""
    src = wav(path.with_suffix(".wav"), parts)
    meta = [x for k, v in tags.items() for x in ("-metadata", f"{k}={v}")]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *meta, "-c:a", "aac", str(path)], check=True)
    return path


def snapshot(d):
    return {p.relative_to(d): p.read_bytes() for p in d.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Unexpected HTTP request: tests must use fake_api or cached responses")
    monkeypatch.setattr(httpx, "post", blocked)


# --- audio -------------------------------------------------------------------

def test_scan_duration_and_long_silences(tmp_path):
    dur, sil = scribe.scan(str(wav(tmp_path / "a.wav", [(2, 0.3), (6, 0), (2, 0.3)])))
    assert dur == pytest.approx(10, abs=0.1)
    assert len(sil) == 1 and sil[0] == pytest.approx((2, 8), abs=0.2)


def test_plan_skips_a_silent_lead_in(tmp_path):
    p = str(wav(tmp_path / "a.wav", [(7, 0), (3, 0.3)]))
    dur, sil = scribe.scan(p)
    (a, b), = scribe.plan(p, dur, sil)
    assert a == pytest.approx(6.7, abs=0.2) and b == dur  # the chunk starts 0.3 s before the speech


def test_plan_cuts_long_audio_at_quiet_points(tmp_path, monkeypatch):
    p = str(wav(tmp_path / "a.wav", [(4.5, 0.3), (1, 0.01), (6.5, 0.3)]))  # 12 s with a dip at 4.5-5.5
    monkeypatch.setattr(scribe, "CHUNK", 4)
    quietest = scribe.quietest
    monkeypatch.setattr(scribe, "quietest", lambda p, t: quietest(p, t, radius=1))
    dur, sil = scribe.scan(p)
    spans = scribe.plan(p, dur, sil)
    assert len(spans) == 3 and spans[0][0] == 0 and spans[-1][1] == dur
    assert all(e == s for (_, e), (s, _) in zip(spans, spans[1:]))  # no gaps
    assert 4.5 < spans[0][1] < 5.5  # the first cut lands in the dip


def test_tags_recorded_and_zoom_timeline(tmp_path):
    desc = json.dumps({"speakers": [[0, "Ann"], [6.5, None], [8, "Bob"]], "me": [[1, 3], [2, 4], [9, 10]]})
    p = m4a(tmp_path / "z.m4a", [(2, 0.3)], creation_time="2026-10-16T09:29:01Z", description=desc)
    tg = scribe.tags(str(p))
    assert scribe.recorded(tg) == datetime(2026, 10, 16, 9, 29, 1, tzinfo=timezone.utc)
    assert scribe.zoom_timeline(tg) == ([(0, "Ann"), (6.5, None), (8, "Bob")], [[1, 4], [9, 10]])
    assert scribe.recorded({"creation_time": "1904-01-01T00:00:00.000000Z"}) is None  # MP4 epoch: never set
    assert scribe.recorded({}) is None and scribe.zoom_timeline({}) is None
    assert scribe.tags(str(wav(tmp_path / "w.wav", [(1, 0.3)]))) == {}


# --- speakers ----------------------------------------------------------------

def sw(lo, hi, lab):
    return [{"word": "w", "start": t, "end": t + 0.4, "speaker_label": lab} for t in range(lo, hi)]


def test_link_joins_labels_through_the_overlap():
    c0 = sw(0, 500, "speaker_0") + sw(500, 1000, "speaker_1")
    c1 = sw(700, 1000, "speaker_0") + sw(1000, 1500, "speaker_1")  # the next chunk numbers the voices anew
    out = scribe.link([c0, c1], [0, 1000])
    assert [w["start"] for w in out] == list(range(1500))
    who = {w["start"]: w["speaker_label"] for w in out}
    assert (who[0], who[499], who[500], who[999], who[1000]) == ("S1", "S1", "S2", "S2", "S3")
    assert {who[t] for t in range(500, 1000)} == {"S2"}  # one person on both sides of the cut


def test_assign_by_overlap_and_smooths_blips():
    turns = [[0, 4, "S1"], [4, 4.2, "S2"], [4.2, 10, "S1"], [10, 20, "S3"]]
    words = [{"word": "w", "start": t, "end": t + 0.5} for t in (0.5, 1.5, 2.5, 3.9, 5.0, 9.9, 25.0)]
    out = scribe.assign(words, turns)  # 3.9: a weak S2 blip inside S1; 9.9: mostly S3; 25: no turn, follows S3
    assert [w["spk"] for w in out] == ["S1", "S1", "S1", "S1", "S1", "S3", "S3"]


def zw(ts, lab):
    return [{"word": "w", "start": float(t), "end": t + 0.5, "spk": lab} for t in ts]


def test_by_zoom_names_scribe_labels_after_zooms_highlight():
    words = zw(range(0, 10), "S1") + zw(range(10, 20), "S2") + zw(range(20, 30), "S1")
    names, own, share = scribe.by_zoom(words, ([(0, "Ann"), (10.5, "Bob"), (20.5, "Ann")], []))
    assert names == {"1": "Ann", "2": "Bob"} and own is None and share == 1
    assert [w["spk"] for w in words] == ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 10


def test_by_zoom_keeps_a_clear_label_through_a_brief_wrong_highlight():
    words = zw(range(0, 30), "S1")  # one voice all along, the highlight lags behind
    scribe.by_zoom(words, ([(0, "Ann"), (10.5, "Bob"), (12.5, "Ann")], []))
    assert {w["spk"] for w in words} == {"S1"}


def test_by_zoom_gives_a_clear_labels_turn_to_whom_zoom_clearly_shows():
    words = zw(range(0, 8), "S1") + zw(range(10, 14), "S1") + zw(range(16, 40), "S1")  # Scribe merged two voices
    scribe.by_zoom(words, ([(0, "Ann"), (10.5, "Bob"), (16.5, "Ann")], []))
    assert [w["spk"] for w in words] == ["S1"] * 8 + ["S2"] * 4 + ["S1"] * 24


def test_by_zoom_finds_the_user_by_the_microphone():
    words = (zw(range(0, 30), "S1") + zw(range(30, 40), "S2") + zw(range(40, 50), "S1")
             + zw(range(50, 60), "S2") + zw(range(60, 70), "S1"))
    zoom = ([(0, "Ann"), (30.5, "Me"), (40.5, "Ann"), (50.5, "Me"), (60.5, "Ann")], [[30, 40], [50, 60]])
    names, own, _ = scribe.by_zoom(words, zoom)  # gallery view: Zoom highlights the user as they start to speak
    assert names == {"1": "Ann", "2": "Me"} and own == 2
    assert [w["spk"] for w in words] == ["S1"] * 30 + ["S2"] * 10 + ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 10


def test_by_zoom_ignores_a_highlight_stuck_on_the_user():
    words = (zw(range(0, 10), "S1") + zw(range(10, 20), "S2") + zw(range(20, 30), "S1")
             + zw(range(30, 40), "S2") + zw(range(40, 60), "S1"))
    zoom = ([(0, "Ann"), (10.5, "Me"), (20.5, "Ann"), (30.5, "Me")], [[10, 20], [30, 40]])  # stuck on Me from 30.5
    names, own, _ = scribe.by_zoom(words, zoom)
    assert names == {"1": "Ann", "2": "Me"} and own == 2
    assert [w["spk"] for w in words] == ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 20


def test_by_zoom_finds_the_user_whose_microphone_catches_their_start_late():
    words = (zw(range(0, 10), "S1") + zw(range(10, 20), "S2") + zw(range(20, 30), "S1")
             + zw(range(30, 40), "S2") + zw(range(40, 60), "S1"))
    zoom = ([(0, "Ann"), (10.5, "Me"), (20.5, "Ann"), (30.5, "Me")], [[12, 20], [33, 40]])  # quiet starts missed
    names, own, _ = scribe.by_zoom(words, zoom)  # Scribe's turns of S2 show when the user began: the highlight followed
    assert names == {"1": "Ann", "2": "Me"} and own == 2  # so Ann's words under the stuck highlight are not the user's
    assert [w["spk"] for w in words] == ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 10 + ["S2"] * 10 + ["S1"] * 20


def test_by_zoom_ignores_one_name_lit_all_call_long():
    words = zw(range(0, 10), "S1") + zw(range(10, 40), "S2") + zw(range(40, 50), "S1")  # S2: the user merged with S1
    zoom = ([(0, "Me")], [[10, 20]])  # the highlight never moved; the microphone hears a third of S2
    names, own, share = scribe.by_zoom(words, zoom)
    assert names == {} and own == 1 and share == 10 / 50
    assert [w["spk"] for w in words] == ["S2"] * 10 + ["S1"] * 10 + ["S3"] * 20 + ["S2"] * 10
    words = zw(range(0, 10), "S1")  # the user silent: one name all along is just who spoke
    assert scribe.by_zoom(words, ([(0, "Ann")], []))[0] == {"1": "Ann"} and {w["spk"] for w in words} == {"S1"}


def test_by_zoom_numbers_the_user_and_unseen_labels_when_zoom_shows_nothing():
    words = zw(range(0, 30), "S1") + zw(range(30, 40), "S2") + zw(range(40, 50), "S3") + zw(range(50, 56), "S4")
    zoom = ([(0, "Ann"), (40.5, "Bob"), (49.5, None)], [[30, 40]])  # speaker view; the window hidden from 49.5
    names, own, share = scribe.by_zoom(words, zoom)
    assert names == {"1": "Ann", "2": "Bob"} and own == 3 and share == 50 / 56
    assert [w["spk"] for w in words] == ["S1"] * 30 + ["S3"] * 10 + ["S2"] * 10 + ["S7"] * 6  # S7: 3 ids + Scribe's 4


# --- output ------------------------------------------------------------------

def test_utterances_render_and_names(tmp_path):
    words = [{"word": w, "start": t, "end": t + 0.4, "spk": s} for t, w, s in
             [(0, "Привет.", "S1"), (1, "Да.", "S2"), (2, "Как", "S2"), (3, "дела?", "S2"), (100, "Хорошо.", "S2")]]
    utts = scribe.utterances(words)
    assert [(u["speaker"], u["text"]) for u in utts] == [(1, "Привет."), (2, "Да. Как дела?"), (2, "Хорошо.")]
    d = tmp_path / "2026-10-16-test"
    d.mkdir()
    (d / "transcript.json").write_text(json.dumps({
        "source": "x.m4a", "duration": 101, "recorded": "2026-10-16T09:29+00:00", "engine": "e", "speakers": "s",
        "names": {"2": "Мария"}, "utterances": utts}, ensure_ascii=False))
    md = scribe.render(d).read_text()
    assert md.startswith("# 2026-10-16-test\n")
    assert f"· 00:01:41 · {datetime(2026, 10, 16, 9, 29, tzinfo=timezone.utc).astimezone():%d.%m.%Y %H:%M %Z} (creation_time) · e" in md
    assert "Спикеры: Мария (3 с), Спикер 1 (0 с)" in md
    assert "**[00:00:00] Спикер 1:** Привет.\n" in md and "**[00:00:01] Мария:** Да. Как дела?\n" in md
    assert "**[00:01:40] Мария:** Хорошо.\n" in md  # a new paragraph after 90 s
    scribe.names(argparse.Namespace(dir=str(d), pairs=["1=Андрей", "2="]))  # 2= resets to «Спикер 2»
    assert json.loads((d / "transcript.json").read_text())["names"] == {"1": "Андрей"}
    assert "**[00:00:01] Спикер 2:** Да." in (d / "transcript.md").read_text()
    (d / "note.md").write_text("**До записи:** начало не записано.\n")  # survives every re-render
    scribe.names(argparse.Namespace(dir=str(d), pairs=["2=Мария"]))
    assert "(0 с)\n\n**До записи:** начало не записано.\n\n**[00:00:00] Андрей:**" in (d / "transcript.md").read_text()
    with pytest.raises(SystemExit, match="no such speaker"):
        scribe.names(argparse.Namespace(dir=str(d), pairs=["9=Кто-то"]))
    with pytest.raises(SystemExit, match="no transcript.json"):
        scribe.names(argparse.Namespace(dir=str(tmp_path), pairs=["1=Кто-то"]))


def test_share_drops_transcript_links(tmp_path, capsys):
    (tmp_path / "transcript.json").write_text("{}")
    (tmp_path / "followup.md").write_text(
        "# Релиз: план — итоги встречи 16.10.2026\n\n**Участники:** Андрей.\n\n"
        "Таймкоды [ЧЧ:ММ:СС] — время от начала записи.\nТранскрипт: [transcript.md](transcript.md).\n\n"
        "## Кратко\nТекст [00:01:00], см. [второй созвон](../x/transcript.md).\n")
    scribe.share(argparse.Namespace(dir=str(tmp_path), topic=None))
    assert (tmp_path / "Итоги встречи 16.10 — Релиз — план.md").read_text() == (
        "# Релиз: план — итоги встречи 16.10.2026\n\n**Участники:** Андрей.\n\n"
        "Таймкоды [ЧЧ:ММ:СС] — время от начала записи.\n\n## Кратко\n")
    assert "dropped: Транскрипт: [transcript.md](transcript.md)." in capsys.readouterr().err
    scribe.share(argparse.Namespace(dir=str(tmp_path), topic="Релиз 10:30"))
    assert [p.name for p in tmp_path.glob("Итоги*")] == ["Итоги встречи 16.10 — Релиз 10.30.md"]  # the old copy is gone
    (tmp_path / "followup.md").write_text("# Тема — итоги встречи 13.10.2026 (?)\n")
    scribe.share(argparse.Namespace(dir=str(tmp_path), topic=None))
    assert [p.name for p in tmp_path.glob("Итоги*")] == ["Итоги встречи 13.10 — Тема.md"]
    (tmp_path / "followup.md").write_text("# Без даты\n")
    with pytest.raises(SystemExit, match="must start with"):
        scribe.share(argparse.Namespace(dir=str(tmp_path), topic=None))
    (tmp_path / "followup.md").unlink()
    with pytest.raises(SystemExit, match="not found"):
        scribe.share(argparse.Namespace(dir=str(tmp_path), topic=None))


# --- the whole run, OpenRouter played by a fake ------------------------------

MAI = [{"word": "Привет" if t < 6 else "Пока", "start": t, "end": t + 0.4} for t in range(12)]
SCRIBE = [{"word": "w", "start": t, "end": t + 0.4, "speaker_label": f"speaker_{t >= 6:d}"} for t in range(12)]


def fake_api(answers, calls):
    """answers[model]: the words to return, or an HTTP status to fail with."""
    def post(url, json, headers, timeout):
        calls.append(json["model"])
        a = answers[json["model"]]
        if isinstance(a, int):
            return httpx.Response(a, json={"error": {"message": "nope"}})
        return httpx.Response(200, json={"words": a, "usage": {"cost": 0.01}})
    return post


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(scribe, "ROOT", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(scribe.time, "sleep", lambda s: None)
    return tmp_path


def test_run_end_to_end_cache_and_rename(home, monkeypatch, capsys):
    desc = json.dumps({"speakers": [[0, "Ann"], [6.5, "Bob"]], "me": []})
    rec = m4a(home / "Zoom 2026-10-16 11-59.m4a", [(12, 0.3)], creation_time="2026-10-16T09:29:01Z", description=desc)
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    scribe.run(argparse.Namespace(audio=str(rec), name=None, terms=None))
    date = datetime(2026, 10, 16, 9, 29, 1, tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d")
    d, = (home / "meetings").iterdir()
    assert d.name == f"{date}-" + re.sub("-{2,}", "-", "zoom-2026-10-16-11-59".replace(date, "")).strip("-")
    assert sorted(calls) == sorted([scribe.WORDS_MODEL, scribe.SPK_MODEL])  # one chunk, two requests
    assert {p.name for p in (d / "raw").iterdir()} == {"size", "words_0-12.json", "speakers_0-12.json"}
    md = (d / "transcript.md").read_text()
    assert "(creation_time)" in md and "спикеры: Zoom и " in md
    assert "**[00:00:00] Ann:** Привет Привет" in md and "**[00:00:06] Bob:** Пока Пока" in md
    out = capsys.readouterr()
    assert "Zoom and the microphone named 100% of words; own voice: none" in out.out
    assert "cost: $0.02 total, $0.02 this run" in out.err
    # again: everything from the cache, names kept, --terms cannot apply
    scribe.names(argparse.Namespace(dir=str(d), pairs=["2=Боб"]))
    scribe.run(argparse.Namespace(audio=str(rec), name=None, terms="Ann"))
    assert len(calls) == 2
    assert "**[00:00:06] Боб:** Пока" in (d / "transcript.md").read_text()
    err = capsys.readouterr().err
    assert "cost: $0.02 total, $0.00 this run" in err and "--terms had no effect" in err
    # a new --name renames the folder of a known recording instead of paying again
    scribe.run(argparse.Namespace(audio=str(rec), name="sync", terms=None))
    assert not d.exists() and len(calls) == 2
    assert "**[00:00:06] Боб:** Пока" in (home / "meetings" / f"{date}-sync" / "transcript.md").read_text()


def test_run_folders_never_pay_twice(home, monkeypatch, capsys):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    go = lambda name=None: scribe.run(argparse.Namespace(audio=str(rec), name=name, terms=None))
    go()
    first, = (home / "meetings").iterdir()
    # the same path now holds another recording: its folder is not this one, and not renamed either
    wav(rec, [(13, 0.3)])
    with pytest.raises(SystemExit, match="already holds another recording"):
        go()
    go("v2")
    assert len(calls) == 4 and first.exists() and (first / "raw" / "words_0-12.json").exists()
    v2 = first.with_name(first.name[:10] + "-v2")
    assert (v2 / "raw" / "words_0-13.json").exists()
    # a renamed folder keeps the date the user gave it; a taken name is an error, not a second transcription
    v2.rename(home / "meetings" / "2020-01-01-v2")
    go("v3")
    assert (home / "meetings" / "2020-01-01-v3" / "transcript.md").exists() and len(calls) == 4
    (home / "meetings" / "2020-01-01-v4").mkdir()
    with pytest.raises(SystemExit, match="already holds this recording"):
        go("v4")
    go()  # without --name the renamed folder is found as well
    assert len(calls) == 4 and {p.name for p in (home / "meetings").iterdir()} == {first.name, "2020-01-01-v3", "2020-01-01-v4"}
    assert "meetings/2020-01-01-v2 -> meetings/2020-01-01-v3" in capsys.readouterr().err


def test_join_makes_one_transcript_and_reruns_never_pay(home, monkeypatch, capsys):
    one, two = wav(home / "one.wav", [(12, 0.3)]), wav(home / "two.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    go = lambda rec, name=None: scribe.run(argparse.Namespace(audio=str(rec), name=name, terms=None))
    go(one, "sync"), go(two, "sync-2")
    d, d2 = sorted((home / "meetings").iterdir())
    (d / "note.md").write_text("**До записи:** начало не записано.")
    scribe.join(argparse.Namespace(dir=str(d), parts=[str(d2)]))
    assert [p.name for p in (home / "meetings").iterdir()] == [d.name] and (d / "2" / "raw" / "words_0-12.json").exists()
    scribe.names(argparse.Namespace(dir=str(d / "2"), pairs=["1=Ann"]))  # names of a part re-render the meeting
    md = (d / "transcript.md").read_text()
    assert md.startswith(f"# {d.name}\n\nЗаписей: 2. Таймкоды [N·ЧЧ:ММ:СС] — время от начала записи N.\n\n"
                         "Спикеры: Спикер 1·1 (5 с), Спикер 1·2 (5 с), Ann (5 с), Спикер 2·2 (5 с)\n\n"
                         "## Запись 1\n\nИсточник: `")  # labels of different recordings are not summed
    assert "\n\n**До записи:** начало не записано.\n\n**[1·00:00:00] Спикер 1·1:** Привет" in md
    assert "\n## Запись 2\n\nИсточник: `" in md and "**[2·00:00:00] Ann:** Привет" in md
    assert "**[2·00:00:06] Спикер 2·2:** Пока" in md and f"{d2.name} -> {d.name}/2" in capsys.readouterr().err
    assert not (d / "2" / "transcript.md").exists()
    go(two, "other"), go(one)  # both found where they are now: nothing paid, the part is not moved out
    assert len(calls) == 4 and (d / "2" / "transcript.json").exists() and "**[2·00:00:00] Ann:**" in md
    assert "--name ignored" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="a part of a joined meeting"):
        scribe.join(argparse.Namespace(dir=str(d / "2"), parts=[str(d)]))


def test_cache_only_needs_no_key_and_keeps_names(home, monkeypatch, capsys):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    scribe.run(argparse.Namespace(audio=str(rec), name="sync", terms=None))
    d, = (home / "meetings").iterdir()
    scribe.names(argparse.Namespace(dir=str(d), pairs=["1=Андрей"]))
    capsys.readouterr()
    monkeypatch.setattr(scribe, "api_key", lambda: pytest.fail("cache-only must not read an API key"))
    scribe.run(argparse.Namespace(audio=str(rec), name="renamed", terms=None, cache_only=True))
    renamed, = (home / "meetings").iterdir()
    assert renamed.name.endswith("-renamed") and not d.exists()
    assert "Андрей" in (renamed / "transcript.md").read_text() and len(calls) == 2
    assert "$0.00 this run" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["words_0-12.json", "speakers_0-12.json", "changed-plan"])
def test_cache_only_missing_chunk_never_requests_or_changes_files(home, monkeypatch, missing):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    scribe.run(argparse.Namespace(audio=str(rec), name="sync", terms=None))
    d, = (home / "meetings").iterdir()
    if missing == "changed-plan":
        monkeypatch.setattr(scribe, "plan", lambda *args: [(0, 6), (6, 12)])
    else:
        (d / "raw" / missing).unlink()
    before = snapshot(d)
    monkeypatch.setattr(scribe, "api_key", lambda: pytest.fail("cache-only must not read an API key"))
    with pytest.raises(SystemExit, match="--cache-only: missing cached response"):
        scribe.run(argparse.Namespace(audio=str(rec), name="renamed", terms=None, cache_only=True))
    assert len(calls) == 2 and snapshot(d) == before and not (d / "work").exists()
    assert list((home / "meetings").iterdir()) == [d]  # even the requested rename must not happen


def test_cache_only_without_cache_creates_nothing(home, monkeypatch):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    monkeypatch.setattr(scribe, "api_key", lambda: pytest.fail("cache-only must not read an API key"))
    with pytest.raises(SystemExit, match="--cache-only: missing cached response"):
        scribe.run(argparse.Namespace(audio=str(rec), name=None, terms=None, cache_only=True))
    assert not (home / "meetings").exists()


def test_check_joined_meeting_preserves_originals_and_reports_manual_edits(home, monkeypatch, capsys):
    one, two = wav(home / "one.wav", [(12, 0.3)]), wav(home / "two.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    for rec, name in ((one, "sync"), (two, "sync-2")):
        scribe.run(argparse.Namespace(audio=str(rec), name=name, terms=None))
    d, d2 = sorted((home / "meetings").iterdir())
    scribe.join(argparse.Namespace(dir=str(d), parts=[str(d2)]))
    scribe.names(argparse.Namespace(dir=str(d / "2"), pairs=["1=Андрей"]))
    (d / "note.md").write_text("**До записи:** начало не записано.")
    (d / "followup.md").write_text("Итоги для участников.")
    scribe.render(d)
    monkeypatch.setattr(scribe, "api_key", lambda: pytest.fail("check must not read an API key"))
    before = snapshot(d)
    scribe.check(argparse.Namespace(dir=str(d / "2")))  # a part checks the whole joined meeting
    assert "check: unchanged (2 recording(s))" in capsys.readouterr().out
    assert len(calls) == 4 and snapshot(d) == before
    t = json.loads((d / "2" / "transcript.json").read_text())
    t["utterances"][0]["text"] = "Исправлено вручную."
    (d / "2" / "transcript.json").write_text(json.dumps(t, ensure_ascii=False))
    scribe.render(d)
    before = snapshot(d)
    with pytest.raises(SystemExit, match=r"2/transcript.json \(utterances\).*transcript.md"):
        scribe.check(argparse.Namespace(dir=str(d)))
    assert len(calls) == 4 and snapshot(d) == before


def test_check_missing_cache_preserves_originals(home, monkeypatch):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: SCRIBE}, calls))
    scribe.run(argparse.Namespace(audio=str(rec), name="sync", terms=None))
    d, = (home / "meetings").iterdir()
    (d / "raw" / "words_0-12.json").unlink()
    before = snapshot(d)
    with pytest.raises(SystemExit, match="--cache-only: missing cached response"):
        scribe.check(argparse.Namespace(dir=str(d)))
    assert len(calls) == 2 and snapshot(d) == before


def test_run_falls_back_to_scribe_words_when_mai_fails(home, monkeypatch, capsys):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: 500, scribe.SPK_MODEL: SCRIBE}, calls))
    scribe.run(argparse.Namespace(audio=str(rec), name=None, terms=None))
    d, = (home / "meetings").iterdir()
    assert d.name.endswith("-rec") and calls.count(scribe.WORDS_MODEL) == 4  # retried, then gave up
    assert "WARNING: words 00:00:00-00:00:12 failed" in capsys.readouterr().err
    assert f"только {scribe.SPK_MODEL}" in json.loads((d / "transcript.json").read_text())["engine"]
    assert "**[00:00:00] Спикер 1:** w w" in (d / "transcript.md").read_text()


def test_run_stops_when_scribe_fails_or_the_key_is_rejected(home, monkeypatch):
    rec = wav(home / "rec.wav", [(12, 0.3)])
    calls = []
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: 500}, calls))
    with pytest.raises(SystemExit, match="rerun the same command"):
        scribe.run(argparse.Namespace(audio=str(rec), name=None, terms=None))
    d, = (home / "meetings").iterdir()
    assert (d / "raw" / "words_0-12.json").exists() and not (d / "transcript.json").exists()  # MAI's answer is kept
    calls.clear()
    monkeypatch.setattr(scribe.httpx, "post", fake_api({scribe.WORDS_MODEL: MAI, scribe.SPK_MODEL: 402}, calls))
    with pytest.raises(SystemExit, match="HTTP 402"):
        scribe.run(argparse.Namespace(audio=str(rec), name=None, terms=None))
    assert calls == [scribe.SPK_MODEL]  # no retries, and MAI's chunk came from the cache


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__, *sys.argv[1:]]))
