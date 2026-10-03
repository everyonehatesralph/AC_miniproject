"""
03b_build_listening_review.py -- the blind listening-review page: data/review/listening_review.html.

One self-contained page (works offline; open it by double-clicking) that plays every take from
data/review/audio/ (written by 03) and collects, for each take: the word heard, whether the whole word is
audible, whether the automatically located word contains the whole word (if not: corrected start and end),
another voice over the word, and clarity 1-5; for participants with retakes, which take holds the whole word
most clearly.

Blind by construction: the page shows participant codes and take numbers only. It never shows the metadata
word, the feeling text, the ratings, the automatic take choice or the quality flags, and the audio files
have neutral names (<code>_t<k>.wav).

Answers stay in the browser as a draft until the listener presses "Save answers (CSV)". The CSV goes into
data/review/answers/; src/01b_merge_listening_reviews.py merges all listeners.

Usage: .venv/Scripts/python.exe src/03b_build_listening_review.py
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from common import DATA, REPORTS, frame_energy_db, rel, setup_logging

REVIEW_DIR = DATA / "review"
AUDIO_DIR = REVIEW_DIR / "audio"
QA_TABLE = REPORTS / "audio_quality_table.csv"
PAGE = REVIEW_DIR / "listening_review.html"
ENVELOPE_RANGE_DB = 60.0   # the drawn energy curve spans the loudest 60 dB of each clip

log = setup_logging("listening_page", REPORTS / "logs" / "03b_build_listening_review.log")


def envelope_points(y: np.ndarray, sr: int) -> str:
    """Frame energy as SVG polyline points in a 1000 x 120 box (x = time, y = level)."""
    db, times = frame_energy_db(y, sr)
    peak = float(db.max())
    low = max(float(db.min()), peak - ENVELOPE_RANGE_DB)
    level = np.clip((db - low) / max(peak - low, 1e-6), 0.0, 1.0)
    dur = len(y) / sr
    return " ".join(f"{1000 * t / dur:.1f},{114 - 106 * v:.1f}" for t, v in zip(times, level))


def build_items(qa: pd.DataFrame) -> list[dict]:
    participants = []
    for code, g in qa.sort_values(["participant_code", "review_take"]).groupby("participant_code", sort=True):
        takes = []
        for _, r in g.iterrows():
            k = int(r["review_take"])
            clip, word = AUDIO_DIR / f"{code}_t{k}.wav", AUDIO_DIR / f"{code}_t{k}_word.wav"
            if not (clip.is_file() and word.is_file()):
                raise FileNotFoundError(f"{rel(clip)} or its _word file is missing: re-run src/03_prepare_audio.py")
            y, sr = sf.read(clip, dtype="float32")
            takes.append({"k": k, "id": r["drive_file_id"], "dur": round(len(y) / sr, 3),
                          "start": float(r["auto_seg_start_s"]), "end": float(r["auto_seg_end_s"]),
                          "clip": f"audio/{clip.name}", "word": f"audio/{word.name}",
                          "points": envelope_points(y, sr)})
        participants.append({"code": code, "takes": takes})
    return participants


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening review</title>
<style>
:root {
  --bg: #f6f6f3; --surface: #ffffff; --ink: #1c1c1a; --ink-2: #4f4f4a; --muted: #85857d; --line: #e2e2dc;
  --accent: #2563c9; --accent-ink: #ffffff; --seg: rgba(37, 99, 201, .15); --curve: #2563c9;
  --mark: rgba(214, 108, 30, .22); --mark-line: #c8621a; --ok: #23784a; --todo: #9a5b00; --bad: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141412; --surface: #1e1e1b; --ink: #ecece6; --ink-2: #bdbdb4; --muted: #8b8b83; --line: #33332e;
    --accent: #7ea6ff; --accent-ink: #10131a; --seg: rgba(126, 166, 255, .18); --curve: #7ea6ff;
    --mark: rgba(255, 160, 90, .22); --mark-line: #ffa05a; --ok: #6fcf97; --todo: #f2b155; --bad: #ff8a80;
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 0 16px 64px; }
h1 { font-size: 1.45rem; margin: 24px 0 4px; }
.lede { color: var(--ink-2); margin: 0 0 12px; }
details.how { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin: 12px 0; }
details.how summary { cursor: pointer; font-weight: 600; }
details.how ol { margin: 8px 0 4px; padding-left: 20px; }
details.how li { margin: 4px 0; }
.bar { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); padding: 10px 0; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
.bar label { font-weight: 600; }
.bar input[type=text] { width: 6.5em; text-transform: uppercase; }
.progress { color: var(--ink-2); }
.spacer { flex: 1; }
button, input, select { font: inherit; color: inherit; }
input[type=text], input[type=number] { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
input:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
button { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 5px 12px; cursor: pointer; }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); font-weight: 600; }
.notice { border-radius: 8px; padding: 8px 12px; margin: 10px 0; border: 1px solid var(--line); background: var(--surface); }
.notice.warn { border-color: var(--todo); }
section.p { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin: 16px 0; }
section.p > header { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; }
section.p h2 { font-size: 1.1rem; margin: 0; }
.chip { font-size: .8rem; border-radius: 999px; padding: 1px 9px; border: 1px solid var(--line); color: var(--ink-2); }
.chip.done { color: var(--ok); border-color: var(--ok); }
.chip.todo { color: var(--todo); border-color: var(--todo); }
.take { border-top: 1px solid var(--line); margin-top: 12px; padding-top: 12px; }
.take h3 { font-size: .95rem; margin: 0 0 6px; }
.players { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; }
.players span { display: block; font-size: .8rem; color: var(--ink-2); }
audio { height: 34px; max-width: 100%; }
.plot { position: relative; margin: 10px 0 2px; }
svg.env { display: block; width: 100%; height: 96px; cursor: crosshair; background: var(--bg); border-radius: 6px; }
svg.env .seg { fill: var(--seg); }
svg.env .mark { fill: var(--mark); stroke: var(--mark-line); stroke-width: 2; vector-effect: non-scaling-stroke; }
svg.env .curve { fill: none; stroke: var(--curve); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
svg.env .ph { stroke: var(--ink); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.ticks { position: relative; height: 16px; font-size: .72rem; color: var(--muted); }
.ticks span { position: absolute; transform: translateX(-50%); }
.legend { font-size: .78rem; color: var(--muted); margin-bottom: 6px; }
.legend i { display: inline-block; width: 12px; height: 10px; vertical-align: -1px; margin: 0 4px 0 8px; border-radius: 2px; }
.q { display: grid; grid-template-columns: minmax(180px, 280px) 1fr; gap: 6px 14px; align-items: center; margin-top: 8px; }
.q > .lab { color: var(--ink-2); }
.q .opts label { margin-right: 14px; white-space: nowrap; }
.q input[type=text].word { width: min(16em, 100%); }
.q input[type=text].notes { width: 100%; }
.fix { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.fix input[type=number] { width: 6.2em; }
.fix .err { color: var(--bad); font-size: .85rem; }
.choose { border-top: 1px solid var(--line); margin-top: 12px; padding-top: 10px; }
.choose .opts label { margin-right: 16px; white-space: nowrap; }
.hide-done section.p.complete { display: none; }
@media (max-width: 640px) { .q { grid-template-columns: 1fr; } .q > .lab { margin-top: 6px; } }
</style>
</head>
<body>
<main>
<h1>Listening review</h1>
<p class="lede">__N_PARTICIPANTS__ participants, __N_TAKES__ takes. Work alone, with headphones, without looking at the
data sheet. Your answers decide nothing alone: a decision needs at least two listeners who agree.</p>
<details class="how" open>
<summary>How to do it (5 minutes to read)</summary>
<ol>
<li>Type your <b>initials</b> at the top. Don't discuss with the other listeners until everyone has saved.</li>
<li>Don't look at the sheet (spoken words, feelings, ratings) while you listen.</li>
<li>For each take, play the <b>whole clip</b>, then the <b>located word</b> (what the computer cut out).</li>
<li>Type the word <b>as you hear it</b>, in the usual Bisaya spelling. If you can't make it out, leave it blank
and say why under Notes.</li>
<li>The <b>blue band</b> in the picture is where the computer thinks the word is. If any part of the word lies
outside it, answer <b>No</b> and mark the real word: click the picture to move the playhead (or pause the
whole-clip player at the right moment), press <b>Set start</b> / <b>Set end</b>, then check with
<b>Play marked part</b>. You can fine-tune the numbers (seconds).</li>
<li>For a participant with several takes, choose the take in which the <b>whole word</b> is heard most clearly.
Never choose by which one sounds more emotional.</li>
<li>Press <b>Save answers (CSV)</b> and move the file from your Downloads folder into
<code>data/review/answers/</code>. One file per listener: replace your older file.</li>
</ol>
</details>
<div class="bar">
  <label for="initials">Your initials</label><input id="initials" type="text" maxlength="4" autocomplete="off" placeholder="e.g. RB">
  <span class="progress" id="progress"></span>
  <label style="font-weight:400"><input type="checkbox" id="hide-done"> hide finished</label>
  <span class="spacer"></span>
  <button id="clear" type="button">Clear my answers</button>
  <button id="save" class="primary" type="button">Save answers (CSV)</button>
</div>
<div class="notice warn" id="no-storage" hidden>This browser is not keeping a draft of your answers. Finish in one
sitting, or press Save answers (CSV) now and then.</div>
<div class="notice warn" id="stale" hidden></div>
<div class="notice" id="saved" hidden></div>
<div id="list"></div>
</main>
<script type="application/json" id="review-data">__DATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("review-data").textContent);
  var KEY = "g3-listening-review";
  var state = { initials: "", takes: {}, choice: {} };
  var storageOK = true;
  try {
    var saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (saved && saved.takes) state = saved;
  } catch (e) { storageOK = false; }
  if (!storageOK) document.getElementById("no-storage").hidden = false;

  // answers about boundaries refer to the boundaries shown; if those changed, drop them
  var staleTakes = [];
  DATA.participants.forEach(function (p) {
    p.takes.forEach(function (t) {
      var a = state.takes[p.code + "|" + t.id];
      if (a && a.shown && (a.shown[0] !== t.start || a.shown[1] !== t.end)) {
        delete a.bounds; delete a.cs; delete a.ce; a.shown = [t.start, t.end];
        staleTakes.push(p.code + " take " + t.k);
      }
    });
  });
  if (staleTakes.length) {
    var st = document.getElementById("stale");
    st.textContent = "The located word changed since your draft for: " + staleTakes.join(", ") +
      ". Please answer the boundary question for these takes again.";
    st.hidden = false;
  }

  function persist() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* draft only */ } }
  function answer(code, t) {
    var key = code + "|" + t.id;
    if (!state.takes[key]) state.takes[key] = { shown: [t.start, t.end] };
    return state.takes[key];
  }
  function num(v) { return v === "" || v === undefined || v === null ? NaN : Number(v); }
  function marksValid(a, t) {
    var s = num(a.cs), e = num(a.ce);
    return isFinite(s) && isFinite(e) && s >= 0 && e <= t.dur + 1e-9 && s < e;
  }
  function takeDone(a, t) {
    return !!(a && a.whole && a.bounds && a.other && a.clarity && (a.bounds === "Y" || marksValid(a, t)));
  }
  function participantDone(p) {
    var all = p.takes.every(function (t) { return takeDone(state.takes[p.code + "|" + t.id], t); });
    return all && (p.takes.length === 1 || !!state.choice[p.code]);
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function radios(name, opts, value) {
    return opts.map(function (o) {
      return '<label><input type="radio" name="' + name + '" value="' + o[0] + '"' + (value === o[0] ? " checked" : "") + "> " + o[1] + "</label>";
    }).join("");
  }

  var list = document.getElementById("list");
  var html = [];
  DATA.participants.forEach(function (p) {
    html.push('<section class="p" id="p-' + esc(p.code) + '" data-code="' + esc(p.code) + '"><header><h2>' + esc(p.code) +
      '</h2><span class="chip">' + p.takes.length + (p.takes.length > 1 ? " takes" : " take") +
      '</span><span class="chip status"></span></header>');
    p.takes.forEach(function (t) {
      var a = state.takes[p.code + "|" + t.id] || {};
      var n = p.code + "-" + t.k;
      var ticks = [], step = t.dur < 3 ? 0.5 : 1;
      for (var s = 0; s <= t.dur + 1e-9; s += step) ticks.push('<span style="left:' + (100 * s / t.dur) + '%">' + s + " s</span>");
      html.push('<div class="take" data-id="' + esc(t.id) + '" data-k="' + t.k + '">' +
        "<h3>Take " + t.k + "</h3>" +
        '<div class="players"><div><span>Whole clip</span><audio class="clip" controls preload="none" src="' + esc(t.clip) + '"></audio></div>' +
        '<div><span>Located word</span><audio controls preload="none" src="' + esc(t.word) + '"></audio></div></div>' +
        '<div class="plot"><svg class="env" viewBox="0 0 1000 120" preserveAspectRatio="none" role="img" aria-label="Energy of take ' + t.k + ' over time; click to move the playhead">' +
        '<rect class="seg" x="' + (1000 * t.start / t.dur) + '" y="0" width="' + (1000 * (t.end - t.start) / t.dur) + '" height="120"></rect>' +
        '<rect class="mark" x="0" y="2" width="0" height="116" visibility="hidden"></rect>' +
        '<polyline class="curve" points="' + t.points + '"></polyline>' +
        '<line class="ph" x1="0" x2="0" y1="0" y2="120" visibility="hidden"></line></svg>' +
        '<div class="ticks">' + ticks.join("") + "</div></div>" +
        '<div class="legend"><i style="background:var(--seg)"></i>located word (automatic)<i style="background:var(--mark);border:1px solid var(--mark-line)"></i>your marked word<i style="background:var(--ink);width:2px"></i>playhead</div>' +
        '<div class="q">' +
        '<div class="lab">Word you hear</div><div><input type="text" class="word" data-f="heard_word" maxlength="40" autocomplete="off" spellcheck="false" placeholder="type what you hear" value="' + esc(a.heard_word || "") + '"></div>' +
        '<div class="lab">Whole word audible (not cut off)?</div><div class="opts">' + radios(n + "-whole", [["Y", "Yes"], ["N", "No"]], a.whole) + "</div>" +
        '<div class="lab">Whole word inside the blue band?</div><div class="opts">' + radios(n + "-bounds", [["Y", "Yes"], ["N", "No, I mark it"]], a.bounds) + "</div>" +
        '<div class="lab fixlab"' + (a.bounds === "N" ? "" : " hidden") + '>The word really is</div><div class="fix"' + (a.bounds === "N" ? "" : " hidden") + ">" +
        'start <input type="number" data-f="cs" step="0.01" min="0" max="' + t.dur + '" value="' + esc(a.cs === undefined ? "" : a.cs) + '"> <button type="button" data-set="cs">Set start</button>' +
        'end <input type="number" data-f="ce" step="0.01" min="0" max="' + t.dur + '" value="' + esc(a.ce === undefined ? "" : a.ce) + '"> <button type="button" data-set="ce">Set end</button>' +
        '<button type="button" class="play-mark">Play marked part</button><span class="err"></span></div>' +
        '<div class="lab">Another voice over the word?</div><div class="opts">' + radios(n + "-other", [["Y", "Yes"], ["N", "No"]], a.other) + "</div>" +
        '<div class="lab">Clarity of the word</div><div class="opts">' + radios(n + "-clarity", [["1", "1 unclear"], ["2", "2"], ["3", "3"], ["4", "4"], ["5", "5 very clear"]], a.clarity) + "</div>" +
        '<div class="lab">Notes</div><div><input type="text" class="notes" data-f="notes" maxlength="200" autocomplete="off" value="' + esc(a.notes || "") + '"></div>' +
        "</div></div>");
    });
    if (p.takes.length > 1) {
      var opts = p.takes.map(function (t) { return [t.id, "Take " + t.k]; }).concat([["none", "No take holds the whole word"]]);
      html.push('<div class="choose"><div class="lab"><b>Which take holds the whole word most clearly?</b></div><div class="opts">' +
        radios(p.code + "-choice", opts, state.choice[p.code]) + "</div></div>");
    }
    html.push("</section>");
  });
  list.innerHTML = html.join("");

  function find(code) { for (var i = 0; i < DATA.participants.length; i++) if (DATA.participants[i].code === code) return DATA.participants[i]; }
  function takeOf(p, id) { for (var i = 0; i < p.takes.length; i++) if (p.takes[i].id === id) return p.takes[i]; }

  function drawMark(box, a, t) {
    var rect = box.querySelector("rect.mark"), err = box.querySelector(".err");
    var s = num(a.cs), e = num(a.ce);
    if (a.bounds === "N" && isFinite(s) && isFinite(e) && s < e) {
      rect.setAttribute("x", 1000 * Math.max(0, s) / t.dur);
      rect.setAttribute("width", 1000 * (Math.min(e, t.dur) - Math.max(0, s)) / t.dur);
      rect.setAttribute("visibility", "visible");
    } else rect.setAttribute("visibility", "hidden");
    var msg = "";
    if (a.bounds === "N" && (a.cs !== undefined && a.cs !== "") && (a.ce !== undefined && a.ce !== "") && !marksValid(a, t))
      msg = "start must be before end, inside 0–" + t.dur + " s";
    err.textContent = msg;
  }
  function refresh() {
    var done = 0;
    DATA.participants.forEach(function (p) {
      var sec = document.getElementById("p-" + p.code), ok = participantDone(p);
      if (ok) done++;
      sec.classList.toggle("complete", ok);
      var chip = sec.querySelector(".status");
      chip.textContent = ok ? "finished" : "to do";
      chip.className = "chip status " + (ok ? "done" : "todo");
    });
    document.getElementById("progress").textContent = done + " of " + DATA.participants.length + " participants finished";
  }

  list.addEventListener("input", function (ev) {
    var el = ev.target, sec = el.closest("section.p"), box = el.closest(".take");
    if (!sec) return;
    var p = find(sec.dataset.code);
    if (el.type === "radio" && /-choice$/.test(el.name)) { state.choice[p.code] = el.value; persist(); refresh(); return; }
    if (!box) return;
    var t = takeOf(p, box.dataset.id), a = answer(p.code, t);
    if (el.type === "radio") {
      var field = el.name.split("-").pop();
      a[field] = el.value;
      if (field === "bounds") {
        box.querySelector(".fix").hidden = el.value !== "N";
        box.querySelector(".fixlab").hidden = el.value !== "N";
      }
    } else if (el.dataset.f) {
      a[el.dataset.f] = el.value;
    }
    drawMark(box, a, t);
    persist(); refresh();
  });

  var stopAt = null, active = null;
  function playhead(box, audio, t) {
    var line = box.querySelector("line.ph"), x = 1000 * audio.currentTime / t.dur;
    line.setAttribute("x1", x); line.setAttribute("x2", x); line.setAttribute("visibility", "visible");
  }
  function tick() {
    if (!active) return;
    var box = active.box, audio = active.audio;
    playhead(box, audio, active.t);
    if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; }
    if (!audio.paused) requestAnimationFrame(tick);
  }
  document.querySelectorAll(".take").forEach(function (box) {
    var sec = box.closest("section.p"), p = find(sec.dataset.code), t = takeOf(p, box.dataset.id);
    var audio = box.querySelector("audio.clip"), svg = box.querySelector("svg.env");
    drawMark(box, state.takes[p.code + "|" + t.id] || {}, t);
    audio.addEventListener("play", function () { active = { box: box, audio: audio, t: t }; requestAnimationFrame(tick); });
    audio.addEventListener("seeked", function () { playhead(box, audio, t); });
    audio.addEventListener("timeupdate", function () { playhead(box, audio, t); });
    svg.addEventListener("click", function (ev) {
      var r = svg.getBoundingClientRect();
      var time = Math.max(0, Math.min(t.dur, (ev.clientX - r.left) / r.width * t.dur));
      stopAt = null;
      try { audio.currentTime = time; } catch (e) { /* not loaded yet */ }
      var line = box.querySelector("line.ph"), x = 1000 * time / t.dur;
      line.setAttribute("x1", x); line.setAttribute("x2", x); line.setAttribute("visibility", "visible");
      box.dataset.cursor = time;
    });
    box.querySelectorAll("button[data-set]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var a = answer(p.code, t), time = audio.currentTime || Number(box.dataset.cursor || 0);
        a[btn.dataset.set] = time.toFixed(2);
        box.querySelector('input[data-f="' + btn.dataset.set + '"]').value = a[btn.dataset.set];
        drawMark(box, a, t); persist(); refresh();
      });
    });
    box.querySelector(".play-mark").addEventListener("click", function () {
      var a = answer(p.code, t);
      if (!marksValid(a, t)) return;
      audio.currentTime = num(a.cs); stopAt = num(a.ce); audio.play();
    });
  });

  var initials = document.getElementById("initials");
  initials.value = state.initials || "";
  initials.addEventListener("input", function () { state.initials = initials.value.toUpperCase().replace(/[^A-Z]/g, ""); persist(); });
  document.getElementById("hide-done").addEventListener("change", function (ev) { list.classList.toggle("hide-done", ev.target.checked); });
  document.getElementById("clear").addEventListener("click", function () {
    if (!confirm("Delete all your answers in this browser? Save a CSV first if you need them.")) return;
    state = { initials: state.initials, takes: {}, choice: {} };
    persist(); location.reload();
  });

  var COLUMNS = ["reviewer_initials", "participant_code", "take", "drive_file_id", "shown_start_s", "shown_end_s",
    "heard_word", "whole_word_audible_Y_N", "boundaries_ok_Y_N", "corrected_start_s", "corrected_end_s",
    "clarity_1_to_5", "other_voice_over_word_Y_N", "chosen_take_Y_N", "notes", "page_version", "saved_at"];
  function cell(v) {
    v = v === undefined || v === null ? "" : String(v);
    if (/^[=+\-@]/.test(v)) v = "'" + v;            // never a spreadsheet formula
    return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  document.getElementById("save").addEventListener("click", function () {
    var who = (state.initials || "").toUpperCase();
    if (!/^[A-Z]{2,4}$/.test(who)) { alert("Type your initials (2 to 4 letters) at the top first."); initials.focus(); return; }
    var now = new Date().toISOString(), rows = [COLUMNS.join(",")];
    DATA.participants.forEach(function (p) {
      p.takes.forEach(function (t) {
        var a = state.takes[p.code + "|" + t.id] || {}, ch = state.choice[p.code];
        var chosen = p.takes.length > 1 && ch ? (ch === t.id ? "Y" : "N") : "";
        var marks = a.bounds === "N" && marksValid(a, t);
        rows.push([who, p.code, t.k, t.id, t.start.toFixed(3), t.end.toFixed(3), (a.heard_word || "").trim(), a.whole, a.bounds,
          marks ? Number(a.cs).toFixed(2) : "", marks ? Number(a.ce).toFixed(2) : "", a.clarity, a.other, chosen,
          (a.notes || "").trim(), DATA.version, now].map(cell).join(","));
      });
    });
    var blob = new Blob(["﻿" + rows.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "listening_review_" + who + "_" + now.slice(0, 10) + ".csv";
    document.body.appendChild(link); link.click(); link.remove();
    var msg = document.getElementById("saved");
    msg.textContent = "Saved " + link.download + ". Move it into data/review/answers/ (replace your older file), then re-run the pipeline from step 01b.";
    msg.hidden = false;
  });
  refresh();
})();
</script>
</body>
</html>
"""


def main() -> int:
    if not QA_TABLE.is_file():
        raise FileNotFoundError(f"{rel(QA_TABLE)} not found: run src/03_prepare_audio.py first")
    qa = pd.read_csv(QA_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "review_take" not in qa.columns:
        raise ValueError(f"{rel(QA_TABLE)} predates the listening review: re-run src/03_prepare_audio.py")
    participants = build_items(qa)
    key = [(p["code"], t["id"], t["start"], t["end"], t["dur"]) for p in participants for t in p["takes"]]
    version = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
    data = json.dumps({"version": version, "participants": participants}, separators=(",", ":"))
    page = (PAGE_TEMPLATE.replace("__DATA__", data.replace("</", "<\\/"))
            .replace("__N_PARTICIPANTS__", str(len(participants)))
            .replace("__N_TAKES__", str(len(key))))
    PAGE.write_text(page, encoding="utf-8")
    log.info("listening review page: %d participants, %d takes, version %s -> %s", len(participants), len(key),
             version, rel(PAGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
