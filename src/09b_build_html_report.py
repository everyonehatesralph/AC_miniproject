"""
09b_build_html_report.py -- browser-viewable version of every report.

Converts the generated Markdown reports and the documentation into self-contained HTML pages:
figures are embedded (base64), one stylesheet is inlined in every page, and each page carries a
navigation bar. An overview page shows the headline results (the README block written by
09_build_final_report.py) and the deliverable status matrix.

Output: reports/html/index.html -- double-click to open in any browser; works offline, and every
page is a single file that can be sent on its own. Nothing is uploaded anywhere.

Usage: .venv/Scripts/python.exe src/09b_build_html_report.py
"""

from __future__ import annotations

import base64
import html
import re
import sys
from datetime import datetime
from pathlib import Path

import markdown
import pandas as pd

from common import PROJECT_ROOT, REPORTS, setup_logging

OUT = REPORTS / "html"
DOCS = PROJECT_ROOT / "docs"
PAGES = [  # (file stem, navigation label, source markdown, one-line description)
    ("index", "Overview", None, "headline results and the status of the 12 expected outputs"),
    ("final_report", "Final report", REPORTS / "final_group_research_report.md",
     "the group research report (Phases 14–19)"),
    ("data_quality", "Data quality", REPORTS / "data_quality_report.md",
     "recordings, retake decisions, audio QA, segmentation review"),
    ("exploratory", "Exploratory analysis", REPORTS / "exploratory_analysis.md",
     "affect distributions, feature–affect statistics, confounds"),
    ("model_evaluation", "Model evaluation", REPORTS / "model_evaluation.md",
     "speaker-independent results, confusion matrices, model comparison"),
    ("lexical", "Lexical control", REPORTS / "lexical_control_analysis.md",
     "same-word comparison and the word-identity confound"),
    ("validation", "Validation", REPORTS / "validation_report.md", "automated checks of the whole project"),
    ("phase0", "Phase 0 audit", REPORTS / "phase0_data_audit.md", "initial inventory of metadata and recordings"),
    ("documentation", "Documentation", DOCS / "DOCUMENTATION.md", "how every step works"),
    ("decisions", "Decision log", DOCS / "decision_log.md", "every methodological decision and its status"),
    ("data_dictionary", "Data dictionary", DOCS / "data_dictionary.md", "every column of every data file"),
]
GUIDES = [  # pages listed on the overview under "Guides and next steps" (kept out of the navigation bar)
    ("running", "How to run it", DOCS / "RUNNING.md", "the team guide: install, run, check, troubleshoot"),
    ("listening_review", "Listening review", REPORTS / "listening_review_report.md",
     "blind listening review: agreement between listeners and the decisions used"),
    ("supplied_audio", "Supplied audio audit", REPORTS / "supplied_audio_audit.md",
     "audio delivered later (CS2B_v2) checked against the originals and the sheet"),
    ("collection_checklist", "Data collection checklist", DOCS / "data_collection_checklist.md",
     "one page to bring to every recording session"),
    ("analysis_plan", "Pre-registered analysis plan", DOCS / "analysis_plan_pooled_study.md",
     "hypotheses, power and tests for a larger or pooled study"),
    ("consent", "Consent documentation", PROJECT_ROOT / "consent" / "README.md", "how to supply the consent log"),
    ("email_instructor", "Questions for the instructor", DOCS / "next_steps" / "01_email_to_instructor.md",
     "draft email with the open protocol questions"),
    ("message_collectors", "Requests to the collectors", DOCS / "next_steps" / "02_message_to_collectors.md",
     "draft message: missing words, recordings, dates"),
]
LOCAL_LINKS = [  # (label, path relative to reports/html, description): files that exist only after a local run
    ("Listening review tool", "../../data/review/listening_review.html",
     "the blind listening page (needs the audio from a local pipeline run)"),
    ("Presentation slides", "../presentation/G3_results_presentation.pptx", "12 slides with speaker notes (PowerPoint)"),
]

CSS = """
:root { color-scheme: light; --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --good:#006300; --good-bg:#e3f3e3; --critical:#b42f2f; --critical-bg:#fbe7e7;
  --code-bg:#f0efec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#0d0d0d;
  --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5;
  --good:#5fd35f; --good-bg:#16301a; --critical:#f08a8a; --critical-bg:#3a1a1a; --code-bg:#262624; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --good:#5fd35f; --good-bg:#16301a; --critical:#f08a8a;
  --critical-bg:#3a1a1a; --code-bg:#262624; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
nav { position: sticky; top: 0; z-index: 10; background: var(--surface); border-bottom: 1px solid var(--line); }
nav .inner { max-width: 1160px; margin: 0 auto; padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 4px 6px;
  align-items: center; }
nav .brand { font-weight: 600; margin-right: 10px; white-space: nowrap; }
nav a { color: var(--ink-2); text-decoration: none; padding: 4px 9px; border-radius: 6px; font-size: 13.5px; }
nav a:hover { background: var(--code-bg); color: var(--ink); }
nav a.current { background: var(--accent); color: #fff; }
main { max-width: 1160px; margin: 0 auto; padding: 24px 16px 56px; }
article { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 28px 32px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 12px; }
h2 { font-size: 20px; margin: 34px 0 10px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 16.5px; margin: 24px 0 8px; }
p, li { color: var(--ink); } li { margin: 3px 0; }
a { color: var(--accent); }
em { color: var(--ink-2); }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 0.88em;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; overflow-wrap: anywhere; }
pre { background: var(--code-bg); padding: 12px 14px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; }
.table-wrap { overflow-x: auto; margin: 12px 0 18px; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
th { background: var(--code-bg); font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
img { display: block; max-width: 100%; height: auto; margin: 14px 0; border-radius: 8px;
  border: 1px solid var(--line); background: #fcfcfb; }
blockquote { margin: 12px 0; padding: 8px 14px; border-left: 3px solid var(--accent); background: var(--code-bg);
  border-radius: 0 8px 8px 0; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600;
  white-space: nowrap; }
.badge.ok { color: var(--good); background: var(--good-bg); }
.badge.blocked { color: var(--critical); background: var(--critical-bg); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin: 14px 0 6px; }
.card { display: block; text-decoration: none; color: var(--ink); background: var(--bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 14px; }
.card:hover { border-color: var(--accent); }
.card b { display: block; margin-bottom: 2px; }
.card span { color: var(--ink-2); font-size: 13px; }
.note { color: var(--ink-2); font-size: 13px; }
footer { max-width: 1160px; margin: 0 auto; padding: 0 16px 36px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 640px) { article { padding: 18px 16px; } h1 { font-size: 22px; } }
"""

log = setup_logging("html_report", REPORTS / "logs" / "09b_build_html_report.log")


def embed_image(src: str, base: Path) -> str:
    path = (base / src).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"figure referenced in {base} not found: {src}")
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def fix_links(body: str, base: Path, page_of: dict[Path, str]) -> str:
    def img(m):
        return f'<img{m.group(1)}src="{embed_image(html.unescape(m.group(2)), base)}"'

    def link(m):
        target, anchor = m.group(1), m.group(2) or ""
        resolved = (base / html.unescape(target)).resolve()
        if resolved in page_of:
            return f'<a href="{page_of[resolved]}.html{anchor}"'
        try:
            relative = Path("../..") / resolved.relative_to(PROJECT_ROOT)
            return f'<a href="{relative.as_posix()}{anchor}"'
        except ValueError:
            return m.group(0)

    body = re.sub(r'<img([^>]*?)src="([^"]+)"', img, body)
    body = re.sub(r'<a href="([^"#:]+)(#[^"]*)?"', link, body)
    return re.sub(r"<table>(.*?)</table>", r'<div class="table-wrap"><table>\1</table></div>', body, flags=re.S)


def page(stem: str, label: str, body: str, stamp: str) -> str:
    nav = "".join(f'<a href="{s}.html"{" class=current" if s == stem else ""}>{html.escape(lbl)}</a>'
                  for s, lbl, _, _ in PAGES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Group 3 · {html.escape(label)}</title><style>{CSS}</style></head>
<body><nav><div class="inner"><span class="brand">Group 3 · Bisaya SER</span>{nav}</div></nav>
<main><article>{body}</article></main>
<footer>Generated by <code>src/09b_build_html_report.py</code> on {stamp} from the pipeline outputs (every number comes
from the data). Pseudonymous research data: do not share outside the research team until the consent documentation
is verified.</footer></body></html>"""


def overview(page_of: dict[Path, str]) -> str:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    results_html = markdown.markdown(block, extensions=["tables", "sane_lists"])
    status = pd.read_csv(REPORTS / "deliverable_status_matrix.csv")
    rows = []
    for r in status.itertuples(index=False):
        ok = r.Status == "COMPLETE"
        badge = f'<span class="badge {"ok" if ok else "blocked"}">{"✓ COMPLETE" if ok else "✕ " + html.escape(r.Status)}</span>'
        rows.append(f"<tr><td>{html.escape(r.Deliverable)}</td><td>{badge}</td><td>{html.escape(str(r.Evidence))}</td>"
                    f"<td>{html.escape(str(r._4) if str(r._4) != 'nan' else '—')}</td></tr>")
    table = ("<div class='table-wrap'><table><thead><tr><th>Expected output (§XVIII)</th><th>Status</th><th>Evidence</th>"
             "<th>Remaining issue</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    cards = "".join(f'<a class="card" href="{s}.html"><b>{html.escape(lbl)}</b><span>{html.escape(desc)}</span></a>'
                    for s, lbl, _, desc in PAGES if s != "index")
    guides = "".join(f'<a class="card" href="{s}.html"><b>{html.escape(lbl)}</b><span>{html.escape(desc)}</span></a>'
                     for s, lbl, _, desc in GUIDES)
    guides += "".join(f'<a class="card" href="{href}"><b>{html.escape(lbl)}</b><span>{html.escape(desc)}</span></a>'
                      for lbl, href, desc in LOCAL_LINKS)
    figs = REPORTS.parent / "figures"
    fig_html = "".join(
        f"<h3>{html.escape(title)}</h3><img alt='{html.escape(title)}' src='{embed_image(str(path), Path('.'))}'>"
        for title, path in (("Speaker-independent balanced accuracy", figs / "evaluation" / "fig_eval_balanced_accuracy.png"),
                            ("Sensitivity analyses", figs / "evaluation" / "fig_eval_sensitivity.png"),
                            ("Lexical control: Kapoy recordings only", figs / "lexical" / "fig_lexical_kapoy_features_vs_affect.png"))
        if path.is_file())
    return (f"<h1>Bisaya Post-Class Speech and Affective State Recognition</h1>"
            f"<p class='note'>Group 3 · Second-Year Computer Science students · pilot study. Start with the cards below, "
            f"or read the <a href='final_report.html'>final report</a>.</p>"
            f"<div class='cards'>{cards}</div>"
            f"<h2>Guides and next steps</h2><div class='cards'>{guides}</div>"
            f"<h2>Results at a glance</h2>{results_html}"
            f"<h2>Expected research outputs: status</h2>{table}"
            f"<h2>Key figures</h2>{fig_html}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    page_of = {src.resolve(): stem for stem, _, src, _ in PAGES + GUIDES if src is not None}
    md = markdown.Markdown(extensions=["tables", "sane_lists", "fenced_code", "toc"])
    written = []
    for stem, label, src, _ in PAGES + GUIDES:
        if src is None:
            body = overview(page_of)
        else:
            if not src.is_file():
                raise FileNotFoundError(f"report missing: {src}")
            md.reset()
            body = fix_links(md.convert(src.read_text(encoding="utf-8")), src.parent, page_of)
        (OUT / f"{stem}.html").write_text(page(stem, label, body, stamp), encoding="utf-8")
        written.append(f"{stem}.html")
    sizes = {f: round((OUT / f).stat().st_size / 1e6, 2) for f in written}
    log.info("wrote %d pages to %s (MB: %s)", len(written), OUT.relative_to(PROJECT_ROOT).as_posix(), sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
