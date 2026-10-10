"""Portable HTML evidence report with escaped caller-controlled content."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    def read(name: str) -> dict:
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    spec, status, evaluation = read("run.lock.json"), read("status.json"), read("evaluation.json")
    def dump(value: object) -> str:
        return escape(json.dumps(value, ensure_ascii=False, indent=2))

    links = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "report.html":
            relative = path.relative_to(run_dir).as_posix()
            links.append(f'<li><a href="./{quote(relative, safe="/")}">{escape(relative)}</a></li>')
    verdict = escape(str(evaluation.get("verdict", "INCONCLUSIVE")))
    report = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniRSI experiment report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;background:#fafafa;color:#182432}}
pre{{background:white;border:1px solid #ddd;padding:16px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}}
h1,h2{{line-height:1.3}}a{{color:#1558b0}}.verdict{{font-size:24px;font-weight:700}}</style>
<h1>OmniRSI experiment report</h1><p class="verdict">{verdict}</p>
<p>Caller-declared units, scope and workload are preserved. These results do not establish
statistical significance or accelerator support. Successful quality validation covers only the supplied command.</p>
<h2>Evaluation</h2><pre>{dump(evaluation)}</pre>
<h2>Run status</h2><pre>{dump(status)}</pre>
<h2>Frozen CLI contract</h2><pre>{dump(spec)}</pre>
<h2>Environment</h2><pre>{dump(read('environment.json'))}</pre>
<h2>Evidence files</h2><ul>{''.join(links)}</ul></html>"""
    destination = run_dir / "report.html"
    destination.write_text(report, encoding="utf-8")
    return destination
