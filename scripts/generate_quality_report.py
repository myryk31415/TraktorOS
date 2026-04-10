#!/usr/bin/env python3
"""Generate an HTML report from quality_report.json showcasing good and bad images."""
import json
import base64
import sys
from pathlib import Path

DATA_DIR = Path('data/training')
REPORT_PATH = DATA_DIR / 'quality_report.json'
OUTPUT_PATH = Path('quality_report.html')
SAMPLES_PER_CATEGORY = 5


def img_to_base64(path):
    try:
        data = path.read_bytes()
        suffix = path.suffix.lower()
        mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'bmp': 'bmp', 'webp': 'webp'}.get(suffix.lstrip('.'), 'jpeg')
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


def metric_bar(value, low, high, invert=False):
    pct = max(0, min(100, (value - low) / (high - low) * 100))
    if invert:
        pct = 100 - pct
    color = f"hsl({pct * 1.2}, 70%, 50%)"
    return f'<div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def card(r, img_b64):
    issues = r.get('issues', [])
    status_cls = 'bad' if issues else 'good'
    status_label = ', '.join(issues) if issues else 'Good quality'
    badges = ''.join(f'<span class="badge badge-{status_cls}">{i}</span>' for i in issues) if issues else '<span class="badge badge-good">OK</span>'

    return f'''<div class="card {status_cls}">
  <img src="{img_b64}" alt="{r['file']}" loading="lazy">
  <div class="card-body">
    <div class="card-title" title="{r['file']}">{Path(r['file']).name}</div>
    <div class="badges">{badges}</div>
    <table>
      <tr><td>Blur</td><td>{r.get('blur_score','—')}</td><td>{metric_bar(r.get('blur_score',0), 0, 1000)}</td></tr>
      <tr><td>Brightness</td><td>{r.get('brightness','—')}</td><td>{metric_bar(r.get('brightness',0), 0, 255)}</td></tr>
      <tr><td>Contrast</td><td>{r.get('contrast','—')}</td><td>{metric_bar(r.get('contrast',0), 0, 100)}</td></tr>
      <tr><td>BRISQUE</td><td>{r.get('brisque','—')}</td><td>{metric_bar(r.get('brisque',50), 0, 100, invert=True)}</td></tr>
      <tr><td>NIMA</td><td>{r.get('nima','—')}</td><td>{metric_bar(r.get('nima',5), 1, 10)}</td></tr>
    </table>
    <div class="meta">{r.get('resolution','?')}</div>
  </div>
</div>'''


if __name__ == '__main__':
    if not REPORT_PATH.exists():
        print(f"{REPORT_PATH} not found. Run analyze_quality.py first.")
        sys.exit(1)

    results = json.load(open(REPORT_PATH))
    good = [r for r in results if r.get('sufficient_for_detection')]
    bad = [r for r in results if not r.get('sufficient_for_detection')]

    # Group bad by issue type
    by_issue = {}
    for r in bad:
        for issue in r.get('issues', []):
            by_issue.setdefault(issue, []).append(r)

    # Pick diverse samples
    good_sorted = sorted(good, key=lambda r: -r.get('blur_score', 0))
    good_samples = good_sorted[:SAMPLES_PER_CATEGORY]

    sections_html = ""

    # Good section
    cards = ""
    for r in good_samples:
        img_b64 = img_to_base64(DATA_DIR / r['file'])
        cards += card(r, img_b64)
    sections_html += f'<h2>✓ Good Quality ({len(good)} images)</h2><div class="grid">{cards}</div>'

    # Bad sections by issue type
    for issue, items in sorted(by_issue.items(), key=lambda x: -len(x[1])):
        samples = sorted(items, key=lambda r: r.get('blur_score', 0))[:SAMPLES_PER_CATEGORY]
        cards = ""
        for r in samples:
            img_b64 = img_to_base64(DATA_DIR / r['file'])
            cards += card(r, img_b64)
        sections_html += f'<h2>✗ {issue.title()} ({len(items)} images)</h2><div class="grid">{cards}</div>'

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TraktorOS Image Quality Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0c1728;color:#e5eef7;padding:2rem}}
h1{{text-align:center;margin-bottom:.5rem;font-size:2rem}}
.subtitle{{text-align:center;color:#8fb5ff;margin-bottom:2rem}}
.stats{{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:2.5rem}}
.stat{{background:rgba(10,18,32,0.72);border:1px solid rgba(148,163,184,0.18);border-radius:12px;padding:1rem 1.5rem;text-align:center;min-width:120px}}
.stat-value{{font-size:1.5rem;font-weight:700;color:#f8fbff}}
.stat-label{{font-size:.8rem;color:#8fb5ff;margin-top:.25rem}}
h2{{margin:2rem 0 1rem;color:#f8fbff;border-bottom:1px solid rgba(148,163,184,0.18);padding-bottom:.5rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}}
.card{{background:rgba(10,18,32,0.72);border:1px solid rgba(148,163,184,0.18);border-radius:16px;overflow:hidden}}
.card.bad{{border-color:rgba(239,68,68,0.4)}}
.card.good{{border-color:rgba(34,197,94,0.4)}}
.card img{{width:100%;height:180px;object-fit:cover;display:block}}
.card-body{{padding:1rem}}
.card-title{{font-size:.75rem;color:#8fb5ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:.5rem}}
.badges{{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.75rem}}
.badge{{font-size:.7rem;padding:.2rem .5rem;border-radius:6px;font-weight:600}}
.badge-bad{{background:rgba(239,68,68,0.2);color:#fca5a5}}
.badge-good{{background:rgba(34,197,94,0.2);color:#86efac}}
table{{width:100%;font-size:.8rem;border-collapse:collapse}}
td{{padding:.2rem 0}}
td:first-child{{color:#8fb5ff;width:70px}}
td:nth-child(2){{width:45px;text-align:right;padding-right:.5rem;font-variant-numeric:tabular-nums}}
.bar{{flex:1;height:6px;background:rgba(148,163,184,0.15);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px}}
.meta{{font-size:.7rem;color:#64748b;margin-top:.5rem;text-align:right}}
</style></head><body>
<h1>🚜 TraktorOS Image Quality Report</h1>
<p class="subtitle">{len(results)} images analyzed</p>
<div class="stats">
  <div class="stat"><div class="stat-value">{len(results)}</div><div class="stat-label">Total</div></div>
  <div class="stat"><div class="stat-value" style="color:#86efac">{len(good)}</div><div class="stat-label">Good</div></div>
  <div class="stat"><div class="stat-value" style="color:#fca5a5">{len(bad)}</div><div class="stat-label">Issues</div></div>
</div>
{sections_html}
</body></html>'''

    OUTPUT_PATH.write_text(html)
    print(f"Report saved to {OUTPUT_PATH}")
    print(f"  {len(good)} good, {len(bad)} with issues, {len(results)} total")
