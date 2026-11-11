"""Build a self-contained page for the rebuttal videos.

ICLR rebuttals are double-blind, so the clips need a host that does not carry
the authors' identity: an anonymous GitHub Pages repo from a throwaway account,
or anonymous.4open.science. A personal Drive or Dropbox link leaks the account
name and must not be used.

The page exists rather than a bare directory listing because the labelling
caveats matter and a file name cannot carry them -- which run a clip comes
from, at what budget, and whether it is the reported configuration or a
rerun. Reviewer LMQH asked specifically for expert-learner comparisons, so the
provenance has to travel with the video.

    python plotter/make_video_page.py --videos "ICLR Reviews/videos" --out site/
"""

import argparse
import html
import os
import shutil

# (filename, task, caption). Captions state provenance, because none of the
# reported runs' checkpoints survive -- every learner clip is a rerun of a
# configuration, not the exact run behind a number in the paper.
ENTRIES = [
    ("snu_humanoid_expert_vs_variants_bones.mp4", "SNU Humanoid — 152 muscle actuators",
     "skeleton",
     "Expert demonstration against three alignment objectives, all at the 15M "
     "interaction budget: FOCUS-OT-L2 (0.99x expert), FOCUS-OT-Cos (0.95x) and "
     "FOCUS-L2 with no alignment (0.88x). Drawn from the model's own bone "
     "meshes. The dflex SNU model is a lower body, so pelvis, femurs, tibiae, "
     "tali and foot segments are the whole articulation."),
    ("snu_humanoid_muscles_over_skeleton.mp4", "SNU Humanoid — muscle activations",
     "muscles",
     "The same skeleton with all 152 muscles drawn over it, coloured from blue "
     "at rest to red at full contraction. Activation is the policy's own action "
     "divided by each muscle's strength, which is exactly the quantity the "
     "simulator applies; the expert's comes from its recorded actions."),
    ("humanoid_expert_vs_variants.mp4", "Humanoid", "capsules",
     "Expert against FOCUS-OT-Cos (1.00x expert), FOCUS-OT-L2 (0.97x) and "
     "FOCUS-L2 with no alignment (0.90x), at the 15M budget."),
    ("hopper_expert_vs_variants.mp4", "Hopper", "capsules",
     "Expert against FOCUS-OT-L2 (0.99x expert), FOCUS-OT-Cos (0.99x) and "
     "FOCUS-L2 with no alignment (0.98x), at the 10M budget. These are the "
     "runs reported in the paper: Hopper is the one task whose original "
     "checkpoints survived."),
    ("ant_expert_vs_focus_ot_l2.mp4", "Ant", "capsules",
     "Expert against FOCUS-OT-L2 (1.01x expert) at the 10M budget."),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FOCUS — expert and learner rollouts</title>
<style>
  :root {{
    --bg:#fbfcfd; --panel:#ffffff; --fg:#10151b; --muted:#5d6b7a;
    --line:#e3e9ef; --accent:#2f6fb8; --accent-soft:#eaf2fb;
    --shadow:0 1px 2px rgba(16,21,27,.04), 0 8px 24px rgba(16,21,27,.06);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0e1216; --panel:#151b21; --fg:#e9eef3; --muted:#9aa7b4;
      --line:#232c35; --accent:#6aa9e8; --accent-soft:#17222e;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0e1216; --panel:#151b21; --fg:#e9eef3; --muted:#9aa7b4;
    --line:#232c35; --accent:#6aa9e8; --accent-soft:#17222e;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }}
  * {{ box-sizing:border-box; }}
  html {{ scroll-behavior:smooth; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.6 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing:antialiased; }}
  .hero {{ position:relative; overflow:hidden; border-bottom:1px solid var(--line);
    background:
      radial-gradient(1200px 400px at 15% -10%, var(--accent-soft), transparent 60%),
      radial-gradient(900px 360px at 95% 0%, var(--accent-soft), transparent 55%); }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:0 20px; }}
  .hero .wrap {{ padding-top:64px; padding-bottom:44px; }}
  .eyebrow {{ display:inline-block; font-size:.76rem; letter-spacing:.09em;
    text-transform:uppercase; color:var(--accent); font-weight:650;
    border:1px solid var(--line); background:var(--panel);
    padding:5px 11px; border-radius:999px; margin-bottom:18px; }}
  h1 {{ font-size:clamp(1.9rem, 4.4vw, 2.9rem); line-height:1.12; margin:0 0 14px;
    letter-spacing:-.025em; font-weight:700; }}
  .lede {{ font-size:1.08rem; color:var(--muted); margin:0; max-width:64ch; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:26px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:9px 14px; font-size:.88rem; color:var(--muted); box-shadow:var(--shadow); }}
  .stat b {{ color:var(--fg); font-weight:650; }}
  main {{ padding:40px 0 80px; }}
  .note {{ background:var(--panel); border:1px solid var(--line);
    border-left:3px solid var(--accent); border-radius:12px; padding:18px 20px;
    margin:0 0 40px; color:var(--muted); font-size:.95rem; box-shadow:var(--shadow); }}
  .note strong {{ color:var(--fg); }}
  .grid {{ display:grid; gap:30px; }}
  figure {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:14px; overflow:hidden; box-shadow:var(--shadow); }}
  .bar {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
    padding:16px 20px 13px; border-bottom:1px solid var(--line); }}
  .bar h2 {{ font-size:1.02rem; margin:0; font-weight:650; letter-spacing:-.01em; }}
  .tag {{ font-size:.73rem; letter-spacing:.05em; text-transform:uppercase;
    color:var(--accent); background:var(--accent-soft); border-radius:999px;
    padding:3px 9px; font-weight:650; }}
  video {{ width:100%; height:auto; display:block; background:#eef1f5; }}
  figcaption {{ padding:15px 20px 19px; color:var(--muted); font-size:.92rem; }}
  footer {{ border-top:1px solid var(--line); margin-top:56px; padding:22px 0 0;
    color:var(--muted); font-size:.87rem; }}
  @media (max-width:640px) {{ .hero .wrap {{ padding-top:44px; }} }}
</style></head>
<body>
<header class="hero"><div class="wrap">
  <span class="eyebrow">Submission 44492 — supplementary</span>
  <h1>Learning locomotion from unaligned<br>state-only demonstrations</h1>
  <p class="lede">Expert demonstrations beside the policies FOCUS learns from
  them, on four DFlex systems. Each pair is rendered through the same
  forward-kinematics path, so a difference on screen is a difference in the
  motion — not in the drawing.</p>
  <div class="stats">
    <span class="stat"><b>{count}</b> clips</span>
    <span class="stat"><b>4</b> environments</span>
    <span class="stat"><b>152</b> muscle actuators on SNU Humanoid</span>
    <span class="stat">returns <b>expert-normalized</b></span>
  </div>
</div></header>
<main><div class="wrap">
<div class="note">
<strong>What you are looking at.</strong> Every learner clip is a rerun of the
reported configuration at the reported interaction budget. The checkpoints of
the exact runs behind the numbers in the paper did not survive a storage purge,
so these are the same setup and budget rather than the same individual run; the
achieved return is given for each. The expert is drawn in amber and every learned policy in a cool colour. Camera distance is fixed within a comparison and tracks the body in both directions, so panels are directly comparable in scale.
</div>
<div class="grid">
{items}
</div>
<footer>H.264 MP4, no audio, rendered headless. Playback is muted and looped.</footer>
</div></main></body></html>
"""

ITEM = """<figure>
  <div class="bar"><h2>{title}</h2><span class="tag">{tag}</span></div>
  <video controls loop muted playsinline preload="metadata" src="{src}"></video>
  <figcaption>{caption}</figcaption>
</figure>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="ICLR Reviews/videos")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items, n = [], 0
    for fn, title, tag, caption in ENTRIES:
        src = os.path.join(args.videos, fn)
        if not os.path.exists(src):
            print(f"  skip (missing): {fn}")
            continue
        shutil.copy(src, os.path.join(args.out, fn))
        items.append(ITEM.format(title=html.escape(title), src=html.escape(fn),
                                 tag=html.escape(tag), caption=html.escape(caption)))
        n += 1
        print(f"  added: {fn}  ({os.path.getsize(src) / 1e6:.1f} MB)")
    page = PAGE.format(items="\n".join(items), count=n)
    index = os.path.join(args.out, "index.html")
    with open(index, "w") as f:
        f.write(page)
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"\nwrote {index}  ({n} clips, {total / 1e6:.1f} MB total)")
    print("Host anonymously: an anonymous GitHub Pages repo, or "
          "anonymous.4open.science. Do not use a personal Drive/Dropbox link.")


if __name__ == "__main__":
    main()
