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
    ("snu_humanoid_expert_vs_variants_bones.mp4", "SNU Humanoid (152 muscles)",
     "Expert demonstration against three alignment objectives, all at the 15M "
     "interaction budget: FOCUS-OT-L2 (0.99x expert), FOCUS-OT-Cos (0.95x) and "
     "FOCUS-L2 with no alignment (0.88x). Rendered from the model's own bone "
     "meshes; the dflex SNU model is a lower body, so pelvis, femurs, tibiae, "
     "tali and foot segments are the whole articulation."),
    ("humanoid_expert_vs_variants.mp4", "Humanoid",
     "Expert against FOCUS-OT-Cos (1.00x expert), FOCUS-OT-L2 (0.97x) and "
     "FOCUS-L2 with no alignment (0.90x), at the 15M budget."),
    ("ant_expert_vs_focus_ot_l2.mp4", "Ant",
     "Expert against FOCUS-OT-L2 (1.01x expert) at the 10M budget."),
    ("hopper_expert_vs_focus_ot_l2.mp4", "Hopper",
     "Expert against FOCUS-OT-L2 (1.00x expert) at the 10M budget."),
    ("hopper_expert_vs_focus_l2.mp4", "Hopper — no alignment",
     "The same backbone with the OT matching removed, for comparison."),
    ("hopper_expert_vs_ild.mp4", "Hopper — ILD baseline",
     "ILD, which uses full-trajectory matching with teacher forcing."),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expert and learner rollouts</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1b1f24; --muted: #5b6672;
    --line: #e4e8ec; --card: #f7f9fb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #14171a; --fg: #e8ecf0; --muted: #a3adb8;
      --line: #2a2f35; --card: #1b1f23;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #14171a; --fg: #e8ecf0; --muted: #a3adb8;
    --line: #2a2f35; --card: #1b1f23;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 40px 16px 72px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); margin: 0 0 28px; }}
  .note {{ background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 16px; margin: 0 0 32px; color: var(--muted); }}
  .item {{ border-top: 1px solid var(--line); padding: 26px 0 4px; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 10px; }}
  video {{ width: 100%; height: auto; border-radius: 10px;
    background: var(--card); border: 1px solid var(--line); display: block; }}
  p.cap {{ color: var(--muted); margin: 10px 0 0; font-size: 0.93rem; }}
  footer {{ margin-top: 44px; color: var(--muted); font-size: 0.88rem;
    border-top: 1px solid var(--line); padding-top: 16px; }}
</style></head>
<body><div class="wrap">
<h1>Expert and learner rollouts</h1>
<p class="sub">Supplementary material for submission 44492.</p>
<div class="note">
Each clip shows the expert demonstration beside one or more learned policies,
rendered through the same forward-kinematics code path so a difference on
screen is a difference in the motion. Returns are expert-normalized at the
interaction budget stated in each caption.
<br><br>
The checkpoints of the exact runs reported in the paper were lost to a storage
purge, so every learner clip is a <em>rerun of the reported configuration</em>
at the reported budget, not the individual run behind a number in a table. The
configuration and achieved return are given for each.
</div>
{items}
<footer>Videos are H.264 MP4. {count} clips, rendered headless; no audio.</footer>
</div></body></html>
"""

ITEM = """<div class="item">
  <h2>{title}</h2>
  <video controls loop muted playsinline preload="metadata" src="{src}"></video>
  <p class="cap">{caption}</p>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="ICLR Reviews/videos")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items, n = [], 0
    for fn, title, caption in ENTRIES:
        src = os.path.join(args.videos, fn)
        if not os.path.exists(src):
            print(f"  skip (missing): {fn}")
            continue
        shutil.copy(src, os.path.join(args.out, fn))
        items.append(ITEM.format(title=html.escape(title), src=html.escape(fn),
                                 caption=html.escape(caption)))
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
