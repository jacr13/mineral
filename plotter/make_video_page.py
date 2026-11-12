"""Build a self-contained page for the rebuttal videos.

ICLR rebuttals are double-blind, so the clips need a host that does not carry
the authors' identity: an anonymous GitHub Pages repo from a throwaway account,
or anonymous.4open.science. A personal Drive or Dropbox link leaks the account
name and must not be used.

The page exists rather than a bare directory listing because the labelling
caveats matter and a file name cannot carry them -- which configuration a clip
shows, at what budget, and the return it reached. Reviewer LMQH asked for
expert-learner comparisons specifically, so provenance has to travel with the
video.

    python plotter/make_video_page.py --videos "ICLR Reviews/videos" --out site/
"""

import argparse
import html
import os
import shutil

# The no-break space keeps "for" from being left stranded at the end of a
# line: greedy wrapping moves the whole "for Imitation" unit down instead, so
# the title breaks after "Supervision", at the phrase boundary.
TITLE = ("First-Order Control Using Optimal Transport Supervision "
         "for\u00a0Imitation from Observations")

# The submission's own abstract, word for word, so the page says what the
# method is without paraphrasing it into something the paper does not claim.
# Only the paragraph breaks and the emphasis are added: each entry is one
# paragraph, and text between ** ** is set in bold.
ABSTRACT = [
    "Imitation learning from observations must recover behavior from "
    "state-only demonstrations without known temporal correspondence. "
    "Differentiable simulators provide pathwise policy gradients, but "
    "differentiating through long, contact-rich trajectories can make "
    "optimization unstable.",

    "We introduce FOCUS, which combines **local optimal-transport alignment "
    "with short-horizon differentiable actor\u2013critic learning**. FOCUS "
    "matches agent rollout windows to multiple sampled expert segments using "
    "a soft best-of-K objective that emphasizes locally compatible matches. "
    "The resulting transport-based costs provide differentiable imitation "
    "rewards, while a learned critic bootstraps value beyond the rollout "
    "horizon.",

    "**FOCUS requires neither temporally aligned trajectories nor "
    "expert-aligned initialization or teacher forcing.**",

    "Across contact-rich locomotion tasks in DFlex, including **a "
    "musculoskeletal humanoid controlled by 152 muscle activations**, FOCUS "
    "improves sample efficiency and training stability over adversarial "
    "imitation-from-observation methods and prior differentiable imitation "
    "baselines.",
]


def abstract_html():
    """ABSTRACT -> one <p> per entry, with **...** turned into <strong>."""
    out = []
    for para in ABSTRACT:
        parts = html.escape(para).split("**")
        out.append("<p>" + "".join(
            f"<strong>{t}</strong>" if i % 2 else t
            for i, t in enumerate(parts)) + "</p>")
    return "\n    ".join(out)


# One card per task. ``views`` holds one or more named videos; more than one
# renders a toggle, which is how SNU Humanoid shows the same rollouts with and
# without its muscles instead of splitting them across two cards.
# One card per task. ``views`` holds one or more named videos; more than one
# renders a toggle, which is how SNU Humanoid shows the same rollouts with and
# without its muscles instead of splitting them across two cards.
#
# Every clip has the same panel order -- expert, FOCUS-L2, FOCUS-OT-L2,
# FOCUS-OT-Cos -- and each method keeps one colour across tasks, so the
# captions list the variants in that same order.
CARDS = [
    {
        "title": "SNU Humanoid",
        "note": "152 muscle actuators",
        "views": [("Muscles", "snu_humanoid_muscles.mp4"),
                  ("Skeleton", "snu_humanoid_skeleton.mp4")],
        "caption": (
            "Expert against FOCUS-L2 with no alignment (0.82x expert), "
            "FOCUS-OT-L2 (0.96x) and FOCUS-OT-Cos (0.88x), at the 15M budget. "
            "The DFlex SNU model is a lower body (pelvis, femurs, tibiae, tali "
            "and feet) and is drawn with its bone meshes. The muscle view adds "
            "the 152 muscles, coloured by activation from blue at rest to red "
            "at full contraction. The learner's activations are the policy's "
            "actions and the expert's are its recorded actions."),
    },
    {
        "title": "Humanoid",
        "note": "21 actuated joints",
        "views": [("Rollouts", "humanoid_expert_vs_variants.mp4")],
        "caption": (
            "Expert against FOCUS-L2 with no alignment (0.87x expert), "
            "FOCUS-OT-L2 (0.94x) and FOCUS-OT-Cos (0.98x), at the 15M budget."),
    },
    {
        "title": "Ant",
        "note": "8 actuated joints",
        "views": [("Rollouts", "ant_expert_vs_variants.mp4")],
        "caption": (
            "Expert against FOCUS-L2 with no alignment (0.68x expert), "
            "FOCUS-OT-L2 (1.00x) and FOCUS-OT-Cos (0.91x), at the 10M budget."),
    },
    {
        "title": "Hopper",
        "note": "3 actuated joints",
        "views": [("Rollouts", "hopper_expert_vs_variants.mp4")],
        "caption": (
            "Expert against FOCUS-L2 with no alignment (0.97x expert), "
            "FOCUS-OT-L2 (0.99x) and FOCUS-OT-Cos (0.99x), at the 10M budget."),
    },
]

# Must match METHOD_COLOURS in plotter/render3d.py, which is what the coloured
# square next to each panel label is drawn from.
LEGEND = [("Expert", "#c98a3c"), ("FOCUS-L2", "#8a6fb0"),
          ("FOCUS-OT-L2", "#3f7fbf"), ("FOCUS-OT-Cos", "#4a9e6a")]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FOCUS &mdash; Videos</title>
<style>
  :root {{
    --bg:#fbfcfd; --panel:#ffffff; --fg:#0f141a; --muted:#5a6774;
    --line:#e4eaf0; --accent:#2f6fb8; --accent-soft:#eaf2fb; --chip:#f4f7fa;
    --shadow:0 1px 2px rgba(15,20,26,.04), 0 10px 28px rgba(15,20,26,.07);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0d1115; --panel:#151a20; --fg:#e9eef3; --muted:#98a5b2;
      --line:#222b34; --accent:#6aa9e8; --accent-soft:#16212d; --chip:#101820;
      --shadow:0 1px 2px rgba(0,0,0,.45), 0 12px 32px rgba(0,0,0,.4);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0d1115; --panel:#151a20; --fg:#e9eef3; --muted:#98a5b2;
    --line:#222b34; --accent:#6aa9e8; --accent-soft:#16212d; --chip:#101820;
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 12px 32px rgba(0,0,0,.4);
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",
      Roboto,"Helvetica Neue",Arial,sans-serif; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1060px; margin:0 auto; padding:0 20px; }}
  header .wrap {{ max-width:1160px; }}
  header {{ border-bottom:1px solid var(--line); background:
    radial-gradient(1100px 380px at 12% -20%, var(--accent-soft), transparent 62%),
    radial-gradient(880px 320px at 92% -10%, var(--accent-soft), transparent 58%); }}
  header .wrap {{ padding:68px 20px 52px; text-align:center; }}
  .eyebrow {{ display:inline-block; font-size:.75rem; letter-spacing:.1em;
    text-transform:uppercase; color:var(--accent); font-weight:650;
    border:1px solid var(--line); background:var(--panel);
    padding:5px 12px; border-radius:999px; margin-bottom:22px; }}
  /* `balance` evens the lines instead of filling each one to the edge, which
     is what produced the lopsided title break. The max-width keeps it to two
     lines on a desktop; `balance` then picks where they fall. */
  h1 {{ font-size:clamp(1.65rem,3.6vw,2.45rem); line-height:1.2; margin:0 auto 28px;
    letter-spacing:-.025em; font-weight:700; max-width:1110px;
    text-wrap:pretty; overflow-wrap:break-word; }}
  .abstract {{ max-width:84ch; margin:0 auto; }}
  .abstract .label {{ display:block; font-size:.74rem; letter-spacing:.11em;
    text-transform:uppercase; color:var(--muted); font-weight:650; margin-bottom:9px; }}
  .abstract p + p {{ margin-top:12px; }}
  .abstract p strong {{ color:var(--fg); font-weight:650; }}
  .abstract p {{ margin:0; color:var(--muted); font-size:.99rem;
    text-align:justify; text-justify:inter-word; hyphens:none; }}
  main {{ padding:40px 0 84px; }}
  .note {{ background:var(--panel); border:1px solid var(--line);
    border-left:3px solid var(--accent); border-radius:12px; padding:18px 20px;
    margin:0 0 36px; color:var(--muted); font-size:.95rem; box-shadow:var(--shadow); }}
  .note strong {{ color:var(--fg); }}
  .note p {{ margin:0; }}
  .note p + p {{ margin-top:11px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:7px 18px; margin-top:14px;
    padding-top:13px; border-top:1px solid var(--line); }}
  .legend span {{ display:inline-flex; align-items:center; gap:8px;
    font-size:.88rem; color:var(--fg); }}
  .legend i {{ width:13px; height:13px; border-radius:3px; display:inline-block;
    box-shadow:inset 0 0 0 1px rgba(0,0,0,.12); }}
  .grid {{ display:grid; gap:28px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
    overflow:hidden; box-shadow:var(--shadow); }}
  .head {{ display:flex; align-items:center; gap:13px; flex-wrap:wrap;
    padding:17px 20px 14px; border-bottom:1px solid var(--line); }}
  .head h2 {{ font-size:1.08rem; margin:0; font-weight:650; letter-spacing:-.012em; }}
  .head .n {{ color:var(--muted); font-size:.88rem; }}
  .toggle {{ margin-left:auto; display:inline-flex; background:var(--chip);
    border:1px solid var(--line); border-radius:999px; padding:3px; gap:2px; }}
  .toggle button {{ border:0; background:transparent; color:var(--muted);
    font:inherit; font-size:.85rem; font-weight:600; padding:5px 15px;
    border-radius:999px; cursor:pointer; transition:background .15s,color .15s; }}
  .toggle button:hover {{ color:var(--fg); }}
  .toggle button[aria-pressed="true"] {{ background:var(--accent); color:#fff; }}
  video {{ width:100%; height:auto; display:block; background:#eef1f5; }}
  video[hidden] {{ display:none; }}
  figcaption {{ padding:15px 20px 19px; color:var(--muted); font-size:.92rem; }}
  footer {{ border-top:1px solid var(--line); margin-top:52px; padding:20px 0 0;
    color:var(--muted); font-size:.76rem; line-height:1.55; }}
  footer p {{ margin:0; }}
  @media (max-width:640px) {{ header .wrap {{ padding-top:44px; }}
    .toggle {{ margin-left:0; width:100%; }} .toggle button {{ flex:1; }} }}
</style></head>
<body>
<header><div class="wrap">
  <span class="eyebrow">ICLR &mdash; submission 44492</span>
  <h1>{title}</h1>
  <div class="abstract">
    <span class="label">Abstract</span>
    {abstract}
  </div>
</div></header>
<main><div class="wrap">
<div class="note">
<p><strong>Reading the clips.</strong> Each learner clip is the
configuration reported in the paper, at the reported budget. The panels are not
synchronised, by design: every episode starts from its own randomly perturbed
state, so the gaits drift apart in phase. FOCUS assumes neither expert-aligned
initialisation nor teacher forcing, so a clip should be read for the quality
and periodicity of the motion, not for frame-by-frame agreement with the
expert.</p>
<div class="legend">{legend}</div>
</div>
<div class="grid">
{cards}
</div>
<footer><p>Each clip shows one full episode of 1000 environment steps,
simulated at 60&nbsp;Hz and played at 60&nbsp;fps, so it runs in real time with
nothing cut or retimed. The expert panel replays a recorded demonstration; each
learner panel is a policy rollout. The return given for each method in a
caption is that policy's own, as a multiple of the expert's.</p></footer>
</div></main>
<script>
// Only one clip plays at a time. These are 1000-frame videos; decoding
// several at once costs the viewer's machine a lot for nothing, since only
// one can be watched.
document.querySelectorAll('video').forEach(v => {{
  v.addEventListener('play', () => {{
    document.querySelectorAll('video').forEach(o => {{
      if (o !== v && !o.paused) o.pause();
    }});
  }});
}});

document.querySelectorAll('.card').forEach(card => {{
  const btns = card.querySelectorAll('.toggle button');
  if (!btns.length) return;
  btns.forEach(btn => btn.addEventListener('click', () => {{
    const want = btn.dataset.view;
    btns.forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
    card.querySelectorAll('video').forEach(v => {{
      const on = v.dataset.view === want;
      if (on) {{ v.currentTime = 0; }} else {{ v.pause(); }}
      v.hidden = !on;
      if (on) {{ v.play().catch(() => {{}}); }}
    }});
  }}));
}});
</script>
</body></html>
"""

CARD = """<figure class="card">
  <div class="head"><h2>{title}</h2><span class="n">{note}</span>{toggle}</div>
  {videos}
  <figcaption>{caption}</figcaption>
</figure>"""

TOGGLE = """<div class="toggle">{buttons}</div>"""
TOGGLE_BTN = """<button type="button" data-view="{view}" aria-pressed="{pressed}">{view}</button>"""
VIDEO = ('<video data-view="{view}" controls loop muted playsinline '
         'preload="metadata" src="{src}"{hidden}></video>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="ICLR Reviews/videos")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cards, n = [], 0
    for card in CARDS:
        views = [(name, fn) for name, fn in card["views"]
                 if os.path.exists(os.path.join(args.videos, fn))]
        if not views:
            print(f"  skip (no video): {card['title']}")
            continue
        for _, fn in views:
            src = os.path.join(args.videos, fn)
            shutil.copy(src, os.path.join(args.out, fn))
            print(f"  added: {fn}  ({os.path.getsize(src) / 1e6:.1f} MB)")
        toggle = ""
        if len(views) > 1:
            toggle = TOGGLE.format(buttons="".join(
                TOGGLE_BTN.format(view=html.escape(name),
                                  pressed="true" if i == 0 else "false")
                for i, (name, _) in enumerate(views)))
        videos = "\n  ".join(
            VIDEO.format(view=html.escape(name), src=html.escape(fn),
                         hidden="" if i == 0 else " hidden")
            for i, (name, fn) in enumerate(views))
        cards.append(CARD.format(title=html.escape(card["title"]),
                                 note=html.escape(card["note"]),
                                 toggle=toggle, videos=videos,
                                 caption=html.escape(card["caption"])))
        n += 1
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(name)}</span>'
                     for name, c in LEGEND)
    page = PAGE.format(title=html.escape(TITLE),
                       abstract=abstract_html(),
                       legend=legend, cards="\n".join(cards))
    index = os.path.join(args.out, "index.html")
    with open(index, "w") as f:
        f.write(page)
    # GitHub Pages runs Jekyll over a published directory unless this exists;
    # the build wipes the output folder, so write it here rather than by hand.
    open(os.path.join(args.out, ".nojekyll"), "w").close()
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"\nwrote {index}  ({n} cards, {total / 1e6:.1f} MB total)")
    print("Host anonymously: an anonymous GitHub Pages repo, or "
          "anonymous.4open.science. Do not use a personal Drive/Dropbox link.")


if __name__ == "__main__":
    main()
