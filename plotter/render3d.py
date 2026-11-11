"""Shaded 3D rendering of dflex rollouts, headless, via three.js in Chromium.

The matplotlib path in ``render_rollout.py`` draws flat 2D capsule silhouettes.
That is legible but does not look like a physics viewer: no ground plane, no
shadow, no lighting, so limbs read as cut-out shapes rather than solids.

This renders the same geometry with a real renderer. Chromium is driven
headless with SwiftShader (no GPU needed), three.js builds capsule meshes from
the segments ``ShapeSet`` already produces, and a directional light with shadow
mapping casts onto a checkered ground -- the look of the SHAC project page and
of MuJoCo's default viewer.

Frames are captured as sprite sheets rather than one screenshot per frame:
Chromium costs ~1.5 s to start, so 240 individual invocations would take six
minutes per clip, while six sheets of 40 frames take about fifteen seconds.

Needs: chromium-browser (or chromium) on PATH, and plotter/viewer/three.module.js.
Note Snap-packaged Chromium cannot read files under /tmp, so the generated page
is written next to the output.
"""

import base64
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THREE = os.path.join(HERE, "viewer", "three.module.js")

# One colour per panel, and enough of them that a 4-panel comparison never
# reuses the expert's blue for a learner -- which otherwise reads as "this
# policy is the demonstration".
# The expert is warm, every learner is cool, so the reference reads as the
# reference at a glance. Blue is deliberately NOT the expert: in the muscle
# view blue means a relaxed muscle, and reusing it for the demonstration body
# collides with that legend.
PALETTE = {
    "expert": "0xc98a3c",
    "learner": "0x3f7fbf",
    "third": "0x4a9e6a",
    "fourth": "0x8a6fb0",
    "fifth": "0x3f9fa8",
}


def _chromium():
    for b in ("chromium-browser", "chromium", "google-chrome"):
        p = shutil.which(b)
        if p:
            return p
    raise RuntimeError("no chromium on PATH; cannot render in 3D")


def _frame_payload(frame, style):
    """One frame -> {caps: [[x0,y0,z0,x1,y1,z1,r], ...], lines: [[pts, act], ...]}."""
    segs, muscles = frame
    caps = [[float(p0[0]), float(p0[1]), float(p0[2]),
             float(p1[0]), float(p1[1]), float(p1[2]), float(r)]
            for p0, p1, r in segs]
    lines = []
    if style == "muscles" and muscles:
        for pl, act in muscles:
            lines.append([[[float(x), float(y), float(z)] for x, y, z in pl], float(act)])
    return {"caps": caps, "lines": lines}


def _page(payload, cell, cols, rows, colour, style, cam):
    """Self-contained HTML. The data is inlined because fetch() is blocked on
    file:// and we would otherwise need a web server just to read our own JSON."""
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"""<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:#ffffff">
<canvas id="c" width="{cols * cell}" height="{rows * cell}"></canvas>
<script id="payload" type="application/octet-stream">{data}</script>
<script type="module">
import * as THREE from './three.module.js';

const FR = JSON.parse(atob(document.getElementById('payload').textContent));
const CELL = {cell}, COLS = {cols}, ROWS = {rows};
const STYLE = "{style}";
const CAM = {json.dumps(cam)};

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias: true}});
renderer.setSize(COLS * CELL, ROWS * CELL, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setScissorTest(true);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f5f8);

// Checkered ground: the single strongest cue that this is a physics viewer,
// and it gives the eye a fixed reference so forward motion is visible.
const tex = (() => {{
  const s = 256, cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const g = cv.getContext('2d');
  g.fillStyle = '#e9edf2'; g.fillRect(0, 0, s, s);
  g.fillStyle = '#dde3ea';
  g.fillRect(0, 0, s / 2, s / 2); g.fillRect(s / 2, s / 2, s / 2, s / 2);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(60, 60);
  t.anisotropy = 4;
  return t;
}})();
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(240, 240),
  new THREE.MeshStandardMaterial({{map: tex, roughness: 0.98, metalness: 0.0}}));
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
ground.receiveShadow = true;
scene.add(ground);

const key = new THREE.DirectionalLight(0xffffff, 2.3);
key.position.set(4, 8, 5);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 40;
const S = 3.2;
key.shadow.camera.left = -S; key.shadow.camera.right = S;
key.shadow.camera.top = S; key.shadow.camera.bottom = -S;
key.shadow.bias = -0.0015;
scene.add(key);
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x586070, 0.55));

const body = new THREE.MeshStandardMaterial({{
  color: {colour}, roughness: 0.42, metalness: 0.04}});
const joint = new THREE.MeshStandardMaterial({{
  color: 0xf0f3f7, roughness: 0.35, metalness: 0.05}});

const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 120);
const group = new THREE.Group();
scene.add(group);

function clear(g) {{
  while (g.children.length) {{
    const c = g.children.pop();
    if (c.geometry) c.geometry.dispose();
  }}
}}

function capsule(p0, p1, r, mat) {{
  const a = new THREE.Vector3(p0[0], p0[1], p0[2]);
  const b = new THREE.Vector3(p1[0], p1[1], p1[2]);
  const d = new THREE.Vector3().subVectors(b, a);
  const len = d.length();
  let m;
  if (len < 1e-6) {{
    m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), mat);
  }} else {{
    m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 10, 18), mat);
    // CapsuleGeometry runs along +Y; rotate that onto the segment direction.
    m.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), d.clone().normalize());
  }}
  m.position.copy(a).add(b).multiplyScalar(0.5);
  m.castShadow = true;
  return m;
}}

function build(fr) {{
  clear(group);
  for (const c of fr.caps) {{
    group.add(capsule([c[0], c[1], c[2]], [c[3], c[4], c[5]], c[6], body));
    if (STYLE === "skeleton") {{
      // Ball joints at the bone ends read as an articulated figure rather
      // than a set of disconnected sticks.
      for (const p of [[c[0], c[1], c[2]], [c[3], c[4], c[5]]]) {{
        const j = new THREE.Mesh(new THREE.SphereGeometry(c[6] * 1.75, 16, 12), joint);
        j.position.set(p[0], p[1], p[2]); j.castShadow = true;
        group.add(j);
      }}
    }}
  }}
  for (const [pts, act] of fr.lines) {{
    const curve = new THREE.CatmullRomCurve3(
      pts.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const col = new THREE.Color().setHSL(0.62 - 0.62 * Math.min(1, Math.max(0, act)), 0.75, 0.47);
    const m = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.max(8, pts.length * 2), 0.006, 6, false),
      new THREE.MeshStandardMaterial({{color: col, roughness: 0.5}}));
    m.castShadow = true;
    group.add(m);
  }}
}}

// Camera follows the body along x only: a fixed height and distance make the
// gait's vertical motion readable instead of cancelling it out.
function place(fr) {{
  // Track the body in BOTH x and z. Tracking x alone and looking at z=0 means
  // a policy that drifts sideways walks away from the camera, so panels in the
  // same comparison end up at different apparent scales.
  let cx = 0, cz = 0, n = 0, hi = 0;
  for (const c of fr.caps) {{
    cx += c[0] + c[3]; cz += c[2] + c[5]; n += 2;
    hi = Math.max(hi, c[1], c[4]);
  }}
  cx = n ? cx / n : 0; cz = n ? cz / n : 0;
  const d = CAM.dist, h = CAM.height;
  camera.position.set(cx + d * Math.sin(CAM.yaw), h, cz + d * Math.cos(CAM.yaw));
  camera.lookAt(cx, CAM.look, cz);
  key.position.set(cx + 4, 8, cz + 5);
  key.target.position.set(cx, 0, cz);
  key.target.updateMatrixWorld();
}}

for (let i = 0; i < FR.length; i++) {{
  const col = i % COLS, row = Math.floor(i / COLS);
  const x = col * CELL, y = (ROWS - 1 - row) * CELL;   // WebGL origin is bottom-left
  build(FR[i]); place(FR[i]);
  renderer.setViewport(x, y, CELL, CELL);
  renderer.setScissor(x, y, CELL, CELL);
  renderer.render(scene, camera);
}}
document.title = "done";
</script></body></html>
"""


def render_sheets(frames, out_dir, colour, style, cell=360, per_sheet=40,
                  cam=None, timeout=300):
    """Render ``frames`` to sprite-sheet PNGs; returns the list of paths."""
    cam = cam or {"dist": 3.4, "height": 1.35, "yaw": 0.62, "look": 0.75}
    os.makedirs(out_dir, exist_ok=True)
    chrome = _chromium()
    sheets = []
    for start in range(0, len(frames), per_sheet):
        chunk = [_frame_payload(f, style) for f in frames[start:start + per_sheet]]
        cols = int(math.ceil(math.sqrt(len(chunk))))
        rows = int(math.ceil(len(chunk) / cols))
        html = _page(chunk, cell, cols, rows, colour, style, cam)
        page = os.path.join(out_dir, f"sheet_{start:05d}.html")
        png = os.path.join(out_dir, f"sheet_{start:05d}.png")
        with open(page, "w") as f:
            f.write(html)
        # three.module.js must sit beside the page: file:// module imports are
        # resolved relative to the document.
        link = os.path.join(out_dir, "three.module.js")
        if not os.path.exists(link):
            shutil.copy(THREE, link)
        subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
             "--enable-unsafe-swiftshader", "--allow-file-access-from-files",
             "--hide-scrollbars", "--virtual-time-budget=120000",
             f"--screenshot={png}", f"--window-size={cols * cell},{rows * cell}",
             f"file://{page}"],
            check=False, capture_output=True, timeout=timeout)
        if not os.path.exists(png):
            raise RuntimeError(f"chromium produced no screenshot for {page}")
        sheets.append((png, cols, rows, len(chunk)))
    return sheets


def frames_from_sheets(sheets, cell):
    """Slice sprite sheets back into individual frames, in order."""
    from PIL import Image

    out = []
    for png, cols, rows, n in sheets:
        im = Image.open(png).convert("RGB")
        for i in range(n):
            c, r = i % cols, i // cols
            out.append(im.crop((c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)))
    return out


def render_video(frame_sets, labels, out_path, style="capsule", cell=360,
                 fps=30, cam=None):
    """Side-by-side shaded 3D video, one panel per frame set."""
    from PIL import Image, ImageDraw

    import imageio.v2 as imageio

    n = min(len(f) for f in frame_sets)
    work = tempfile.mkdtemp(prefix="r3d_", dir=os.path.dirname(os.path.abspath(out_path)))
    try:
        panels = []
        for k, fs in enumerate(frame_sets):
            colour = list(PALETTE.values())[k % len(PALETTE)]
            sheets = render_sheets(fs[:n], os.path.join(work, f"p{k}"), colour, style,
                                   cell=cell, cam=cam)
            panels.append(frames_from_sheets(sheets, cell))
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
        try:
            for t in range(n):
                row = [p[t] for p in panels]
                w = sum(im.width for im in row)
                canvas = Image.new("RGB", (w, row[0].height + 26), "white")
                x = 0
                for im in row:
                    canvas.paste(im, (x, 26))
                    x += im.width
                d = ImageDraw.Draw(canvas)
                x = 0
                for im, lab in zip(row, labels):
                    d.text((x + 10, 7), lab, fill=(40, 44, 52))
                    x += im.width
                writer.append_data(np.asarray(canvas))
        finally:
            writer.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path


# Per-environment camera. Distance and look-height are set from each model's
# own size: Ant is wide and low, the Humanoid is tall, and a single camera
# either clips the Ant's legs or leaves the Hopper tiny in frame.
CAMERAS = {
    "hopper":          {"dist": 3.0, "height": 1.20, "yaw": 0.50, "look": 0.70},
    "ant":             {"dist": 4.2, "height": 1.35, "yaw": 0.70, "look": 0.45},
    "humanoid":        {"dist": 4.4, "height": 1.30, "yaw": 0.55, "look": 0.75},
    "snu_humanoid":    {"dist": 3.8, "height": 1.05, "yaw": 0.62, "look": 0.55},
    "cartpoleswingup": {"dist": 3.2, "height": 1.00, "yaw": 0.35, "look": 0.55},
}

# SNU Humanoid's collision geometry is 11 capsules dominated by two torso
# spheres, so in 3D it renders as a featureless blob (verified visually); its
# link tree is the only legible option. The other models have real capsule
# geometry and read correctly as solids.
DEFAULT_STYLE = {
    "snu_humanoid": "skeleton",
    "hopper": "capsule",
    "ant": "capsule",
    "humanoid": "capsule",
    "cartpoleswingup": "capsule",
}


def main():
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True, choices=sorted(CAMERAS))
    ap.add_argument("--demos", help="expert demo file for the left panel")
    ap.add_argument("--logdir", action="append", default=[],
                    help="run directory for a learner panel; repeatable")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--steps", type=int, default=None,
                    help="budget to pick the learner checkpoint at")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cell", type=int, default=420)
    ap.add_argument("--style", default=None, choices=["capsule", "skeleton", "muscles"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "_rr", os.path.join(HERE, "render_rollout.py"))
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)

    style = args.style or DEFAULT_STYLE.get(args.env, "capsule")
    env = rr.make_env(args.env, 1, args.device, 1000)
    sets, labels = [], []
    if args.demos:
        fr, _ = rr.frames_from_demo(env, args.env, args.demos, args.frames, 0,
                                    args.device, style=style)
        sets.append(fr)
        labels.append("expert demonstration")
    for i, ld in enumerate(args.logdir):
        fr = rr.frames_from_policy(env, ld, args.frames, args.device, style=style)
        sets.append(fr[0] if isinstance(fr, tuple) else fr)
        labels.append(args.label[i] if i < len(args.label) else os.path.basename(ld))
    if not sets:
        ap.error("nothing to render: pass --demos and/or --logdir")
    render_video(sets, labels, args.out, style=style, cell=args.cell,
                 fps=args.fps, cam=CAMERAS[args.env])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ bones ---
# SNU Humanoid only: real bone meshes placed rigidly per link. See
# plotter/snu_bones.py for why the rest pose has to be divided out first.

def _page_bones(meshes, frames, cell, cols, rows, cam, muscles=None):
    import base64 as _b64

    geo = {}
    for name, b in meshes.items():
        geo[name] = {
            "v": _b64.b64encode(b["verts"].astype(np.float32).tobytes()).decode(),
            "f": _b64.b64encode(b["faces"].astype(np.uint32).tobytes()).decode(),
        }
    payload = _b64.b64encode(json.dumps(
        {"geo": geo, "frames": frames, "muscles": muscles or []}).encode()).decode()
    return f"""<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:#fff">
<canvas id="c" width="{cols * cell}" height="{rows * cell}"></canvas>
<script id="payload" type="application/octet-stream">{payload}</script>
<script type="module">
import * as THREE from './three.module.js';
const D = JSON.parse(atob(document.getElementById('payload').textContent));
const CELL = {cell}, COLS = {cols}, ROWS = {rows}, CAM = {json.dumps(cam)};

function b64f32(s) {{
  const bin = atob(s), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
function b64u32(s) {{
  const bin = atob(s), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Uint32Array(buf);
}}

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias: true}});
renderer.setSize(COLS * CELL, ROWS * CELL, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setScissorTest(true);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f5f8);
const tex = (() => {{
  const s = 256, cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const g = cv.getContext('2d');
  g.fillStyle = '#e9edf2'; g.fillRect(0, 0, s, s);
  g.fillStyle = '#dde3ea';
  g.fillRect(0, 0, s/2, s/2); g.fillRect(s/2, s/2, s/2, s/2);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(60, 60); t.anisotropy = 4;
  return t;
}})();
const ground = new THREE.Mesh(new THREE.PlaneGeometry(240, 240),
  new THREE.MeshStandardMaterial({{map: tex, roughness: 0.98}}));
ground.rotation.x = -Math.PI/2; ground.receiveShadow = true; scene.add(ground);

const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(4, 8, 5); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 40;
const S = 2.4;
key.shadow.camera.left = -S; key.shadow.camera.right = S;
key.shadow.camera.top = S; key.shadow.camera.bottom = -S;
key.shadow.bias = -0.0012;
scene.add(key, new THREE.AmbientLight(0xffffff, 0.5),
          new THREE.HemisphereLight(0xcfe0ff, 0x586070, 0.5));

// Bone-like material: warm off-white, matte, slight subsurface feel from the
// hemisphere light. This is what reads as a skeleton rather than a toy.
const boneMat = new THREE.MeshStandardMaterial({{
  color: 0xeae3d2, roughness: 0.68, metalness: 0.0, flatShading: false}});

const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 120);
const objs = {{}};
for (const [name, g] of Object.entries(D.geo)) {{
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.BufferAttribute(b64f32(g.v), 3));
  bg.setIndex(new THREE.BufferAttribute(b64u32(g.f), 1));
  bg.computeVertexNormals();
  const m = new THREE.Mesh(bg, boneMat);
  m.castShadow = true; m.receiveShadow = false;
  scene.add(m); objs[name] = m;
}}

// Muscles are rebuilt per frame: unlike bones they are not rigid, each
// waypoint rides a different link. Colour runs blue (relaxed) to red (fully
// activated); activation is the policy's own action divided by the muscle's
// strength, which is exactly what the simulator applies.
const muscleGroup = new THREE.Group();
scene.add(muscleGroup);
// Two-stop blue -> red. An HSL sweep would pass through cyan, green and
// yellow, which reads as a rainbow rather than "relaxed vs contracted".
const REST = new THREE.Color(0x2f6fb8), FIRED = new THREE.Color(0xd4322f);
function colourFor(a) {{
  const t = Math.min(1, Math.max(0, a));
  return REST.clone().lerp(FIRED, t);
}}
function buildMuscles(frameMuscles) {{
  while (muscleGroup.children.length) {{
    const c = muscleGroup.children.pop();
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }}
  if (!frameMuscles) return;
  for (const [pts, act] of frameMuscles) {{
    if (pts.length < 2) continue;
    const curve = new THREE.CatmullRomCurve3(
      pts.map(q => new THREE.Vector3(q[0], q[1], q[2])));
    const m = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.max(6, pts.length * 3), 0.0055, 5, false),
      new THREE.MeshStandardMaterial({{color: colourFor(act), roughness: 0.45,
                                      emissive: colourFor(act),
                                      emissiveIntensity: 0.10 + 0.25 * Math.min(1, Math.max(0, act))}}));
    m.castShadow = false;
    muscleGroup.add(m);
  }}
}}

for (let i = 0; i < D.frames.length; i++) {{
  const fr = D.frames[i];
  buildMuscles(D.muscles[i]);
  // Anchor on the pelvis, not the mean of all bone origins: the mean drifts
  // toward whichever limb is extended, so the figure slides around the frame.
  // Track z too, for the same reason as the capsule view.
  let cx = 0, cz = 0, n = 0, px = null, pz = null;
  for (const [name, p, q] of fr) {{
    const o = objs[name];
    if (!o) continue;
    o.position.set(p[0], p[1], p[2]);
    o.quaternion.set(q[0], q[1], q[2], q[3]);
    cx += p[0]; cz += p[2]; n++;
    if (name === "Pelvis") {{ px = p[0]; pz = p[2]; }}
  }}
  cx = px !== null ? px : (n ? cx / n : 0);
  cz = pz !== null ? pz : (n ? cz / n : 0);
  camera.position.set(cx + CAM.dist * Math.sin(CAM.yaw), CAM.height, cz + CAM.dist * Math.cos(CAM.yaw));
  camera.lookAt(cx, CAM.look, cz);
  key.position.set(cx + 4, 8, cz + 5);
  key.target.position.set(cx, 0, cz); key.target.updateMatrixWorld();
  const col = i % COLS, row = Math.floor(i / COLS);
  const x = col * CELL, y = (ROWS - 1 - row) * CELL;
  renderer.setViewport(x, y, CELL, CELL);
  renderer.setScissor(x, y, CELL, CELL);
  renderer.render(scene, camera);
}}
document.title = "done";
</script></body></html>
"""


def render_bones_video(meshes, frame_sets, labels, out_path, cell=420, fps=30,
                       cam=None, per_sheet=40, muscle_sets=None):
    """Side-by-side bone-mesh video. ``frame_sets`` hold per-frame placements."""
    from PIL import Image, ImageDraw

    import imageio.v2 as imageio

    cam = cam or CAMERAS["snu_humanoid"]
    n = min(len(f) for f in frame_sets)
    work = tempfile.mkdtemp(prefix="r3db_", dir=os.path.dirname(os.path.abspath(out_path)))
    chrome = _chromium()
    try:
        panels = []
        for k, fs in enumerate(frame_sets):
            out_dir = os.path.join(work, f"p{k}")
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(THREE, os.path.join(out_dir, "three.module.js"))
            sheets = []
            for start in range(0, n, per_sheet):
                chunk = fs[start:start + per_sheet]
                cols = int(math.ceil(math.sqrt(len(chunk))))
                rows = int(math.ceil(len(chunk) / cols))
                page = os.path.join(out_dir, f"s{start:05d}.html")
                png = os.path.join(out_dir, f"s{start:05d}.png")
                mus = None
                if muscle_sets is not None and muscle_sets[k] is not None:
                    mus = muscle_sets[k][start:start + per_sheet]
                with open(page, "w") as f:
                    f.write(_page_bones(meshes, chunk, cell, cols, rows, cam, muscles=mus))
                subprocess.run(
                    [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                     "--enable-unsafe-swiftshader", "--allow-file-access-from-files",
                     "--hide-scrollbars", "--virtual-time-budget=120000",
                     f"--screenshot={png}",
                     f"--window-size={cols * cell},{rows * cell}", f"file://{page}"],
                    check=False, capture_output=True, timeout=600)
                if not os.path.exists(png):
                    raise RuntimeError(f"chromium produced no screenshot for {page}")
                sheets.append((png, cols, rows, len(chunk)))
            panels.append(frames_from_sheets(sheets, cell))
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
        try:
            for t in range(n):
                row = [p[t] for p in panels]
                w = sum(im.width for im in row)
                canvas = Image.new("RGB", (w, row[0].height + 26), "white")
                x = 0
                for im in row:
                    canvas.paste(im, (x, 26)); x += im.width
                d = ImageDraw.Draw(canvas)
                x = 0
                for im, lab in zip(row, labels):
                    d.text((x + 10, 7), lab, fill=(40, 44, 52)); x += im.width
                writer.append_data(np.asarray(canvas))
        finally:
            writer.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path
