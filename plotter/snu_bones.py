"""Anatomical bone meshes for the SNU Humanoid.

The dflex SNU model ships real bone geometry -- Pelvis, Femur, Tibia, Talus and
the two foot segments per side -- under ``dflex/envs/assets/snu/OBJ``. The env
only uses the ``.usd`` copies, and ``render_rollout.py`` deliberately avoided
meshes entirely, so every clip so far has shown either the collision capsules
(two torso spheres and some stubs, which render as a blob) or a stick figure.
Neither looks like MyoSuite, Hyfydy or the SHAC project page, all of which show
the actual skeleton.

The ``.obj`` files are authored in **centimetres in one shared global rest
pose**, not in per-link frames: ``Pelvis.obj`` has its centroid at y=97.1 while
its link sits at y=1.027 m. So a mesh cannot be attached to a link by its
transform alone -- the rest pose has to be divided out first:

    local = X_bind(link)^-1 . (vertex * 0.01)
    world = X_sc(link, t) . local

``X_bind`` is the model's own forward kinematics at the static initial state.
``body_X_sc`` is zero-filled until the first integration step, so the bind pose
is read after one zero-action step from a deterministic reset.
"""

import os

import numpy as np

CM_TO_M = 0.01


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _qrot(q, v):
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, w * v + np.cross(u, v))


def _qrot_many(q, V):
    u, w = q[:3], q[3]
    return V + 2.0 * np.cross(np.broadcast_to(u, V.shape),
                              w * V + np.cross(np.broadcast_to(u, V.shape), V))


def mesh_map(env):
    """{mesh name: link index} for env 0, skipping unmapped meshes."""
    out = {}
    for sk in env.skeletons:
        for name, link in sk.mesh_map.items():
            if link != -1:
                out[name] = int(link)
    return out


def bind_transforms(env):
    """{link: (pos, quat)} in the skeleton's own bind pose, from the XML.

    NOT the simulated pose. ``Skeleton`` records ``xform_map[node] =
    joint_X_s``, the joint's world transform as parsed from ``human.xml``, and
    that is the frame the OBJ files are authored in -- ``load_utils.py`` uses
    exactly this to localise muscle waypoints:

        joint_X_s = self.xform_map[way_bone]
        way_loc = transform_point(transform_inverse(joint_X_s), way_loc)

    Using simulated forward kinematics instead looks right at the start pose
    and then tears the skeleton apart as joints rotate away from it, because
    each bone carries its own constant error in a frame that is rotating.
    """
    out = {}
    for sk in env.skeletons:
        link_to_name = {int(v): k for k, v in sk.node_map.items()}
        for link, name in link_to_name.items():
            X = sk.xform_map.get(name)
            if X is None:
                continue
            # df.transform() returns a (pos, quat) pair, not a flat 7-vector.
            if isinstance(X, (tuple, list)) and len(X) == 2:
                pos = np.asarray(X[0], dtype=np.float64).reshape(-1)[:3]
                quat = np.asarray(X[1], dtype=np.float64).reshape(-1)[:4]
            else:
                arr = np.asarray(X, dtype=np.float64).reshape(-1)
                pos, quat = arr[:3], arr[3:7]
            out[link] = (pos, quat)
    return out


def load_bones(env, device="cpu", max_faces=4000):
    """-> {name: {"link", "verts" (link-local, metres), "faces"}}.

    ``max_faces`` decimates the heavier meshes; Pelvis and Talus carry most of
    the 20k triangles and the silhouette survives decimation fine.
    """
    import trimesh

    mm = mesh_map(env)
    bind = bind_transforms(env)
    folder = os.path.join(env.asset_folder, "OBJ")
    bones = {}
    for name, link in sorted(mm.items()):
        path = os.path.join(folder, name + ".obj")
        if not os.path.exists(path):
            continue
        m = trimesh.load(path, process=False, force="mesh")
        if max_faces and len(m.faces) > max_faces:
            try:
                m = m.simplify_quadric_decimation(max_faces)
            except Exception:
                pass  # decimation is optional; the raw mesh still renders
        if link not in bind:
            continue
        v = np.asarray(m.vertices, dtype=np.float64) * CM_TO_M
        bp, bq = bind[link]
        local = _qrot_many(_qconj(bq), v - bp)
        bones[name] = {
            "link": link,
            "verts": local.astype(np.float32),
            "faces": np.asarray(m.faces, dtype=np.int32),
        }
    return bones


def bone_pose(bones, body_X_sc):
    """Per-frame rigid placement: [(name, pos[3], quat[4]), ...]."""
    out = []
    for name, b in bones.items():
        X = body_X_sc[b["link"]]
        out.append((name, X[:3].astype(float).tolist(), X[3:].astype(float).tolist()))
    return out
