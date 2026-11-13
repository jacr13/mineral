import importlib
import re

from omegaconf import OmegaConf


def _patch_jumper_table(module):
    """Build Jumper's table mesh as a plain list (usd-core >= 26 compatibility).

    Rewarped 1.3.3's Jumper.create_table assigns xform.Transform(p) (a Vec3d) into a Vec3fArray, which usd-core >= 26
    rejects. Same body otherwise.
    """
    cls = module.Jumper
    if getattr(cls, '_mineral_usd_patch', False):
        return
    import os

    import warp as wp
    from pxr import Gf, Usd, UsdGeom

    def create_table(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(self.asset_dir, "dflex/usd", "table.usda"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/mesh"))
        xform = geom.ComputeLocalToWorldTransform(0.0)
        points = [wp.vec3(*xform.Transform(Gf.Vec3d(*p))) for p in geom.GetPointsAttr().Get()]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.2, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    cls.create_table = create_table
    cls._mineral_usd_patch = True


def make_envs(config):
    env = config.task.env
    env_kwargs = OmegaConf.to_container(env, resolve=True)
    env_name, num_envs = env_kwargs.pop("env_name"), env_kwargs.pop("numEnvs")
    env_suite = env_kwargs.pop("env_suite")

    def camelcase_to_snakecase(str):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

    ENV = importlib.import_module(f"rewarped.envs.{env_suite}.{camelcase_to_snakecase(env_name)}")
    if env_name == "Jumper":
        _patch_jumper_table(ENV)
    env_fn = getattr(ENV, env_name)
    env = env_fn(
        num_envs=num_envs,
        device=config.sim_device,
        seed=config.seed,
        **env_kwargs,
    )

    return env
