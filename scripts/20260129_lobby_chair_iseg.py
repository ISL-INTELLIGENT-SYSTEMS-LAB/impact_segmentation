import shutil
import sys
import time
from pathlib import Path
from typing import List, Set

import omni.kit.app
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom
import carb


# =========================
# CONFIG
# =========================

OUTPUT_ROOT = r"A:\Project\LunarProject\iseg\output_lobby_chairs"
SCENE_USD = r"A:\Project\LunarProject\usd\Collected_World_Lobby\World_Lobby.usd"

CAMERA_PATHS = [
    "/World/Cameras/CamLobbyWide",
    "/World/Cameras/CamLobbySeating",
]

RESOLUTION = (1024, 768)
TOTAL_FRAMES = 1

GLOBAL_SEED = 12345
CLEAN_OUTPUT_EACH_RUN = True

WARMUP_TICKS = 120
TICKS_PER_FRAME = 8
POST_TICKS = 120

KEEP_VISIBLE_PREFIXES = [
    "/World/Cameras",
    "/World/DomeLight",
    "/World/Looks",
]

# Target chair family tokens based on your prim path
CHAIR_PATH_TOKENS = ["CapriceBackless", "Caprice_A", "Caprice_C", "Caprice"]
FURNITURE_CONTAINER = "/World/assembly_Lobby/assets_lobby_furniture"

SEMANTIC_CLASS_NAME = "chair"
RGB_CAMERA_EXPOSURE_OVERRIDE = -2.0


# =========================
# KIT HELPERS
# =========================

def kit_update(n: int = 1, sleep_s: float = 0.0) -> None:
    app = omni.kit.app.get_app()
    for _ in range(max(1, n)):
        app.update()
        if sleep_s:
            time.sleep(sleep_s)

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()

def open_stage(usd_path: str) -> Usd.Stage:
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    kit_update(60)
    stage = get_stage()
    if not stage:
        raise RuntimeError(f"Failed to open stage: {usd_path}")
    return stage

def load_all_payloads(stage: Usd.Stage) -> None:
    try:
        stage.Load(Usd.Stage.LoadAll)
    except Exception:
        try:
            stage.Load()
        except Exception:
            pass
    kit_update(240)


# =========================
# POST SETTINGS
# =========================

def set_rgb_post_defaults():
    carb.settings.get_settings().set("/rtx/post/autoExposure/enabled", True)

def set_mask_post_defaults():
    s = carb.settings.get_settings()
    s.set("/rtx/post/autoExposure/enabled", False)
    s.set("/rtx/post/bloom/enabled", False)
    s.set("/rtx/post/vignette/enabled", False)
    s.set("/rtx/post/filmGrain/enabled", False)
    s.set("/rtx/post/tonemap/enabled", True)


# =========================
# NON-DESTRUCTIVE SESSION EDITS
# =========================

class SessionEdit:
    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.prev_target = None

    def __enter__(self):
        self.prev_target = self.stage.GetEditTarget()
        self.stage.SetEditTarget(Usd.EditTarget(self.stage.GetSessionLayer()))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.prev_target is not None:
            self.stage.SetEditTarget(self.prev_target)

def override_camera_exposure(stage: Usd.Stage, cam_path: str, exposure_value: float) -> None:
    prim = stage.GetPrimAtPath(cam_path)
    if not prim or not prim.IsValid():
        return
    attr = prim.GetAttribute("exposure")
    if not attr:
        attr = prim.CreateAttribute("exposure", UsdGeom.Tokens.float, custom=True)
    attr.Set(float(exposure_value))


# =========================
# CHAIR ROOT DISCOVERY (PATH-BASED)
# =========================

def _contains_any_token(path: str, tokens: List[str]) -> bool:
    p = path.lower()
    return any(t.lower() in p for t in tokens)

def _is_prefix(prefix: str, path: str) -> bool:
    if prefix == "/":
        return True
    if not path.startswith(prefix):
        return False
    if len(path) == len(prefix):
        return True
    return path[len(prefix)] == "/"

def asset_root_under_container(full_path: str, container_path: str) -> str:
    if not _is_prefix(container_path, full_path):
        return full_path
    if full_path == container_path:
        return full_path
    rest = full_path[len(container_path):].lstrip("/")
    first = rest.split("/")[0]
    return f"{container_path}/{first}"

def find_chair_asset_roots_by_path(stage: Usd.Stage) -> List[str]:
    container = stage.GetPrimAtPath(FURNITURE_CONTAINER)
    if not container or not container.IsValid():
        raise RuntimeError(f"Furniture container prim not found: {FURNITURE_CONTAINER}")

    roots: Set[str] = set()
    for prim in Usd.PrimRange(container):
        p = prim.GetPath().pathString
        if _contains_any_token(p, CHAIR_PATH_TOKENS):
            roots.add(asset_root_under_container(p, FURNITURE_CONTAINER))
    return sorted(roots)


# =========================
# ISOLATION (FIXED: KEEP ANCESTORS)
# =========================

def expand_with_ancestors(paths: List[str]) -> List[str]:
    expanded = set()
    for p in paths:
        parts = p.split("/")
        cur = ""
        for part in parts:
            if not part:
                continue
            cur += "/" + part
            expanded.add(cur)
    expanded.add("/World")
    expanded.add("/")
    return sorted(expanded)

def hide_everything_except(stage: Usd.Stage, keep_paths: List[str], keep_prefixes: List[str]) -> int:
    keep_all = set(keep_prefixes) | set(expand_with_ancestors(keep_paths))
    hidden = 0

    def is_kept(path: str) -> bool:
        for k in keep_all:
            if path == k or path.startswith(k + "/"):
                return True
        return False

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = prim.GetPath().pathString
        if is_kept(p):
            continue
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden += 1
    return hidden


# =========================
# SEMANTICS TAGGING
# =========================

def tag_meshes_under_roots(stage: Usd.Stage, root_paths: List[str], label: str) -> int:
    tagged = 0
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                rep.modify.semantics([("class", label)], input_prims=[prim.GetPath().pathString], mode="add")
                tagged += 1
    return tagged


# =========================
# TWO-PASS RENDER
# =========================

def run_two_pass_for_camera(stage: Usd.Stage, cam_path: str, rgb_dir: str, mask_dir: str, chair_roots: List[str]):
    rp = rep.create.render_product(cam_path, RESOLUTION)

    # PASS A: RGB full scene
    set_rgb_post_defaults()
    with SessionEdit(stage):
        if RGB_CAMERA_EXPOSURE_OVERRIDE is not None:
            override_camera_exposure(stage, cam_path, RGB_CAMERA_EXPOSURE_OVERRIDE)

        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=rgb_dir, rgb=True, instance_segmentation=False, semantic_segmentation=False)
        w.attach(rp)

        rep.orchestrator.run(num_frames=TOTAL_FRAMES)
        kit_update(WARMUP_TICKS)
        kit_update(max(1, TOTAL_FRAMES) * TICKS_PER_FRAME + POST_TICKS)
        rep.orchestrator.stop()
        kit_update(20)
        w.detach()

    # PASS B: masks isolated
    set_mask_post_defaults()
    with SessionEdit(stage):
        hidden = hide_everything_except(stage, chair_roots, KEEP_VISIBLE_PREFIXES)
        tagged = tag_meshes_under_roots(stage, chair_roots, SEMANTIC_CLASS_NAME)
        print(f"    [MASK PASS] chair_roots={len(chair_roots)} hidden={hidden} tagged_meshes={tagged}")

        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=mask_dir, rgb=False, instance_segmentation=True, semantic_segmentation=True)
        w.attach(rp)

        rep.orchestrator.run(num_frames=TOTAL_FRAMES)
        kit_update(WARMUP_TICKS)
        kit_update(max(1, TOTAL_FRAMES) * TICKS_PER_FRAME + POST_TICKS)
        rep.orchestrator.stop()
        kit_update(20)
        w.detach()


# =========================
# MAIN
# =========================

def main():
    print(">>> Two-pass: RGB full + isolated chair masks (fixed visibility inheritance)")

    if CLEAN_OUTPUT_EACH_RUN and Path(OUTPUT_ROOT).exists():
        shutil.rmtree(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT)

    rep.set_global_seed(GLOBAL_SEED)

    stage = open_stage(SCENE_USD)
    load_all_payloads(stage)

    chair_roots = find_chair_asset_roots_by_path(stage)
    if not chair_roots:
        raise RuntimeError("No chair roots found. Adjust CHAIR_PATH_TOKENS or FURNITURE_CONTAINER.")

    print(f"[INFO] Found {len(chair_roots)} chair roots:")
    for p in chair_roots:
        print(f"   - {p}")

    out_root = ensure_dir(str(Path(OUTPUT_ROOT) / "train"))
    for cam_path in CAMERA_PATHS:
        cam_name = cam_path.split("/")[-1]
        cam_root = Path(out_root) / cam_name
        rgb_dir = ensure_dir(str(cam_root / "rgb"))
        mask_dir = ensure_dir(str(cam_root / "masks"))

        print(f"[INFO] Camera: {cam_path}")
        run_two_pass_for_camera(stage, cam_path, rgb_dir, mask_dir, chair_roots)

    print("[OK] Done.")
    omni.kit.app.get_app().post_quit()


if __name__ == "__main__":
    main()
    try:
        kit_update(30)
    except Exception:
        pass
    sys.exit(0)
