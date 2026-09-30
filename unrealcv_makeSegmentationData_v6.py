from unrealcv import Client
import os
import math
from datetime import datetime
import json
from shutil import copy2
import random
import sys
import traceback
import io
import imageio.v3 as iio
import numpy as np



# ============================================================
# USER SETTINGS
# ============================================================

COLORMAP = "colormap_output.json"
UNREALCV_COLORMAP_PATH = r"C:\Users\Albert\Documents\Unreal Projects\military_base_3maps\Saved\UnrealCV\colormap_output.json"
ACTORLIST = "actors.json"
HOST = "localhost"
PORT = 9000
EXPERIMENT_LOG_FILENAME = "experiment.log"
EXPERIMENT_CONFIG_NAME = "experiment_config.json"
MIN_TARGET_PIXELS = 500

# ------------------------------------------------------------
# Small / far-background segmentation filtering
# ------------------------------------------------------------
# After the existing depth-based occlusion filter runs, this second filter
# removes an actor's remaining segmentation only when BOTH conditions are true:
#   1) the actor has fewer than SMALL_MASK_PIXEL_THRESHOLD visible pixels, and
#   2) the actor is clearly farther from the camera than TARGET_OBJECT.
#
# This protects nearby/foreground objects that may have only a small visible
# sliver because they are heavily occluded.
ENABLE_SMALL_FAR_MASK_FILTER = True
SMALL_MASK_PIXEL_THRESHOLD = 100

# An actor must satisfy BOTH far-distance tests to be considered background.
# Unreal Engine units are centimeters by default.
FAR_BACKGROUND_DISTANCE_MARGIN_UU = 1500.0
FAR_BACKGROUND_DISTANCE_RATIO = 1.15

# Never remove TARGET_OBJECT with the small/far filter.
ALWAYS_KEEP_TARGET_OBJECT = True
# Both depth captures use Unreal world units and float16 render-target readback.
# Allow a small absolute difference or approximately one float16 relative step.
DEPTH_ABSOLUTE_TOLERANCE_UU = 2.0
DEPTH_RELATIVE_TOLERANCE = 0.001
OCCLUDED_COLOR = (0, 0, 0)  # Unlabelled pixels; must not be an actor color.
TARGET_OBJECT = "Car_C_26"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FOV = 90
DISTANCE_CATEGORY = "close"   # "normal", "far", or "very_far"

DISTANCE_CONFIG = {
    "close": {
            "radii": [1500],
            "elevations": [10, 30, 40, 50],
        },
    "normal": {
        "radii": [3500, 5000],
        "elevations": [10, 30, 50],
    },

    "far": {
        "radii": [7500], 
        "elevations": [10, 30, 50, 65],
    },

    "very_far": {
        "radii": [12000],
        "elevations": [10, 25, 40, 50],
    },
}

config = DISTANCE_CONFIG[DISTANCE_CATEGORY]

RADII = config["radii"]
ELEVATIONS = config["elevations"]

STEPS = 8

OUTPUT_DIR = "capture"

class Tee:
    """
    Write output to both the terminal and a logfile.
    """

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

        # Make sure the log is continuously updated
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return self.terminal.isatty()

def get_camera_info(client, camera_id):
    """
    Query the current UnrealCV camera configuration.
    """

    commands = {
        "location": f"vget /camera/{camera_id}/location",
        "rotation": f"vget /camera/{camera_id}/rotation",
        "pose":     f"vget /camera/{camera_id}/pose",
        "fov":      f"vget /camera/{camera_id}/fov",
        "size":     f"vget /camera/{camera_id}/size",
    }

    info = {}

    print("\nCamera configuration")
    print("--------------------")
    print(f"Camera ID: /camera/{camera_id}")

    for name, command in commands.items():
        try:
            response = client.request(command)
            info[name] = response
            print(f"{name:10s}: {response}")
        except Exception as e:
            info[name] = None
            print(f"{name:10s}: unavailable ({e})")

    return info

def decode_depth(response, command, expected_shape):
    if not isinstance(response, bytes):
        raise RuntimeError(f"Expected NPY bytes from {command}, got {response!r}")
    try:
        depth = np.load(io.BytesIO(response), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Invalid NPY response from {command}") from exc
    if not isinstance(depth, np.ndarray) or depth.shape != expected_shape:
        raise RuntimeError(
            f"Depth shape mismatch for {command}: "
            f"{getattr(depth, 'shape', None)} versus mask {expected_shape}"
        )
    if depth.dtype.kind != "f":
        raise RuntimeError(f"Expected floating-point depth from {command}, got {depth.dtype}")
    return depth


def filter_mask_occlusion(mask_bytes, annotation_response, scene_response,
                            target_obj, color_map):
    """Remove occluded pixels from every label; count target visibility separately."""
    if not isinstance(mask_bytes, bytes):
        raise RuntimeError(f"Expected object-mask PNG bytes, got {mask_bytes!r}")
    mask = iio.imread(io.BytesIO(mask_bytes), extension=".png")
    if mask.ndim != 3 or mask.shape[2] not in (3, 4):
        raise RuntimeError(f"Expected RGB/RGBA object mask, got {mask.shape}")
    annotation_depth = decode_depth(annotation_response, "/depth npy", mask.shape[:2])
    scene_depth = decode_depth(scene_response, "/scene_depth npy", mask.shape[:2])
    target = np.all(mask[:, :, :3] == np.asarray(color_map[target_obj], dtype=np.uint8), axis=2)

    # Annotation depth is the nearest annotation surface at every labelled pixel.
    labelled = np.any(mask[:, :, :3] != np.asarray(OCCLUDED_COLOR), axis=2)
    valid = (np.isfinite(annotation_depth) & (annotation_depth > 0)
             & np.isfinite(scene_depth) & (scene_depth > 0))
    visible = np.zeros(target.shape, dtype=bool)
    comparable = labelled & valid
    tolerance = np.maximum(DEPTH_ABSOLUTE_TOLERANCE_UU,
                           DEPTH_RELATIVE_TOLERANCE * annotation_depth[comparable])
    visible[comparable] = (annotation_depth[comparable]
                           <= scene_depth[comparable] + tolerance)
    # Invalid depths cannot establish visibility: discard and report them.
    removed = labelled & ~visible
    filtered = mask.copy()
    filtered[:, :, :3][removed] = OCCLUDED_COLOR
    stats = {
        "raw": int(np.count_nonzero(target)),
        "visible": int(np.count_nonzero(target & visible)),
        "occluded": int(np.count_nonzero(target & comparable & ~visible)),
        "invalid_depth": int(np.count_nonzero(target & ~valid)),
        "all_labels_raw": int(np.count_nonzero(labelled)),
        "all_labels_occluded": int(np.count_nonzero(comparable & ~visible)),
        "all_labels_invalid_depth": int(np.count_nonzero(labelled & ~valid)),
    }
    return iio.imwrite("<bytes>", filtered, extension=".png"), stats


def capture_occlusion_frame(
    client,
    camera_id,
    target_obj,
    color_map,
    camera_pos,
):
    """
    Queue all four captures for execution in one Unreal game tick.

    Processing order:
      1) depth-based occlusion filtering
      2) small + far-background actor filtering
    """
    response = client.request("vbatch 4")
    if response != "ok":
        raise RuntimeError(f"Could not start capture batch: {response!r}")

    commands = [
        f"vget /camera/{camera_id}/{mode} {fmt}"
        for mode, fmt in (
            ("lit", "png"),
            ("object_mask", "png"),
            ("depth", "npy"),
            ("scene_depth", "npy"),
        )
    ]

    responses = client.request(commands)
    if not isinstance(responses, (list, tuple)) or len(responses) != 4:
        raise RuntimeError("Expected four responses from the capture batch")

    rgb_bytes, mask_bytes, annotation_depth, scene_depth = responses

    if not isinstance(rgb_bytes, bytes):
        raise RuntimeError(f"Expected RGB PNG bytes, got {rgb_bytes!r}")

    # Stage 1: remove mask pixels hidden by actual scene geometry.
    filtered_mask, stats = filter_mask_occlusion(
        mask_bytes,
        annotation_depth,
        scene_depth,
        target_obj,
        color_map,
    )

    # Stage 2: remove only actors that are BOTH tiny and clearly farther
    # from the camera than the selected target object.
    filtered_mask, small_far_report = filter_small_far_masks(
        client=client,
        mask_bytes=filtered_mask,
        camera_pos=camera_pos,
        target_obj=target_obj,
        color_map=color_map,
    )

    stats["small_far_filter"] = small_far_report

    return rgb_bytes, filtered_mask, stats


# ============================================================
# CAMERA SETUP
# ============================================================

def ensure_fusion_camera(client):
    """
    Find an existing FusionCameraSensor.

    If none exists, spawn one.

    Returns the UnrealCV camera ID.
    """

    cameras_response = client.request("vget /cameras")

    if not isinstance(cameras_response, str):
        raise RuntimeError(
            f"Unexpected response from vget /cameras: "
            f"{cameras_response}"
        )

    cameras = cameras_response.split()

    print("Registered cameras:")
    for i, name in enumerate(cameras):
        print(f"  /camera/{i} -> {name}")

    # Find an existing Fusion camera
    for camera_id, name in enumerate(cameras):
        if "FusionCamera" in name:
            print(
                f"\nUsing existing Fusion camera: "
                f"/camera/{camera_id} -> {name}"
            )
            return camera_id

    # --------------------------------------------------------
    # No Fusion camera exists, so spawn one
    # --------------------------------------------------------

    print("\nNo Fusion camera found.")
    print("Spawning Fusion camera...")

    response = client.request(
        "vset /cameras/spawn"
    )

    print(f"Spawn response: {response}")

    # Query again after spawning
    cameras_response = client.request(
        "vget /cameras"
    )

    cameras = cameras_response.split()

    print("\nRegistered cameras after spawn:")
    for i, name in enumerate(cameras):
        print(f"  /camera/{i} -> {name}")

    for camera_id, name in enumerate(cameras):
        if "FusionCamera" in name:
            print(
                f"\nUsing spawned Fusion camera: "
                f"/camera/{camera_id} -> {name}"
            )
            return camera_id

    raise RuntimeError(
        "Fusion camera could not be found after spawning."
    )


# ============================================================
# OBJECT LOCATION
# ============================================================

def get_object_location(client, object_name):
    """
    Get Unreal world location for an object.
    """

    response = client.request(
        f"vget /object/{object_name}/location"
    )

    try:
        x, y, z = map(float, response.split())
    except Exception:
        raise RuntimeError(
            f"Could not parse location for {object_name}: "
            f"{response}"
        )

    return x, y, z


def distance_between_locations(first, second):
    """Euclidean distance between two Unreal world locations."""
    return math.sqrt(
        sum((float(first[i]) - float(second[i])) ** 2 for i in range(3))
    )


def filter_small_far_masks(
    client,
    mask_bytes,
    camera_pos,
    target_obj,
    color_map,
):
    """
    Remove only tiny actor masks that are clearly farther from the camera
    than the selected target object.

    Pixel count alone is NOT enough to remove a mask.
    """
    report = {
        "enabled": ENABLE_SMALL_FAR_MASK_FILTER,
        "pixel_threshold": SMALL_MASK_PIXEL_THRESHOLD,
        "far_distance_margin_uu": FAR_BACKGROUND_DISTANCE_MARGIN_UU,
        "far_distance_ratio": FAR_BACKGROUND_DISTANCE_RATIO,
        "target_object": target_obj,
        "actors": {},
        "removed_actors": [],
    }

    if not ENABLE_SMALL_FAR_MASK_FILTER:
        return mask_bytes, report

    if not isinstance(mask_bytes, bytes):
        raise RuntimeError(
            f"Expected filtered object-mask PNG bytes, got {mask_bytes!r}"
        )

    mask = iio.imread(io.BytesIO(mask_bytes), extension=".png")
    if mask.ndim != 3 or mask.shape[2] not in (3, 4):
        raise RuntimeError(f"Expected RGB/RGBA object mask, got {mask.shape}")

    rgb = mask[:, :, :3]

    target_location = get_object_location(client, target_obj)
    target_distance = distance_between_locations(camera_pos, target_location)
    report["target_distance_uu"] = target_distance

    rejected_colors = []

    for actor_name, color in color_map.items():
        actor_color = np.asarray(color, dtype=np.uint8)

        if actor_color.shape != (3,):
            report["actors"][actor_name] = {
                "kept": True,
                "reason": f"invalid actor color shape: {actor_color.shape}",
            }
            continue

        actor_pixels = np.all(rgb == actor_color, axis=2)
        visible_pixels = int(np.count_nonzero(actor_pixels))
        small_mask = 0 < visible_pixels < SMALL_MASK_PIXEL_THRESHOLD

        actor_info = {
            "color_rgb": [int(v) for v in actor_color],
            "visible_pixels": visible_pixels,
            "small_mask": small_mask,
            "kept": True,
            "reason": "kept: visible mask meets pixel threshold",
        }

        if visible_pixels == 0:
            actor_info["reason"] = "not visible in this mask"
            report["actors"][actor_name] = actor_info
            continue

        protected_target = (
            ALWAYS_KEEP_TARGET_OBJECT and actor_name == target_obj
        )

        try:
            actor_location = get_object_location(client, actor_name)
            actor_distance = distance_between_locations(
                camera_pos, actor_location
            )
        except Exception as exc:
            actor_info["reason"] = (
                "kept because actor distance could not be measured: "
                f"{exc}"
            )
            report["actors"][actor_name] = actor_info
            continue

        actor_info["distance_uu"] = actor_distance
        actor_info["distance_beyond_target_uu"] = (
            actor_distance - target_distance
        )
        actor_info["distance_ratio_to_target"] = (
            actor_distance / target_distance
            if target_distance > 0.0
            else None
        )

        far_by_margin = (
            actor_distance
            >= target_distance + FAR_BACKGROUND_DISTANCE_MARGIN_UU
        )
        far_by_ratio = (
            actor_distance
            >= target_distance * FAR_BACKGROUND_DISTANCE_RATIO
        )
        is_far_background = far_by_margin and far_by_ratio

        actor_info["far_by_margin"] = far_by_margin
        actor_info["far_by_ratio"] = far_by_ratio
        actor_info["far_background"] = is_far_background

        should_remove = (
            small_mask
            and is_far_background
            and not protected_target
        )

        if should_remove:
            rejected_colors.append(actor_color.copy())
            actor_info["kept"] = False
            actor_info["reason"] = (
                "removed: below pixel threshold and sufficiently farther "
                "than target"
            )
            report["removed_actors"].append(actor_name)

        elif protected_target:
            actor_info["reason"] = (
                "kept: selected target object is protected"
            )

        elif small_mask and not is_far_background:
            actor_info["reason"] = (
                "kept: small pixel count, but actor is not far enough away; "
                "preserves nearby/occluded foreground objects"
            )

        else:
            actor_info["reason"] = (
                "kept: visible mask meets pixel threshold"
            )

        report["actors"][actor_name] = actor_info

    if rejected_colors:
        filtered = mask.copy()

        for actor_color in rejected_colors:
            remove_pixels = np.all(
                filtered[:, :, :3] == actor_color,
                axis=2,
            )
            filtered[:, :, :3][remove_pixels] = OCCLUDED_COLOR

        mask_bytes = iio.imwrite(
            "<bytes>",
            filtered,
            extension=".png",
        )

    return mask_bytes, report


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def calculate_look_at_rotation(camera_pos, target_pos):
    """
    Calculate Unreal pitch/yaw/roll needed to point
    the camera toward target_pos.
    """

    cx, cy, cz = camera_pos
    tx, ty, tz = target_pos

    dx = tx - cx
    dy = ty - cy
    dz = tz - cz

    horizontal_distance = math.sqrt(
        dx * dx + dy * dy
    )

    yaw = math.degrees(
        math.atan2(dy, dx)
    )

    pitch = math.degrees(
        math.atan2(dz, horizontal_distance)
    )

    roll = 0.0

    return pitch, yaw, roll


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image_bytes(path, data):
    """
    Save UnrealCV binary image response.
    """

    if not isinstance(data, bytes):
        raise RuntimeError(
            f"Expected bytes for {path}, "
            f"but UnrealCV returned:\n{data}"
        )

    with open(path, "wb") as f:
        f.write(data)


# ============================================================
# SINGLE ORBIT
# ============================================================

def capture_orbit(
    client,
    camera_id,
    target_obj,
    radius,
    elevation_deg,
    steps,
    outdir,
    color_map
):
    """
    Capture one 360-degree orbit around target_obj
    at a specified horizontal radius and elevation angle.
    """

    target = get_object_location(
        client,
        target_obj
    )

    tx, ty, tz = target

    # --------------------------------------------------------
    # Height is calculated from elevation angle.
    #
    # tan(elevation) = height / radius
    #
    # therefore:
    #
    # height = radius * tan(elevation)
    # --------------------------------------------------------

    height = radius * math.tan(
        math.radians(elevation_deg)
    )

    print()
    print("=" * 70)
    print(
        f"Radius: {radius} uu "
        f"({radius / 100.0:.1f} m)"
    )
    print(
        f"Elevation: {elevation_deg} deg"
    )
    print(
        f"Height above target: {height:.1f} uu "
        f"({height / 100.0:.2f} m)"
    )
    print("=" * 70)

    os.makedirs(outdir, exist_ok=True)

    for i in range(steps):

        azimuth_deg = (
            360.0 * i / steps
        )

        azimuth_rad = math.radians(
            azimuth_deg
        )

        # ----------------------------------------------------
        # Position camera on circular orbit
        # ----------------------------------------------------

        cx = (
            tx
            + radius * math.cos(azimuth_rad)
        )

        cy = (
            ty
            + radius * math.sin(azimuth_rad)
        )

        cz = (
            tz
            + height
        )

        camera_pos = (
            cx,
            cy,
            cz,
        )

        # ----------------------------------------------------
        # Point camera toward target
        # ----------------------------------------------------

        pitch, yaw, roll = (
            calculate_look_at_rotation(
                camera_pos,
                target,
            )
        )

        # ----------------------------------------------------
        # Move Fusion camera
        # ----------------------------------------------------

        location_response = client.request(
            f"vset /camera/{camera_id}/location "
            f"{cx} {cy} {cz}"
        )

        rotation_response = client.request(
            f"vset /camera/{camera_id}/rotation "
            f"{pitch} {yaw} {roll}"
        )

        # ----------------------------------------------------
        # Read back actual position/rotation
        # ----------------------------------------------------

        actual_location = client.request(
            f"vget /camera/{camera_id}/location"
        )

        actual_rotation = client.request(
            f"vget /camera/{camera_id}/rotation"
        )

        print(
            f"[{i + 1:03d}/{steps:03d}] "
            f"azimuth={azimuth_deg:6.1f} deg | "
            f"loc={actual_location} | "
            f"rot={actual_rotation}"
        )

        # Capture matching images/depths and filter BEFORE counting or saving.
        rgb_bytes, mask_bytes, visibility = capture_occlusion_frame(
            client,
            camera_id,
            target_obj,
            color_map,
            camera_pos,
        )

        target_pixels = visibility["visible"]

        small_far_report = visibility.get("small_far_filter", {})
        removed_actors = small_far_report.get("removed_actors", [])
        print(
            f"Small/far filter: removed {len(removed_actors)} actor(s) | "
            f"{removed_actors if removed_actors else 'none'}"
        )
        print(
            f"Target mask: raw={visibility['raw']} | "
            f"visible={target_pixels} | occluded={visibility['occluded']} | "
            f"invalid_depth={visibility['invalid_depth']} | "
            f"all_labels_occluded={visibility['all_labels_occluded']} | "
            f"all_labels_invalid_depth={visibility['all_labels_invalid_depth']}"
        )

        if target_pixels < MIN_TARGET_PIXELS:
            print(
                f"[SKIPPED] Target not visible | "
                f"target={target_obj} | "
                f"radius={radius} | "
                f"elevation={elevation_deg} | "
                f"azimuth={azimuth_deg:.1f}"
            )
            continue

        # ----------------------------------------------------
        # Valid pair -- now save both
        # ----------------------------------------------------

        # ----------------------------------------------------
        # File names
        # ----------------------------------------------------
        print(
            f"Target visible: {target_pixels} pixels"
        )
        
        filename_base = (
            f"{target_obj}"
            f"_r{int(radius):04d}"
            f"_el{int(elevation_deg):02d}"
            f"_az{int(round(azimuth_deg)):03d}"
        )

        rgb_path = os.path.join(
            outdir,
            f"{filename_base}_rgb.png"
        )

        mask_path = os.path.join(
            outdir,
            f"{filename_base}_mask.png"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_image_bytes(
            rgb_path,
            rgb_bytes
        )

        save_image_bytes(
            mask_path,
            mask_bytes
        )


# ============================================================
# MULTI-RADIUS / MULTI-ELEVATION CAPTURE
# ============================================================

def capture_dataset(
    client,
    camera_id,
    target_obj,
    radii,
    elevations,
    steps,
    output_dir,
    color_map
):
    """
    Capture every requested:
        radius x elevation x azimuth
    combination.
    """

    total_images = (
        len(radii)
        * len(elevations)
        * steps
    )

    print()
    print("Capture configuration")
    print("---------------------")
    print(f"Target:       {target_obj}")
    print(f"Camera ID:    {camera_id}")
    print(f"Radii:        {radii}")
    print(f"Elevations:   {elevations}")
    print(f"Orbit steps:  {steps}")
    print(f"RGB frames:   {total_images}")
    print(f"Mask frames:  {total_images}")
    print(
        f"Total files:  {total_images * 2}"
    )
    print(f"Output:       {output_dir}")

    for radius in radii:

        for elevation in elevations:

            orbit_dir = os.path.join(
                output_dir,
                f"radius_{int(radius):04d}",
                f"elevation_{int(elevation):02d}",
            )

            capture_orbit(
                client=client,
                camera_id=camera_id,
                target_obj=target_obj,
                radius=radius,
                elevation_deg=elevation,
                steps=steps,
                outdir=orbit_dir,
                color_map=color_map
            )

    print()
    print("=" * 70)
    print("Capture complete.")
    print(f"Saved to: {output_dir}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    # ========================================================
    # CREATE EXPERIMENT DIRECTORY
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_{TARGET_OBJECT}_{DISTANCE_CATEGORY}"
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    logfile_path = os.path.join(
        run_dir,
        EXPERIMENT_LOG_FILENAME
    )

    logfile = open(
        logfile_path,
        "a",
        encoding="utf-8"
    )

    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Send print() and errors to terminal + logfile
    sys.stdout = Tee(
        original_stdout,
        logfile
    )

    sys.stderr = Tee(
        original_stderr,
        logfile
    )

    try:

        print("=" * 70)
        print("UNREALCV CAPTURE EXPERIMENT")
        print("=" * 70)

        print(f"Start time: {datetime.now()}")
        print(f"Target: {TARGET_OBJECT}")
        print(f"Experiment directory: {run_dir}")
        print(f"Log file: {logfile_path}")
        
        # --------------------------------------------------------
        # Connect
        # --------------------------------------------------------

        client = Client(
            (HOST, PORT)
        )

        client.connect()

        if not client.isconnected():
            raise RuntimeError(
                f"Could not connect to UnrealCV "
                f"at {HOST}:{PORT}"
            )

        print("Connected to UnrealCV.")
        
        # --------------------------------------------------------
        # Confirm target exists
        # --------------------------------------------------------

        objects = client.request(
            "vget /objects"
        ).split()

        if TARGET_OBJECT not in objects:
            raise RuntimeError(
                f"Target object '{TARGET_OBJECT}' "
                f"was not found in UnrealCV."
            )
          
        # Check if the color map file exists on your system
        if os.path.exists(ACTORLIST):
            with open(ACTORLIST, "r") as actorfile:
                actors = json.load(actorfile)
            print(f"{ACTORLIST} loaded successfully.")
        else:
            raise RuntimeError(
                f"{ACTORLIST} was not loaded "
            )
        if os.path.exists(COLORMAP):
            with open(COLORMAP, "r") as colorfile:
                color_map = json.load(colorfile)
            print(f"{COLORMAP} loaded successfully.")
            any_missing = set(actors) - color_map.keys()    
            if len(any_missing) > 0: 
                raise RuntimeError(f"{COLORMAP} doesnt match your intended objects")
        else:
            print("File not found. Initialized empty data.")
            
            color_map = {}

            # Set how many unique combinations you want to generate
            number_of_combinations = len(actors)
            unique_combinations = set()

            while len(unique_combinations) < number_of_combinations:
                # Individual numbers can repeat (e.g., 42, 42, 105)
                new_tuple = tuple(random.randint(1, 254) for _ in range(3))
                
                # The set ensures the exact same combination isn't added twice
                unique_combinations.add(new_tuple)

            # Print the results
            color_map = dict(zip(actors, unique_combinations))
            #print(color_mapping)
            # Save data directly to a file
            with open(COLORMAP, "w") as file:
                json.dump(color_map, file, indent=4)
            #make sure you copy the file into your unreal colormap path
            copy2(COLORMAP, UNREALCV_COLORMAP_PATH)
          

        if TARGET_OBJECT not in color_map:
            raise RuntimeError(f"Target {TARGET_OBJECT} is missing from {COLORMAP}")
        if any(tuple(color) == OCCLUDED_COLOR for color in color_map.values()):
            raise RuntimeError("Occluded/background color must not be an actor color")

        response = client.request(
            f"vset /objects/annotation_colors_json {COLORMAP}"
        )
        
        print(response)

        # --------------------------------------------------------
        # Find/create Fusion camera
        # --------------------------------------------------------

        camera_id = ensure_fusion_camera(
            client
        )
        client.request(
            f"vset /camera/{camera_id}/size "
            f"{CAMERA_WIDTH} {CAMERA_HEIGHT}"
        )

        client.request(
            f"vset /camera/{camera_id}/fov "
            f"{CAMERA_FOV}"
        )
        
        initial_camera_info = get_camera_info(
            client,
            camera_id
        )
        
        # Save experiment parameters:
        experiment_config = {
            "timestamp": timestamp,
            "target_object": TARGET_OBJECT,
            "camera": {
                "id": camera_id,
                "name": "FusionCameraSensor",
                "initial_location": initial_camera_info["location"],
                "initial_rotation": initial_camera_info["rotation"],
                "initial_fov": initial_camera_info["fov"],
                "image_size": initial_camera_info["size"],
            },
            "radii_uu": RADII,
            "radii_meters": [
                radius / 100.0
                for radius in RADII
            ],
            "elevations_deg": ELEVATIONS,
            "steps_per_orbit": STEPS,
            "occlusion_filter": {
                "scope": "all_non_background_labels",
                "annotation_depth_command": "depth npy",
                "scene_depth_command": "scene_depth npy",
                "absolute_tolerance_uu": DEPTH_ABSOLUTE_TOLERANCE_UU,
                "relative_tolerance": DEPTH_RELATIVE_TOLERANCE,
                "removed_pixel_color": OCCLUDED_COLOR,
                "invalid_depth_policy": "discard_labelled_pixel",
                "minimum_visible_target_pixels": MIN_TARGET_PIXELS,
            },
            "small_far_mask_filter": {
                "enabled": ENABLE_SMALL_FAR_MASK_FILTER,
                "pixel_threshold": SMALL_MASK_PIXEL_THRESHOLD,
                "far_distance_margin_uu": FAR_BACKGROUND_DISTANCE_MARGIN_UU,
                "far_distance_ratio": FAR_BACKGROUND_DISTANCE_RATIO,
                "always_keep_target_object": ALWAYS_KEEP_TARGET_OBJECT,
                "processing_order": "after_depth_occlusion_filter",
            },
            "actor_list_file": ACTORLIST,
            "colormap_file": COLORMAP,
            "host": HOST,
            "port": PORT,
        }

        config_path = os.path.join(
            run_dir,
            EXPERIMENT_CONFIG_NAME
        )

        with open(
            config_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                experiment_config,
                f,
                indent=4
            )

        print(
            f"Experiment configuration saved: "
            f"{config_path}"
  )

        # --------------------------------------------------------
        # Capture
        # --------------------------------------------------------

        capture_dataset(
            client=client,
            camera_id=camera_id,
            target_obj=TARGET_OBJECT,
            radii=RADII,
            elevations=ELEVATIONS,
            steps=STEPS,
            output_dir=run_dir,
            color_map = color_map
        )
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print(f"End time: {datetime.now()}")
        print("=" * 70)

    except Exception:

        print()
        print("=" * 70)
        print("EXPERIMENT FAILED")
        print("=" * 70)

        # Full traceback goes into console AND log
        traceback.print_exc()

        raise

    finally:

      # Restore normal stdout/stderr BEFORE closing file
      sys.stdout = original_stdout
      sys.stderr = original_stderr

      logfile.close()

      print(
          f"Experiment log saved to: "
          f"{logfile_path}"
      )

if __name__ == "__main__":
    main()