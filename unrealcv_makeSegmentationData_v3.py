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
UNREALCV_COLORMAP_PATH = r"C:\Users\casab\Documents\Unreal Projects\military_base_3maps\Saved\UnrealCV\colormap_output.json"
ACTORLIST = "actors.json"
HOST = "localhost"
PORT = 9000
EXPERIMENT_LOG_FILENAME = "experiment.log"
EXPERIMENT_CONFIG_NAME = "experiment_config.json"
MIN_TARGET_PIXELS = 500
TARGET_OBJECT = "BP_Fighter_jet_2_C_3"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FOV = 90

# Unreal units: 100 uu = 1 meter
RADII = [
    500,    # 5 m
    1000,   # 10 m
    1500,   # 15 m
    2000,   # 20 m
]

# Elevation angle above the target, in degrees.
# Use [15] if you only want one elevation.
ELEVATIONS = [
    5,
    15,
    30,
]

# Number of positions around each 360-degree orbit
STEPS = 36

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

def count_target_pixels(mask_bytes, target_obj, color_map):
    """
    Count how many pixels in the object mask belong
    to the target object.
    """

    mask = iio.imread(
        io.BytesIO(mask_bytes),
        extension=".png"
    )

    rgb = mask[:, :, :3]

    target_color = np.array(
        color_map[target_obj],
        dtype=np.uint8
    )

    matches = np.all(
        rgb == target_color,
        axis=2
    )

    return np.count_nonzero(matches)
    
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

        # ----------------------------------------------------
        # Capture RGB
        # ----------------------------------------------------

        rgb_bytes = client.request(
            f"vget /camera/{camera_id}/lit png"
        )

        # ----------------------------------------------------
        # Capture instance/object mask
        # ----------------------------------------------------

        mask_bytes = client.request(
            f"vget /camera/{camera_id}/object_mask png"
        )
        # ----------------------------------------------------
        # Validate mask BEFORE saving pair
        # ----------------------------------------------------

        target_pixels = count_target_pixels(
            mask_bytes,
            target_obj,
            color_map
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
        f"{timestamp}_{TARGET_OBJECT}"
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