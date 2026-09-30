from __future__ import print_function

import base64
import json
import math
import os
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
from unrealcv import Client


# ---------------------------------------------------------------------------
# UnrealCV connection and output settings
# ---------------------------------------------------------------------------
UNREALCV_HOST = "localhost"
UNREALCV_PORT = 9000

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# The script uses `vget /objects` and performs a case-insensitive search for
# this text inside every Unreal runtime actor ID.
OBJECT_ID_SEARCH_TERM = "shahed"

# When several matching actor IDs are found, all of them are configured for
# object-mask output. The user selects one matching actor as the camera target.
COLOR_MAP_FILE = "colormap_output.json"
OUTPUT_DIRECTORY = "unrealcv_captures"

# ---------------------------------------------------------------------------
# Small-object / far-background segmentation filtering
# ---------------------------------------------------------------------------
# UnrealCV initially paints every matching actor. After each object-mask image
# is captured, the script counts the visible pixels belonging to each actor.
# An actor is removed from the saved mask ONLY when it is BOTH:
#   1) smaller than SMALL_MASK_PIXEL_THRESHOLD, and
#   2) clearly farther from the camera than the selected target actor.
#
# This two-part rule is intentional: a foreground actor that is only showing
# a tiny visible sliver because of occlusion is still retained.
ENABLE_SMALL_FAR_MASK_FILTER = True
SMALL_MASK_PIXEL_THRESHOLD = 100

# To qualify as "far background", the actor must satisfy BOTH tests below.
# 1500 cm = 15 m. The ratio test keeps the rule proportional at long ranges.
FAR_BACKGROUND_DISTANCE_MARGIN_CM = 1500.0
FAR_BACKGROUND_DISTANCE_RATIO = 1.15

# The actor explicitly chosen as the camera target is always protected, even
# if only a few of its pixels remain visible in a heavily occluded image.
ALWAYS_KEEP_TARGET_ACTOR = True

# Spawn an independent camera so a PlayerController or FusionCameraSensor
# does not reset camera 0 to its original transform every frame.
SPAWN_DEDICATED_CAMERA = True
FALLBACK_CAMERA_ID = 0

# Unreal Engine generally uses centimeters.
#
# The user selects one of these starting-distance categories at runtime.
# A new starting distance is randomly selected from the chosen range every
# time the script runs.
#
# The closest option begins at exactly 5800 cm, as requested.
DISTANCE_PRESETS = {
    "close": {
        "label": "Close",
        "minimum_cm": 5800.0,
        "maximum_cm": 7500.0,
    },
    "medium": {
        "label": "Medium",
        "minimum_cm": 8000.0,
        "maximum_cm": 11000.0,
    },
    "long": {
        "label": "Long",
        "minimum_cm": 12000.0,
        "maximum_cm": 18000.0,
    },
}

# These values are initialized here and replaced after the user chooses a
# distance category. Keeping them initialized also makes the helper functions
# safe to import independently.
SELECTED_DISTANCE_PRESET = "close"
START_CAMERA_DISTANCE = DISTANCE_PRESETS["close"]["minimum_cm"]

START_AZIMUTH_DEGREES = 180.0
START_ELEVATION_DEGREES = 15.0

# Every randomized distance after the first image is derived from the randomly
# selected START_CAMERA_DISTANCE.
DISTANCE_VARIATION_FRACTION = 0.20
MIN_CAMERA_DISTANCE = START_CAMERA_DISTANCE * (1.0 - DISTANCE_VARIATION_FRACTION)
MAX_CAMERA_DISTANCE = START_CAMERA_DISTANCE * (1.0 + DISTANCE_VARIATION_FRACTION)

# Elevation controls the camera's orbital position around the target.
MIN_ELEVATION_DEGREES = 8.0
MAX_ELEVATION_DEGREES = 25.0

# Prevent consecutive samples from landing on almost the same side.
MIN_AZIMUTH_CHANGE_DEGREES = 35.0

# ---------------------------------------------------------------------------
# Distance-scaled camera rotation randomization
# ---------------------------------------------------------------------------
# At exactly START_CAMERA_DISTANCE, the camera can deviate from the exact
# look-at rotation by these amounts. The first image remains exactly centered;
# these offsets are used for randomized images 2 and later.
BASE_YAW_JITTER_DEGREES = 5.0
BASE_PITCH_JITTER_DEGREES = 3.0
BASE_ROLL_JITTER_DEGREES = 0.0

# The jitter scale is distance / START_CAMERA_DISTANCE. Clamp the scale so a
# large distance cannot rotate the camera completely away from the ship.
MIN_ROTATION_DISTANCE_SCALE = 0.60
MAX_ROTATION_DISTANCE_SCALE = 1.60

# Absolute safety caps.
MAX_YAW_JITTER_DEGREES = 10.0
MAX_PITCH_JITTER_DEGREES = 6.0
MAX_ROLL_JITTER_DEGREES = 1.0

# Leave False for upright dataset images. Set True to introduce slight roll.
RANDOMIZE_CAMERA_ROLL = False

# Aim above the actor origin, which is often near the waterline.
# For a large ship, this helps aim toward the center of the visible hull and
# superstructure instead of toward the waterline.
TARGET_Z_OFFSET = 500.0

CAMERA_SETTLE_SECONDS = 0.75
CAMERA_SPAWN_SETTLE_SECONDS = 1.0

# Maximum allowed disagreement between requested and reported camera pose.
LOCATION_TOLERANCE_CM = 10.0
ROTATION_TOLERANCE_DEGREES = 3.0


Pose = Tuple[float, float, float, float, float, float]
Location = Tuple[float, float, float]
Rotation = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def ask_for_distance_preset() -> Tuple[str, float]:
    """Prompt for a distance category and randomize the starting distance."""
    aliases = {
        "1": "close",
        "c": "close",
        "close": "close",
        "2": "medium",
        "m": "medium",
        "medium": "medium",
        "3": "long",
        "l": "long",
        "long": "long",
    }

    print("\nChoose the camera starting-distance category:")
    print(
        "  1. Close  ({:.0f}-{:.0f} cm)".format(
            DISTANCE_PRESETS["close"]["minimum_cm"],
            DISTANCE_PRESETS["close"]["maximum_cm"],
        )
    )
    print(
        "  2. Medium ({:.0f}-{:.0f} cm)".format(
            DISTANCE_PRESETS["medium"]["minimum_cm"],
            DISTANCE_PRESETS["medium"]["maximum_cm"],
        )
    )
    print(
        "  3. Long   ({:.0f}-{:.0f} cm)".format(
            DISTANCE_PRESETS["long"]["minimum_cm"],
            DISTANCE_PRESETS["long"]["maximum_cm"],
        )
    )

    while True:
        answer = input(
            "Enter 1/2/3, or close/medium/long: "
        ).strip().lower()

        preset_name = aliases.get(answer)
        if preset_name is None:
            print(
                "Invalid selection. Enter 1, 2, 3, close, medium, or long."
            )
            continue

        preset = DISTANCE_PRESETS[preset_name]

        # Round to the nearest centimeter so the selected value remains random
        # but is easy to read in the console and metadata.
        start_distance = float(
            random.randint(
                int(preset["minimum_cm"]),
                int(preset["maximum_cm"]),
            )
        )

        return preset_name, start_distance


def apply_start_distance(
    preset_name: str,
    start_distance: float,
) -> None:
    """Update all runtime distance values from the selected start distance."""
    global SELECTED_DISTANCE_PRESET
    global START_CAMERA_DISTANCE
    global MIN_CAMERA_DISTANCE
    global MAX_CAMERA_DISTANCE

    if preset_name not in DISTANCE_PRESETS:
        raise ValueError(
            "Unknown distance preset: {!r}".format(preset_name)
        )

    if start_distance <= 0.0:
        raise ValueError("The starting camera distance must be greater than zero.")

    SELECTED_DISTANCE_PRESET = preset_name
    START_CAMERA_DISTANCE = float(start_distance)

    MIN_CAMERA_DISTANCE = START_CAMERA_DISTANCE * (
        1.0 - DISTANCE_VARIATION_FRACTION
    )
    MAX_CAMERA_DISTANCE = START_CAMERA_DISTANCE * (
        1.0 + DISTANCE_VARIATION_FRACTION
    )


def ask_for_image_count() -> int:
    while True:
        value = input("How many RGB/mask pairs would you like to capture? ").strip()

        try:
            count = int(value)
        except ValueError:
            print("Enter a whole number, such as 10.")
            continue

        if count <= 0:
            print("The number must be greater than zero.")
            continue

        return count


def sanitize_filename(value: str) -> str:
    """Return an actor name that is safe to use as part of a filename."""
    safe_characters = []

    for character in value:
        if character.isalnum() or character in ("-", "_", "."):
            safe_characters.append(character)
        else:
            safe_characters.append("_")

    sanitized = "".join(safe_characters).strip("._")
    return sanitized or "ship_actor"


def find_matching_actor_ids(
    objects: List[str],
    search_term: str,
) -> List[str]:
    """Find actor IDs containing search_term, ignoring letter case."""
    normalized_term = search_term.strip().casefold()

    if not normalized_term:
        raise ValueError("OBJECT_ID_SEARCH_TERM cannot be empty.")

    return sorted(
        object_name
        for object_name in objects
        if normalized_term in object_name.casefold()
    )


def choose_target_actor(
    matching_actors: List[str],
    search_term: str,
) -> str:
    """Choose which matching actor should be used as the camera target."""
    if not matching_actors:
        raise ValueError("matching_actors cannot be empty.")

    print(
        '\nUnrealCV actor IDs containing "{}":'.format(search_term)
    )

    for index, actor_name in enumerate(matching_actors, start=1):
        print("  {}. {}".format(index, actor_name))

    if len(matching_actors) == 1:
        selected_actor = matching_actors[0]
        print(
            "Only one matching actor was found; selecting it automatically: "
            "{}".format(selected_actor)
        )
        return selected_actor

    while True:
        value = input(
            "Select the actor number to use as the camera target: "
        ).strip()

        try:
            selected_index = int(value)
        except ValueError:
            print(
                "Enter a number from 1 to {}.".format(len(matching_actors))
            )
            continue

        if not 1 <= selected_index <= len(matching_actors):
            print(
                "Enter a number from 1 to {}.".format(len(matching_actors))
            )
            continue

        return matching_actors[selected_index - 1]


def response_to_text(response: object) -> str:
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace").strip()
    return str(response).strip()


def parse_floats(response: object, expected_count: Optional[int] = None) -> List[float]:
    text = response_to_text(response)
    values = [float(value) for value in text.replace(",", " ").split()]

    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            "Expected {} numeric values but received {}: {!r}".format(
                expected_count, len(values), text
            )
        )

    return values


def angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def normalize_angle_degrees(angle: float) -> float:
    """Normalize an angle to the Unreal-friendly range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def location_error(requested: Sequence[float], actual: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(requested[:3], actual[:3]))
    )


def rotation_error(requested: Sequence[float], actual: Sequence[float]) -> float:
    return max(
        angular_difference(float(a), float(b))
        for a, b in zip(requested[3:6], actual[3:6])
    )


def save_png_response(response: Union[bytes, str], output_path: str) -> bool:
    if isinstance(response, bytes):
        image_bytes = response
    elif isinstance(response, str):
        try:
            image_bytes = base64.b64decode(response, validate=True)
        except Exception:
            print("Capture failed for {}: {}".format(output_path, response))
            return False
    else:
        print(
            "Capture failed for {}: unsupported response type {}".format(
                output_path, type(response).__name__
            )
        )
        return False

    # A valid PNG begins with this eight-byte signature.
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        print("Capture failed for {}: response is not a PNG.".format(output_path))
        return False

    with open(output_path, "wb") as image_file:
        image_file.write(image_bytes)

    return True


# ---------------------------------------------------------------------------
# UnrealCV camera helpers
# ---------------------------------------------------------------------------
def list_cameras(client: Client) -> List[str]:
    response = client.request("vget /cameras")
    cameras = response_to_text(response).split()
    return cameras


def spawn_capture_camera(client: Client) -> Tuple[int, List[str]]:
    cameras_before = list_cameras(client)
    print("Cameras before spawn: {}".format(cameras_before))

    if not SPAWN_DEDICATED_CAMERA:
        return FALLBACK_CAMERA_ID, cameras_before

    spawn_response = client.request("vset /cameras/spawn")
    print("Spawn response: {}".format(response_to_text(spawn_response)))
    time.sleep(CAMERA_SPAWN_SETTLE_SECONDS)

    cameras_after = list_cameras(client)
    print("Cameras after spawn: {}".format(cameras_after))

    if len(cameras_after) <= len(cameras_before):
        print(
            "WARNING: UnrealCV did not report a new camera. "
            "Falling back to camera {}.".format(FALLBACK_CAMERA_ID)
        )
        return FALLBACK_CAMERA_ID, cameras_after

    camera_id = len(cameras_after) - 1
    print(
        "Using spawned camera ID {} ({}) for capture.".format(
            camera_id, cameras_after[camera_id]
        )
    )
    return camera_id, cameras_after


def get_camera_pose(client: Client, camera_id: int) -> Pose:
    # Prefer the combined pose command.
    response = client.request("vget /camera/{}/pose".format(camera_id))

    try:
        values = parse_floats(response, expected_count=6)
        return tuple(values)  # type: ignore[return-value]
    except (TypeError, ValueError):
        # Fall back to separate location and rotation commands.
        location = parse_floats(
            client.request("vget /camera/{}/location".format(camera_id)),
            expected_count=3,
        )
        rotation = parse_floats(
            client.request("vget /camera/{}/rotation".format(camera_id)),
            expected_count=3,
        )
        return tuple(location + rotation)  # type: ignore[return-value]


def set_camera_pose_verified(
    client: Client,
    camera_id: int,
    requested_pose: Pose,
) -> Pose:
    x, y, z, pitch, yaw, roll = requested_pose

    pose_command = (
        "vset /camera/{}/pose "
        "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}"
    ).format(camera_id, x, y, z, pitch, yaw, roll)

    pose_response = client.request(pose_command)
    print("  pose set response: {}".format(response_to_text(pose_response)))
    time.sleep(CAMERA_SETTLE_SECONDS)

    actual_pose = get_camera_pose(client, camera_id)
    loc_error = location_error(requested_pose, actual_pose)
    rot_error = rotation_error(requested_pose, actual_pose)

    # Some UnrealCV builds expose /pose but do not correctly implement setting it.
    # Retry with the older location and rotation commands when needed.
    if loc_error > LOCATION_TOLERANCE_CM or rot_error > ROTATION_TOLERANCE_DEGREES:
        location_response = client.request(
            "vset /camera/{}/location {:.3f} {:.3f} {:.3f}".format(
                camera_id, x, y, z
            )
        )
        rotation_response = client.request(
            "vset /camera/{}/rotation {:.3f} {:.3f} {:.3f}".format(
                camera_id, pitch, yaw, roll
            )
        )

        print(
            "  location set response: {}".format(
                response_to_text(location_response)
            )
        )
        print(
            "  rotation set response: {}".format(
                response_to_text(rotation_response)
            )
        )

        time.sleep(CAMERA_SETTLE_SECONDS)
        actual_pose = get_camera_pose(client, camera_id)
        loc_error = location_error(requested_pose, actual_pose)
        rot_error = rotation_error(requested_pose, actual_pose)

    if loc_error > LOCATION_TOLERANCE_CM or rot_error > ROTATION_TOLERANCE_DEGREES:
        raise RuntimeError(
            "UnrealCV did not apply the requested camera pose.\n"
            "Requested: {}\n"
            "Reported:  {}\n"
            "Location error: {:.2f} cm\n"
            "Rotation error: {:.2f} degrees".format(
                requested_pose, actual_pose, loc_error, rot_error
            )
        )

    return actual_pose


# ---------------------------------------------------------------------------
# Target and pose generation
# ---------------------------------------------------------------------------
def get_actor_location(client: Client, actor_name: str) -> Location:
    """Return the actor's raw Unreal world location without any aim offset."""
    location_response = client.request(
        "vget /object/{}/location".format(actor_name)
    )
    x, y, z = parse_floats(location_response, expected_count=3)
    return x, y, z


def get_target_location(client: Client, actor_name: str) -> Location:
    x, y, z = get_actor_location(client, actor_name)
    return x, y, z + TARGET_Z_OFFSET


def calculate_look_at_rotation(
    camera_location: Location,
    target_location: Location,
) -> Rotation:
    camera_x, camera_y, camera_z = camera_location
    target_x, target_y, target_z = target_location

    delta_x = target_x - camera_x
    delta_y = target_y - camera_y
    delta_z = target_z - camera_z

    horizontal_distance = math.hypot(delta_x, delta_y)
    yaw = math.degrees(math.atan2(delta_y, delta_x))
    pitch = math.degrees(math.atan2(delta_z, horizontal_distance))
    roll = 0.0

    return pitch, yaw, roll


def choose_azimuth(previous_azimuth: Optional[float]) -> float:
    for _ in range(100):
        candidate = random.uniform(0.0, 360.0)

        if previous_azimuth is None:
            return candidate

        if angular_difference(candidate, previous_azimuth) >= MIN_AZIMUTH_CHANGE_DEGREES:
            return candidate

    # This should be extremely unlikely, but guarantees forward progress.
    return ((previous_azimuth or 0.0) + MIN_AZIMUTH_CHANGE_DEGREES) % 360.0


def get_distance_scaled_rotation_limits(
    distance: float,
) -> Tuple[float, float, float, float]:
    """Return distance scale and pitch/yaw/roll jitter limits.

    At START_CAMERA_DISTANCE, the scale is 1.0. A closer camera receives less
    rotation jitter because the ship fills more of the frame. A farther camera
    receives more jitter because the ship occupies less of the frame.
    """
    if START_CAMERA_DISTANCE <= 0.0:
        raise ValueError("START_CAMERA_DISTANCE must be greater than zero.")

    distance_scale = clamp(
        distance / START_CAMERA_DISTANCE,
        MIN_ROTATION_DISTANCE_SCALE,
        MAX_ROTATION_DISTANCE_SCALE,
    )

    pitch_limit = min(
        BASE_PITCH_JITTER_DEGREES * distance_scale,
        MAX_PITCH_JITTER_DEGREES,
    )
    yaw_limit = min(
        BASE_YAW_JITTER_DEGREES * distance_scale,
        MAX_YAW_JITTER_DEGREES,
    )

    if RANDOMIZE_CAMERA_ROLL:
        roll_limit = min(
            BASE_ROLL_JITTER_DEGREES * distance_scale,
            MAX_ROLL_JITTER_DEGREES,
        )
    else:
        roll_limit = 0.0

    return distance_scale, pitch_limit, yaw_limit, roll_limit


def apply_distance_scaled_rotation_randomization(
    base_rotation: Rotation,
    distance: float,
) -> Tuple[Rotation, Dict[str, float]]:
    """Add controlled pitch/yaw/roll offsets to the look-at rotation."""
    base_pitch, base_yaw, base_roll = base_rotation

    distance_scale, pitch_limit, yaw_limit, roll_limit = (
        get_distance_scaled_rotation_limits(distance)
    )

    pitch_offset = random.uniform(-pitch_limit, pitch_limit)
    yaw_offset = random.uniform(-yaw_limit, yaw_limit)
    roll_offset = (
        random.uniform(-roll_limit, roll_limit)
        if roll_limit > 0.0
        else 0.0
    )

    randomized_rotation = (
        base_pitch + pitch_offset,
        normalize_angle_degrees(base_yaw + yaw_offset),
        base_roll + roll_offset,
    )

    metadata = {
        "enabled": True,
        "distance_scale": distance_scale,
        "pitch_limit_degrees": pitch_limit,
        "yaw_limit_degrees": yaw_limit,
        "roll_limit_degrees": roll_limit,
        "pitch_offset_degrees": pitch_offset,
        "yaw_offset_degrees": yaw_offset,
        "roll_offset_degrees": roll_offset,
        "base_pitch_degrees": base_pitch,
        "base_yaw_degrees": base_yaw,
        "base_roll_degrees": base_roll,
    }

    return randomized_rotation, metadata


def generate_camera_pose(
    target_location: Location,
    azimuth: float,
    elevation: float,
    distance: float,
    randomize_rotation: bool = False,
) -> Tuple[Pose, Dict[str, float]]:
    """Create a camera pose at a spherical offset from the target.

    Camera location is controlled by azimuth, elevation, and distance.
    Rotation first looks directly at the target, then optionally receives
    distance-scaled pitch/yaw/roll randomization.
    """
    target_x, target_y, target_z = target_location

    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)

    horizontal_distance = distance * math.cos(elevation_radians)
    height = distance * math.sin(elevation_radians)

    camera_location = (
        target_x + horizontal_distance * math.cos(azimuth_radians),
        target_y + horizontal_distance * math.sin(azimuth_radians),
        target_z + height,
    )

    base_rotation = calculate_look_at_rotation(
        camera_location,
        target_location,
    )

    if randomize_rotation:
        rotation, rotation_metadata = (
            apply_distance_scaled_rotation_randomization(
                base_rotation,
                distance,
            )
        )
    else:
        distance_scale, pitch_limit, yaw_limit, roll_limit = (
            get_distance_scaled_rotation_limits(distance)
        )
        rotation = base_rotation
        rotation_metadata = {
            "enabled": False,
            "distance_scale": distance_scale,
            "pitch_limit_degrees": pitch_limit,
            "yaw_limit_degrees": yaw_limit,
            "roll_limit_degrees": roll_limit,
            "pitch_offset_degrees": 0.0,
            "yaw_offset_degrees": 0.0,
            "roll_offset_degrees": 0.0,
            "base_pitch_degrees": base_rotation[0],
            "base_yaw_degrees": base_rotation[1],
            "base_roll_degrees": base_rotation[2],
        }

    pitch, yaw, roll = rotation

    pose = (
        camera_location[0],
        camera_location[1],
        camera_location[2],
        pitch,
        yaw,
        roll,
    )

    return pose, rotation_metadata


def generate_random_camera_pose(
    target_location: Location,
    previous_azimuth: Optional[float],
) -> Tuple[Pose, float, float, float, Dict[str, float]]:
    azimuth = choose_azimuth(previous_azimuth)
    elevation = random.uniform(
        MIN_ELEVATION_DEGREES,
        MAX_ELEVATION_DEGREES,
    )

    # This range is automatically based on START_CAMERA_DISTANCE.
    distance = random.uniform(
        MIN_CAMERA_DISTANCE,
        MAX_CAMERA_DISTANCE,
    )

    pose, rotation_metadata = generate_camera_pose(
        target_location,
        azimuth,
        elevation,
        distance,
        randomize_rotation=True,
    )

    return pose, azimuth, elevation, distance, rotation_metadata


# ---------------------------------------------------------------------------
# Object-mask color setup
# ---------------------------------------------------------------------------
def load_or_create_color_mapping(
    actors: List[str],
) -> Dict[str, List[int]]:
    if os.path.exists(COLOR_MAP_FILE):
        with open(COLOR_MAP_FILE, "r") as file:
            color_mapping = json.load(file)

        missing = [actor for actor in actors if actor not in color_mapping]
        if not missing:
            print("Loaded segmentation colors from {}.".format(COLOR_MAP_FILE))
            return color_mapping

        print(
            "The existing color map is missing actors; regenerating it: {}".format(
                missing
            )
        )

    unique_colors = set()
    while len(unique_colors) < len(actors):
        unique_colors.add(
            tuple(random.randint(1, 254) for _ in range(3))
        )

    color_mapping = {
        actor: list(color)
        for actor, color in zip(actors, unique_colors)
    }

    with open(COLOR_MAP_FILE, "w") as file:
        json.dump(color_mapping, file, indent=4)

    print("Created segmentation colors in {}.".format(COLOR_MAP_FILE))
    return color_mapping


def set_object_mask_colors(
    client: Client,
    objects: List[str],
    actors: List[str],
    color_mapping: Dict[str, List[int]],
) -> None:
    actor_set = set(actors)

    for object_name in objects:
        if object_name in actor_set:
            red, green, blue = color_mapping[object_name]
        else:
            red, green, blue = 0, 0, 0

        response = client.request(
            "vset /object/{}/color {} {} {}".format(
                object_name, red, green, blue
            )
        )

        response_text = response_to_text(response).lower()
        if "error" in response_text:
            print(
                "WARNING: color command failed for {}: {}".format(
                    object_name, response_to_text(response)
                )
            )


def distance_between_locations(first: Sequence[float], second: Sequence[float]) -> float:
    """Euclidean distance between two Unreal world locations, in centimeters."""
    return math.sqrt(
        sum(
            (float(first[index]) - float(second[index])) ** 2
            for index in range(3)
        )
    )


def filter_small_far_actor_masks(
    client: Client,
    mask_path: str,
    camera_pose: Pose,
    actors: List[str],
    target_actor: str,
    color_mapping: Dict[str, List[int]],
) -> Dict[str, object]:
    """Remove only tiny matching actors that are clearly in the background.

    Pixel count alone is deliberately NOT enough to remove an actor. A small
    mask can be caused by distance, but it can also be caused by foreground
    occlusion. The actor therefore has to fail the pixel threshold AND pass
    both far-distance tests before its mask color is replaced with black.
    """
    report: Dict[str, object] = {
        "enabled": ENABLE_SMALL_FAR_MASK_FILTER,
        "pixel_threshold": SMALL_MASK_PIXEL_THRESHOLD,
        "far_distance_margin_cm": FAR_BACKGROUND_DISTANCE_MARGIN_CM,
        "far_distance_ratio": FAR_BACKGROUND_DISTANCE_RATIO,
        "actors": {},
        "removed_actors": [],
    }

    if not ENABLE_SMALL_FAR_MASK_FILTER:
        return report

    camera_location = (camera_pose[0], camera_pose[1], camera_pose[2])

    # Use the selected target's true actor origin as the depth reference.
    target_origin = get_actor_location(client, target_actor)
    target_distance = distance_between_locations(camera_location, target_origin)
    report["target_actor_distance_cm"] = target_distance

    with Image.open(mask_path) as opened_mask:
        mask_image = opened_mask.convert("RGB")

    # UnrealCV object masks use the exact actor color, so a color histogram gives
    # us the visible pixel count for every matching actor in a single pass.
    pixel_counts = Counter(mask_image.getdata())
    rejected_colors = set()

    for actor_name in actors:
        actor_color = tuple(int(value) for value in color_mapping[actor_name])
        visible_pixels = int(pixel_counts.get(actor_color, 0))

        actor_info: Dict[str, object] = {
            "color_rgb": list(actor_color),
            "visible_pixels": visible_pixels,
            "small_mask": (
                0 < visible_pixels < SMALL_MASK_PIXEL_THRESHOLD
            ),
            "kept": True,
            "reason": "visible mask meets pixel threshold",
        }

        if visible_pixels == 0:
            actor_info["reason"] = "not visible in this mask"
            report["actors"][actor_name] = actor_info  # type: ignore[index]
            continue

        try:
            actor_location = get_actor_location(client, actor_name)
            actor_distance = distance_between_locations(
                camera_location,
                actor_location,
            )
        except Exception as exc:
            # Conservative behavior: if depth cannot be measured, keep the mask.
            actor_info["reason"] = (
                "kept because actor distance could not be measured: {}".format(exc)
            )
            report["actors"][actor_name] = actor_info  # type: ignore[index]
            continue

        actor_info["distance_cm"] = actor_distance
        actor_info["distance_beyond_target_cm"] = actor_distance - target_distance
        actor_info["distance_ratio_to_target"] = (
            actor_distance / target_distance if target_distance > 0.0 else None
        )

        far_by_margin = (
            actor_distance
            >= target_distance + FAR_BACKGROUND_DISTANCE_MARGIN_CM
        )
        far_by_ratio = (
            actor_distance
            >= target_distance * FAR_BACKGROUND_DISTANCE_RATIO
        )
        is_far_background = far_by_margin and far_by_ratio

        actor_info["far_by_margin"] = far_by_margin
        actor_info["far_by_ratio"] = far_by_ratio
        actor_info["far_background"] = is_far_background

        protected_target = ALWAYS_KEEP_TARGET_ACTOR and actor_name == target_actor
        small_mask = 0 < visible_pixels < SMALL_MASK_PIXEL_THRESHOLD
        should_remove = (
            small_mask
            and is_far_background
            and not protected_target
        )

        if should_remove:
            rejected_colors.add(actor_color)
            actor_info["kept"] = False
            actor_info["reason"] = (
                "removed: below pixel threshold and sufficiently farther "
                "than target"
            )
            report["removed_actors"].append(actor_name)  # type: ignore[union-attr]
        elif protected_target:
            actor_info["reason"] = "kept: selected target actor is protected"
        elif small_mask and not is_far_background:
            actor_info["reason"] = (
                "kept: small pixel count, but actor is not far enough away; "
                "this preserves small foreground/occluded objects"
            )
        elif not small_mask:
            actor_info["reason"] = "kept: visible mask meets pixel threshold"

        report["actors"][actor_name] = actor_info  # type: ignore[index]

    if rejected_colors:
        pixels = list(mask_image.getdata())
        filtered_pixels = [
            (0, 0, 0) if pixel in rejected_colors else pixel
            for pixel in pixels
        ]
        mask_image.putdata(filtered_pixels)
        mask_image.save(mask_path)

    return report


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def main() -> None:
    preset_name, randomized_start_distance = ask_for_distance_preset()
    apply_start_distance(
        preset_name,
        randomized_start_distance,
    )

    preset_label = DISTANCE_PRESETS[preset_name]["label"]

    print(
        "\nSelected {} distance.".format(preset_label)
    )
    print(
        "Randomized starting camera distance: {:.0f} cm ({:.1f} m)".format(
            START_CAMERA_DISTANCE,
            START_CAMERA_DISTANCE / 100.0,
        )
    )

    image_count = ask_for_image_count()
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    client = Client((UNREALCV_HOST, UNREALCV_PORT))
    client.connect()

    status = client.request("vget /unrealcv/status")
    print("UnrealCV status: {}".format(response_to_text(status)))

    version = client.request("vget /unrealcv/version")
    print("UnrealCV version: {}".format(response_to_text(version)))

    # Ask UnrealCV for every runtime actor ID in the active Unreal level.
    objects_response = client.request("vget /objects")
    objects = response_to_text(objects_response).split()

    if not objects:
        raise RuntimeError(
            "vget /objects returned no Unreal actor IDs. "
            "Confirm that Unreal is in Play/PIE mode and UnrealCV is active."
        )

    # Search case-insensitively, so names containing ship, Ship, SHIP, etc.
    # are all accepted.
    actors = find_matching_actor_ids(
        objects,
        OBJECT_ID_SEARCH_TERM,
    )

    if not actors:
        preview_count = min(25, len(objects))
        object_preview = "\n".join(
            "  - {}".format(name)
            for name in sorted(objects)[:preview_count]
        )

        raise RuntimeError(
            'No Unreal actor ID containing "{}" was found with '
            "vget /objects.\n"
            "The first {} available actor IDs were:\n{}".format(
                OBJECT_ID_SEARCH_TERM,
                preview_count,
                object_preview,
            )
        )

    # If several actor IDs contain "ship", ask which one should define the
    # target location used by the camera look-at calculation.
    target_actor = choose_target_actor(
        actors,
        OBJECT_ID_SEARCH_TERM,
    )

    output_actor_name = sanitize_filename(target_actor)

    print(
        "\nFound {} total Unreal objects and {} actor ID(s) containing "
        '"{}".'.format(
            len(objects),
            len(actors),
            OBJECT_ID_SEARCH_TERM,
        )
    )
    print("Camera target actor: {}".format(target_actor))
    print(
        "All matching actor IDs will be assigned object-mask colors."
    )

    color_mapping = load_or_create_color_mapping(actors)
    set_object_mask_colors(client, objects, actors, color_mapping)

    camera_id, camera_names = spawn_capture_camera(client)

    size_response = client.request(
        "vset /camera/{}/size {} {}".format(
            camera_id, IMAGE_WIDTH, IMAGE_HEIGHT
        )
    )
    print("Camera size response: {}".format(response_to_text(size_response)))

    initial_pose = get_camera_pose(client, camera_id)
    print("Initial spawned-camera pose: {}".format(initial_pose))
    print(
        "First capture stand-off: preset={}, distance={:.1f} cm, "
        "azimuth={:.1f}, elevation={:.1f}".format(
            DISTANCE_PRESETS[SELECTED_DISTANCE_PRESET]["label"],
            START_CAMERA_DISTANCE,
            START_AZIMUTH_DEGREES,
            START_ELEVATION_DEGREES,
        )
    )
    print(
        "Randomized distance range derived from start distance: "
        "{:.1f}-{:.1f} cm".format(
            MIN_CAMERA_DISTANCE,
            MAX_CAMERA_DISTANCE,
        )
    )
    print(
        "At the start distance, randomized captures use up to "
        "+/-{:.1f} deg yaw and +/-{:.1f} deg pitch jitter.".format(
            BASE_YAW_JITTER_DEGREES,
            BASE_PITCH_JITTER_DEGREES,
        )
    )

    camera_records = []
    successful_pairs = 0
    previous_azimuth = None

    for image_number in range(1, image_count + 1):
        # Refresh the target point in case the ship moves during simulation.
        target_location = get_target_location(client, target_actor)

        if image_number == 1:
            # Capture the first pair from a known stand-off location before
            # beginning the randomized camera sequence.
            azimuth = START_AZIMUTH_DEGREES
            elevation = START_ELEVATION_DEGREES
            distance = START_CAMERA_DISTANCE

            requested_pose, rotation_randomization = generate_camera_pose(
                target_location,
                azimuth,
                elevation,
                distance,
                randomize_rotation=False,
            )
        else:
            (
                requested_pose,
                azimuth,
                elevation,
                distance,
                rotation_randomization,
            ) = generate_random_camera_pose(
                target_location,
                previous_azimuth,
            )

        previous_azimuth = azimuth

        print(
            "\n[{}/{}] Requested azimuth={:.1f}, elevation={:.1f}, "
            "distance={:.1f} cm".format(
                image_number,
                image_count,
                azimuth,
                elevation,
                distance,
            )
        )

        if rotation_randomization["enabled"]:
            print(
                "  rotation scale={:.3f} from distance/start_distance; "
                "pitch offset={:+.2f} deg, yaw offset={:+.2f} deg, "
                "roll offset={:+.2f} deg".format(
                    rotation_randomization["distance_scale"],
                    rotation_randomization["pitch_offset_degrees"],
                    rotation_randomization["yaw_offset_degrees"],
                    rotation_randomization["roll_offset_degrees"],
                )
            )
        else:
            print(
                "  first capture uses the exact look-at rotation "
                "(rotation randomization disabled)."
            )

        actual_pose = set_camera_pose_verified(
            client,
            camera_id,
            requested_pose,
        )

        print(
            "  verified pose: location=({:.1f}, {:.1f}, {:.1f}), "
            "rotation=({:.1f}, {:.1f}, {:.1f})".format(*actual_pose)
        )

        # Request and discard one frame to flush any stale render-target contents
        # after a camera teleport.
        client.request("vget /camera/{}/lit png".format(camera_id))
        time.sleep(0.10)

        rgb_path = os.path.join(
            OUTPUT_DIRECTORY,
            "{}_{:04d}_rgb.png".format(
                output_actor_name,
                image_number,
            ),
        )
        mask_path = os.path.join(
            OUTPUT_DIRECTORY,
            "{}_{:04d}_mask.png".format(
                output_actor_name,
                image_number,
            ),
        )

        rgb_response = client.request(
            "vget /camera/{}/lit png".format(camera_id)
        )
        mask_response = client.request(
            "vget /camera/{}/object_mask png".format(camera_id)
        )

        rgb_saved = save_png_response(rgb_response, rgb_path)
        mask_saved = save_png_response(mask_response, mask_path)

        mask_filter_report: Dict[str, object] = {
            "enabled": ENABLE_SMALL_FAR_MASK_FILTER,
            "status": "not_run",
        }

        if mask_saved and ENABLE_SMALL_FAR_MASK_FILTER:
            try:
                mask_filter_report = filter_small_far_actor_masks(
                    client=client,
                    mask_path=mask_path,
                    camera_pose=actual_pose,
                    actors=actors,
                    target_actor=target_actor,
                    color_mapping=color_mapping,
                )
                mask_filter_report["status"] = "completed"

                removed_actors = mask_filter_report.get("removed_actors", [])
                print(
                    "  mask filter: removed {} small far-background actor(s): {}".format(
                        len(removed_actors),
                        removed_actors if removed_actors else "none",
                    )
                )
            except Exception as exc:
                # Do not destroy an otherwise valid capture if post-processing fails.
                # Leave the original UnrealCV mask intact and record the failure.
                mask_filter_report = {
                    "enabled": True,
                    "status": "failed",
                    "error": str(exc),
                }
                print(
                    "  WARNING: small/far mask filtering failed; "
                    "original mask was kept: {}".format(exc)
                )

        status_text = "saved" if rgb_saved and mask_saved else "failed"
        if status_text == "saved":
            successful_pairs += 1

        print(
            "  {}: {} and {}".format(
                status_text.upper(),
                rgb_path,
                mask_path,
            )
        )

        camera_records.append(
            {
                "image_number": image_number,
                "status": status_text,
                "camera_id": camera_id,
                "camera_name": (
                    camera_names[camera_id]
                    if camera_id < len(camera_names)
                    else None
                ),
                "object_search_term": OBJECT_ID_SEARCH_TERM,
                "matching_actor_ids": actors,
                "target_actor": target_actor,
                "output_actor_name": output_actor_name,
                "target_location": {
                    "x": target_location[0],
                    "y": target_location[1],
                    "z": target_location[2],
                },
                "sample": {
                    "azimuth_degrees": azimuth,
                    "elevation_degrees": elevation,
                    "distance_cm": distance,
                    "distance_preset": SELECTED_DISTANCE_PRESET,
                    "distance_preset_label": DISTANCE_PRESETS[
                        SELECTED_DISTANCE_PRESET
                    ]["label"],
                    "distance_preset_minimum_cm": DISTANCE_PRESETS[
                        SELECTED_DISTANCE_PRESET
                    ]["minimum_cm"],
                    "distance_preset_maximum_cm": DISTANCE_PRESETS[
                        SELECTED_DISTANCE_PRESET
                    ]["maximum_cm"],
                    "start_distance_cm": START_CAMERA_DISTANCE,
                    "distance_variation_fraction": DISTANCE_VARIATION_FRACTION,
                    "randomized_capture_minimum_cm": MIN_CAMERA_DISTANCE,
                    "randomized_capture_maximum_cm": MAX_CAMERA_DISTANCE,
                },
                "rotation_randomization": rotation_randomization,
                "requested_pose": {
                    "x": requested_pose[0],
                    "y": requested_pose[1],
                    "z": requested_pose[2],
                    "pitch": requested_pose[3],
                    "yaw": requested_pose[4],
                    "roll": requested_pose[5],
                },
                "verified_pose": {
                    "x": actual_pose[0],
                    "y": actual_pose[1],
                    "z": actual_pose[2],
                    "pitch": actual_pose[3],
                    "yaw": actual_pose[4],
                    "roll": actual_pose[5],
                },
                "rgb_path": rgb_path,
                "mask_path": mask_path,
                "mask_filter": mask_filter_report,
            }
        )

    poses_path = os.path.join(OUTPUT_DIRECTORY, "camera_poses.json")
    with open(poses_path, "w") as file:
        json.dump(camera_records, file, indent=4)

    print(
        "\nFinished: {} of {} RGB/mask pairs saved in '{}'.".format(
            successful_pairs,
            image_count,
            OUTPUT_DIRECTORY,
        )
    )
    print(
        'UnrealCV search term "{}" matched {} actor ID(s); '
        "camera target was {}.".format(
            OBJECT_ID_SEARCH_TERM,
            len(actors),
            target_actor,
        )
    )
    print("Verified camera metadata saved to '{}'.".format(poses_path))


if __name__ == "__main__":
    main()