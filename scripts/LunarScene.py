
from pxr import Usd, UsdGeom, Gf
import os, math, imageio.v3 as iio, numpy as np, random

# Load height map & reuse parameters
height_map = "A:/Project/LunarProject/_assets/ldem_hw5x3.tif"
hmap = iio.imread(height_map).astype(np.float32)
height_scale = 0.0026
radius = 17.374
scale = 1000.0

def sample_height(lat_deg, lon_deg):
    # Map lat/lon → texel (mirror what you did in moon.py)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    u = (lon + np.pi) / (2 * np.pi)   # 0..1
    v = (lat + np.pi/2) / np.pi       # 0..1

    x = int(u * (hmap.shape[1] - 1))
    y = int(v * (hmap.shape[0] - 1))

    return hmap[y, x] * height_scale  # in km
#placing Objects Function
def place_object_on_moon(prim, lat_deg, lon_deg, clearance_km=0):
    h = sample_height(lat_deg, lon_deg)
    R = radius + h + clearance_km  # in km

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    #convert to x,y,z (z-up)
    world_R = R * scale  # km → world units
    x = world_R * math.cos(lat) * math.cos(lon)
    y = world_R * math.cos(lat) * math.sin(lon)
    z = world_R * math.sin(lat)


    translation = Gf.Vec3d(float(x), float(y), float(z))
    normal = Gf.Vec3d(float(x), float(y), float(z)).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    axis = up ^ normal
    axis_len = axis.GetLength()   # cross product
    if axis_len < 1e-6:
        euler = Gf.Vec3f(0, 0, 0)
    else:
        axis.Normalize()
        dot = max(-1.0, min(1.0, up * normal))
        angle = math.acos(dot)

        # Build rotation matrix and extract Euler XYZ
        r = Gf.Rotation(axis, math.degrees(angle))
        m = Gf.Matrix3d(r)
        euler = Gf.Vec3f(
            math.degrees(math.atan2(m[2][1], m[2][2])),
            math.degrees(math.atan2(-m[2][0], math.sqrt(m[2][1]**2 + m[2][2]**2))),
            math.degrees(math.atan2(m[1][0], m[0][0]))
        )

        #Apply the Transform
        xform_api = UsdGeom.XformCommonAPI(prim)
        xform_api.SetRotate(euler, UsdGeom.XformCommonAPI.RotationOrderXYZ)
        xform_api.SetTranslate(translation)

#Create the Master Stage
print('Creating the Stage...')
master_stage = Usd.Stage.CreateNew("A:/Project/LunarProject/usd_outputfiles/LunarScene.usda")
UsdGeom.SetStageUpAxis(master_stage, UsdGeom.Tokens.z)

# Reference Moon
print('Importing moon...')
moon_root = UsdGeom.Xform.Define(master_stage, "/Moon")
moon_root.GetPrim().GetReferences().AddReference("MoonStage.usda")

# Reference Sun and background
print("Loading Background...")
sky_root = UsdGeom.Xform.Define(master_stage, "/Sky")
sky_root.GetPrim().GetReferences().AddReference("SkyStage.usda")
print('Importing Sun...')
sun_root = UsdGeom.Xform.Define(master_stage, "/Sun")
sun_root.GetPrim().GetReferences().AddReference("SunStage.usda")

#Reference Flag
print('Loading Flags...')
# List of flags to place
flags = [
    {"name": "Apollo11", "lat": 0.67416,  "lon": 28.47314},
    {"name": "Apollo12", "lat": -3.0128,  "lon": -23.4219},
    {"name": "Apollo14", "lat": -3.64417, "lon": -17.47865},
]
for flag_info in flags:
    prim_path = f"/FlagAssembly_{flag_info['name']}"
    flag_ref = UsdGeom.Xform.Define(master_stage, prim_path)
    flag_ref.GetPrim().GetReferences().AddReference("flag.usda")
    place_object_on_moon(flag_ref, flag_info["lat"], flag_info["lon"], clearance_km=0.05)
#rotate 11flag for better view
apollo11_flag = UsdGeom.XformCommonAPI(master_stage.GetPrimAtPath("/FlagAssembly_Apollo11"))
apollo11_flag.SetRotate(Gf.Vec3f(88.3079, -10.51773, -64.97482), UsdGeom.XformCommonAPI.RotationOrderXYZ)


#Reference Camera
print("Importing Camera...")
camera_root = master_stage.DefinePrim("/CameraApollo11", "Camera")
camera_root.GetReferences().AddReference("camera.usda")

# Apollo 14 coordinates
lat = -0.67416 + .9
lon = 23.47314 + 1

# Place the camera slightly above the surface (offset upward)
place_object_on_moon(camera_root, lat, lon, clearance_km=0.00)
UsdGeom.XformCommonAPI(camera_root).SetRotate(Gf.Vec3f(110, -90, 180))

#Place the rock
print('Importing Rocks...')

# Define parameters
num_rocks = 200
rlat_min, rlat_max = -1.62584, 7.67416
rlon_min, rlon_max = 31.47314, 36.47314

# Loop to create and place multiple rocks
for i in range(num_rocks):
    rock_name = f"/MoonRock_{i+1}"
    rock_root = UsdGeom.Xform.Define(master_stage, rock_name)
    rock_root.GetPrim().GetReferences().AddReference("moon_rock.usda")
    
    # Random latitude and longitude within the given ranges
    rlat = random.uniform(rlat_min, rlat_max)
    rlon = random.uniform(rlon_min, rlon_max)

    # Compute rock height (per-instance)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(rock_root.GetPrim())
    extent = bbox.ComputeAlignedRange()
    rock_height = extent.GetSize()[2]  # z-up world
    offset = (-rock_height * 0.5)
    
    # Place the rock on the moon
    place_object_on_moon(rock_root, rlat, rlon)

    # Add random rotation around the surface normal (z-up)
    rand_angle = random.uniform(0, 360)  # degrees
    xform_api = UsdGeom.XformCommonAPI(rock_root)
    xform_api.SetRotate(Gf.Vec3f(0, 0, rand_angle), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print(f"Placed {rock_name} at lat={rlat:.5f}, lon={rlon:.5f}")
# Moon Rock-1
mrnum_rocks = 200
mrrlat_min, mrrlat_max = 1.67416, 7.67416
mrrlon_min, mrrlon_max = 20, 25

# Loop to create and place multiple rocks
for i in range(num_rocks):
    rock_name = f"/MoonRock_1_{i+1}"
    rock_root = UsdGeom.Xform.Define(master_stage, rock_name)
    rock_root.GetPrim().GetReferences().AddReference("moon_rock.usda")
    
    # Random latitude and longitude within the given ranges
    rlat = random.uniform(mrrlat_min, mrrlat_max)
    rlon = random.uniform(mrrlon_min, mrrlon_max)

    # Compute rock height (per-instance)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(rock_root.GetPrim())
    extent = bbox.ComputeAlignedRange()
    rock_height = extent.GetSize()[2]  # z-up world
    offset = (-rock_height * 0.5)
    
    # Place the rock on the moon
    place_object_on_moon(rock_root, rlat, rlon)

    # Add random rotation around the surface normal (z-up)
    rand_angle = random.uniform(0, 360)  # degrees
    xform_api = UsdGeom.XformCommonAPI(rock_root)
    xform_api.SetRotate(Gf.Vec3f(0, 0, rand_angle), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print(f"Placed {rock_name} at lat={rlat:.5f}, lon={rlon:.5f}")
# Moon Rock 1
m1num_rocks = 50
m1rlat_min, m1rlat_max = -1.62584, 7.67416
m1rlon_min, m1rlon_max = 31.47314, 36.47314

# Loop to create and place multiple rocks
for i in range(num_rocks):
    rock_name = f"/MoonRock1_{i+1}"
    rock_root = UsdGeom.Xform.Define(master_stage, rock_name)
    rock_root.GetPrim().GetReferences().AddReference("moon_rock1.usda")
    
    # Random latitude and longitude within the given ranges
    rlat = random.uniform(m1rlat_min, m1rlat_max)
    rlon = random.uniform(m1rlon_min, m1rlon_max)

    # Compute rock height (per-instance)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(rock_root.GetPrim())
    extent = bbox.ComputeAlignedRange()
    rock_height = extent.GetSize()[2]  # z-up world
    offset = (-rock_height * 0.5)
    
    # Place the rock on the moon
    place_object_on_moon(rock_root, rlat, rlon)

    # Add random rotation around the surface normal (z-up)
    rand_angle = random.uniform(0, 360)  # degrees
    xform_api = UsdGeom.XformCommonAPI(rock_root)
    xform_api.SetRotate(Gf.Vec3f(0, 0, rand_angle), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print(f"Placed {rock_name} at lat={rlat:.5f}, lon={rlon:.5f}")
# Moon Rock 1-1
m11num_rocks = 50
m11rlat_min, m11rlat_max = 1.67416, 7.67416
m11rlon_min, m11rlon_max = 20, 25
# Loop to create and place multiple rocks
for i in range(num_rocks):
    rock_name = f"/MoonRock11_{i+1}"
    rock_root = UsdGeom.Xform.Define(master_stage, rock_name)
    rock_root.GetPrim().GetReferences().AddReference("moon_rock1.usda")
    
    # Random latitude and longitude within the given ranges
    rlat = random.uniform(m11rlat_min, m11rlat_max)
    rlon = random.uniform(m11rlon_min, m11rlon_max)

    # Compute rock height (per-instance)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(rock_root.GetPrim())
    extent = bbox.ComputeAlignedRange()
    rock_height = extent.GetSize()[2]  # z-up world
    offset = (-rock_height * 0.5)
    
    # Place the rock on the moon
    place_object_on_moon(rock_root, rlat, rlon)

    # Add random rotation around the surface normal (z-up)
    rand_angle = random.uniform(0, 360)  # degrees
    xform_api = UsdGeom.XformCommonAPI(rock_root)
    xform_api.SetRotate(Gf.Vec3f(0, 0, rand_angle), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    print(f"Placed {rock_name} at lat={rlat:.5f}, lon={rlon:.5f}")
#Import in a rover
# Create a parent prim for placement
rover_parent = UsdGeom.Xform.Define(master_stage, "/ApolloRover")

# Child prim that actually references the rover asset
rover_geom = UsdGeom.Xform.Define(master_stage, "/ApolloRover/Geom")
rover_geom.GetPrim().GetReferences().AddReference(
    "A:/Project/LunarProject/usd_outputfiles/Lunar_Rover.usda"
)
# 1) Place parent on the Moon (aligns +Z with surface normal)
rover_lat = -1.67416    # pick your lat
rover_lon = 25    # pick your lon near rocks/flag
place_object_on_moon(rover_parent, rover_lat, rover_lon, clearance_km=0.00)

# 2) Fix orientation in local space of the rover Z up
rover_fix = UsdGeom.XformCommonAPI(rover_geom)
rover_fix.SetRotate(Gf.Vec3f(-90, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

master_stage.GetRootLayer().Save()
