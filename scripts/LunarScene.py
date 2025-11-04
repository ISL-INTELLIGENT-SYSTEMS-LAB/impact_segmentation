
from pxr import Usd, UsdGeom, Gf
import os, math
#Variables from moon.py, make sure they are the same
moon_scale = 1000
moon_radius = 17.374 * moon_scale

#placing Objects Function
def place_object_on_moon(prim, moon_radius, lat_deg, lon_deg, offset= 0.0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    #convert to x,y,z (z-up)
    x = (moon_radius + offset) * math.cos(lat) * math.cos(lon)
    y = (moon_radius + offset) * math.cos(lat) * math.sin(lon)
    z = (moon_radius + offset) * math.sin(lat)

    translation = Gf.Vec3d(x, y, z)
    normal = Gf.Vec3d(x, y, z).GetNormalized()
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

# Reference Sun
print('Importing Sun...')
sun_root = UsdGeom.Xform.Define(master_stage, "/Sun")
sun_root.GetPrim().GetReferences().AddReference("SunStage.usda")

#Reference Flag
print('Loading Flags...')
# List of flags to place
flags = [
    {"name": "Apollo11", "lat": 0.67416,  "lon": 23.47314},
    {"name": "Apollo12", "lat": -3.0128,  "lon": -23.4219},
    {"name": "Apollo14", "lat": -3.64417, "lon": -17.47865},
]
for flag_info in flags:
    prim_path = f"/FlagAssembly_{flag_info['name']}"
    flag_ref = UsdGeom.Xform.Define(master_stage, prim_path)
    flag_ref.GetPrim().GetReferences().AddReference("flag.usda")
    place_object_on_moon(flag_ref, moon_radius,
                         flag_info["lat"], flag_info["lon"], offset=0.5)


#Reference Camera
print("Importing Camera...")
camera_root = master_stage.DefinePrim("/CameraApollo11", "Camera")
camera_root.GetReferences().AddReference("camera.usda")

# Apollo 14 coordinates
lat = 0.67416 + .9
lon = 23.47314 + 1

# Place the camera slightly above the surface (offset upward)
place_object_on_moon(camera_root, moon_radius, lat, lon, offset=25.0)
UsdGeom.XformCommonAPI(camera_root).SetRotate(Gf.Vec3f(110, -90, 180))


'''camera_root = UsdGeom.Xform.Define(master_stage, "/ApolloCamera")
camera_root.GetPrim().GetReferences().AddReference("ApolloCamera.usda")'''


master_stage.GetRootLayer().Save()
