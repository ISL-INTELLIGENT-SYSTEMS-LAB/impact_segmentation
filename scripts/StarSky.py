
# sky.py
from pxr import Usd, UsdGeom, UsdLux, Kind, Sdf, Gf
import os

# -------------------------------------------------
# Create Sky Stage
# -------------------------------------------------
filename = "A:/Project/LunarProject/usd_outputfiles/SkyStage.usda"
stage = Usd.Stage.CreateNew(filename)

# Make sure we stay consistent with your pipeline
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

# Root prim for the sky
sky_root = UsdGeom.Xform.Define(stage, "/Sky")
Usd.ModelAPI(sky_root).SetKind(Kind.Tokens.component)
stage.SetDefaultPrim(sky_root.GetPrim())

# -------------------------------------------------
# Dome light with star texture
# -------------------------------------------------
# Point this to your star/equirectangular texture
star_tex = "A:/Project/LunarProject/_assets/hiptyc_2020_4k.exr"  

star_dome = UsdLux.DomeLight.Define(stage, "/Sky/StarDome")

# Use a large dome (radius is mostly visual in some Hydra delegates, but safe to set)
star_dome.CreateIntensityAttr(1.0)  # 0 if you only want it as a background (no lighting)
# If you *want* a little ambient star light, bump this up slightly (e.g. 0.1–1.0)

# Set the texture as the dome environment
star_dome.CreateTextureFileAttr(star_tex)

# Optional: rotate dome so orientation matches your taste
# This rotates the environment map around Z
star_dome.AddRotateZOp().Set(0.0)  # tweak degrees if you want to spin the stars

# Save
stage.GetRootLayer().Save()
print(f"Saved sky stage as {os.path.abspath(filename)}")
