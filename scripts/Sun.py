from pxr import UsdLux, Usd, UsdGeom, Gf

stage = Usd.Stage.CreateNew("A:/Project/LunarProject/usd_outputfiles/SunStage.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

sunroot = UsdGeom.Xform.Define(stage, "/SunRig")
stage.SetDefaultPrim(sunroot.GetPrim())

# === DEFAULT-STYLE VIEWER LIGHT ===
sun = UsdLux.DistantLight.Define(stage, "/SunRig/DefaultAngleSun")

# brightness similar to viewer
sun.GetIntensityAttr().Set(5000.0)
sun.CreateExposureAttr(0.0)

# color slightly warm (like viewer)
sun.GetColorAttr().Set(Gf.Vec3f(1.0, 1, 1))

# usdview default angle ≈ 45° elevation, -45° azimuth
sun.AddRotateXOp().Set(-45)   # tilt downward toward the ground
sun.AddRotateYOp().Set(0)
sun.AddRotateZOp().Set(-45)   # rotate around vertical axis

stage.GetRootLayer().Save()
