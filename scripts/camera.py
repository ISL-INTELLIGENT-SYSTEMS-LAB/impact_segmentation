from pxr import Usd, UsdGeom, Gf

file_path = "A:/Project/LunarProject/usd_outputfiles/camera.usda"
stage = Usd.Stage.CreateNew(file_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

camera = UsdGeom.Camera.Define(stage, "/CameraApollo11")
camera.CreateHorizontalApertureAttr(36)  
camera.CreateVerticalApertureAttr(24)
camera.CreateFocalLengthAttr(12.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(1, 100000.0))

stage.SetDefaultPrim(camera.GetPrim())
stage.Save()
print("Camera USD created:", file_path)
