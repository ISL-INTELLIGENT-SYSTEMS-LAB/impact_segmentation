from pxr import UsdLux, Usd, UsdGeom, Gf

#Create the stage
stage = Usd.Stage.CreateNew("A:/Project/LunarProject/usd_outputfiles/SunStage.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
#Default prim
sunroot = UsdGeom.Xform.Define(stage, "/SunRig")
stage.SetDefaultPrim(sunroot.GetPrim())
sunroot.AddScaleOp().Set(Gf.Vec3f(1000, 1000, 1000)) #scale

#Create a light simulating the sun
print('Adjusting lights...')
sun = UsdLux.DistantLight.Define(stage, "/Sun/SunLight")
sun.GetIntensityAttr().Set(10000.0)
sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9)) #slighty warm white
sun.AddRotateXOp().Set(45)#elevation
sun.AddRotateZOp().Set(-30) #direction nwse

stage.GetRootLayer().Save()