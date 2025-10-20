
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdLux
import numpy as np
import imageio.v3 as iio
from PIL import Image
import os

# Create the stage
filename = 'LunarPlane2.usda'
stage = Usd.Stage.CreateNew(filename)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

# Create parent prim
planeroot = UsdGeom.Xform.Define(stage, "/LunarPlane")
planeroot.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 100.0)) #scale
Usd.ModelAPI(planeroot).SetKind(Kind.Tokens.component)

# Define variables and loading height map
color_map = "LunarProject/textures/lroc_color_poles_4k.tif"
height_map = "LunarProject/textures/ldem_16.tif"
print("Loading heightmap...")
hmap = iio.imread(height_map).astype(np.float32) #read image into a NumPy array
hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())  # normalize 0-1(so higher res heights maps still work)
aspect_ratio = hmap.shape[1]/ hmap.shape[0]
width = 860   
height = width / aspect_ratio   
subdiv_x = 5760
subdiv_y = 2880
height_scale = 10 #increase or decrease z

# Sample grid
xs = np.linspace(-width/2, width/2, subdiv_x) #creates evenly spaced coordinates
ys = np.linspace(-height/2, height/2, subdiv_y)
xv, yv = np.meshgrid(xs, ys) #vertex poistions

# Map texture coordinates to image pixels
hx = np.linspace(0, hmap.shape[1]-1, subdiv_x) 
hy = np.linspace(0, hmap.shape[0]-1, subdiv_y)
X, Y = np.meshgrid(hx, hy)
zv = hmap[Y.astype(int), X.astype(int)] * height_scale
print(zv)
# Flatten points
points = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3) #2D array of points for USD

# Faces 4 vertices per face
faces = []
for j in range(subdiv_y - 1):
    for i in range(subdiv_x - 1):
        v0 = j * subdiv_x + i
        v1 = v0 + 1
        v2 = v0 + subdiv_x + 1
        v3 = v0 + subdiv_x
        faces.append([v0, v1, v2, v3])

# Create mesh
print('Generating Mesh...')
planemesh = UsdGeom.Mesh.Define(stage, "/LunarPlane/PlaneMesh")
planemesh.CreatePointsAttr(points.tolist())
planemesh.CreateFaceVertexCountsAttr([4] * len(faces)) #all faces are 4 vertices
planemesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f]) #flatten array

# Subdivision for displacement
planemesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)

# Extent for viewport
extent = [(-width/2, -height/2, float(zv.min())),
          ( width/2,  height/2, float(zv.max()))]
planemesh.CreateExtentAttr(extent)

# UVs
u = np.linspace(0, 1, subdiv_x)
v = np.linspace(0, 1, subdiv_y)
uu, vv = np.meshgrid(u, v)
uvs = np.stack([uu, 1 - vv], axis=-1).reshape(-1, 2)

texCoords = UsdGeom.PrimvarsAPI(planemesh).CreatePrimvar(
    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
)
texCoords.Set(uvs.tolist())

# Make a material
print('Loading Materials...')
material = UsdShade.Material.Define(stage, '/LunarPlane/lunarcolor')
ColorShader = UsdShade.Shader.Define(stage, '/LunarPlane/lunarcolor/colorshader')
ColorShader.CreateIdAttr("UsdPreviewSurface")
ColorShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
ColorShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
material.CreateSurfaceOutput().ConnectToSource(ColorShader.ConnectableAPI(), "surface")

# Diffuse texture
stReader = UsdShade.Shader.Define(stage, '/LunarPlane/lunarcolor/stReader')
stReader.CreateIdAttr('UsdPrimvarReader_float2')

diffuseTextureSampler = UsdShade.Shader.Define(stage, '/LunarPlane/lunarcolor/diffuseTexture')
diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(color_map)
diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
ColorShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

# Primvar connection
stInput = material.CreateInput('frame:stLunarPlane', Sdf.ValueTypeNames.Token)
stInput.Set('st')
stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

# Bind material
UsdShade.MaterialBindingAPI(planemesh).Bind(material)

#Create a light simulating the sun
print('Adjusting lights...')
sun = UsdLux.DistantLight.Define(stage, "/LunarPlane/SunLight")
sun.GetIntensityAttr().Set(5000.0)
sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9)) #slighty warm white
sun.AddRotateXOp().Set(-45)#elevation
sun.AddRotateZOp().Set(30) #direction nwse

# Save stage
stage.GetRootLayer().Save()
path = os.path.abspath(filename)
print(f'Saved as {filename} at {path}')
