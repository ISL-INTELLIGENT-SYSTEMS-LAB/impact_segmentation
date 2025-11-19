
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdLux
import numpy as np
import imageio.v3 as iio
from PIL import Image
import os

####################################
# Set absolute file paths
####################################

# Create the stage
filename = 'A:/Project/LunarProject/usd_outputfiles/MoonStage.usda'
stage = Usd.Stage.CreateNew(filename)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z) #set z = height defaults to y

# Create parent prim
planeroot = UsdGeom.Xform.Define(stage, "/Moon")
planeroot.AddScaleOp().Set(Gf.Vec3f(1000, 1000, 1000)) #scale
Usd.ModelAPI(planeroot).SetKind(Kind.Tokens.component)
stage.SetDefaultPrim(planeroot.GetPrim()) #set default prim for easy imports

#####################################################################
# Set absoultue texture paths, Usdview use .jpg or .png textures
#####################################################################

# Define variables and loading height map
color_map = "A:/Project/LunarProject/_assets/lroc_color_poles_4k.tif" #usdview use .jpg/.png
height_map = "A:/Project/LunarProject/_assets/ldem_hw5x3.tif" #keep .tif for heightmap

#####################################################################
# Below this point for tweaking moon.py
######################################################################
print("Loading heightmap...")
hmap = iio.imread(height_map).astype(np.float32) #read image into a NumPy array
#hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())  # normalize 0-1(may remove)

# Sphere parameters
radius = 17.374  # Moon mean radius in km (scale as needed)
height_scale = .0026 #For displacement height
subdiv_lat = int(5760) # latitude divisions
subdiv_lon = int(2880)  # longitude divisions
planeroot.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0)) #z=radius to get moon above mesh centered for easy translation

# Latitude/Longitude grids
lats = np.linspace(-np.pi/2, np.pi/2, subdiv_lat)
lons = np.linspace(-np.pi, np.pi, subdiv_lon)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Map texture coordinates to heightmap
hx = np.linspace(0, hmap.shape[1]-1, subdiv_lon)
hy = np.linspace(0, hmap.shape[0]-1, subdiv_lat)
X, Y = np.meshgrid(hx, hy)
heights = hmap[Y.astype(int), X.astype(int)] * height_scale

# Convert to 3D sphere coordinates
R = radius + heights
x = R * np.cos(lat_grid) * np.cos(lon_grid)
y = R * np.cos(lat_grid) * np.sin(lon_grid)
z = R * np.sin(lat_grid)

points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

# Faces
faces = []
for j in range(subdiv_lat - 1):
    for i in range(subdiv_lon - 1):
        v0 = j * subdiv_lon + i
        v1 = v0 + 1
        v2 = v0 + subdiv_lon + 1
        v3 = v0 + subdiv_lon
        faces.append([v0, v1, v2, v3])

print('Generating Sphere Mesh...')
spheremesh = UsdGeom.Mesh.Define(stage, "/Moon/SphereMesh")
spheremesh.CreatePointsAttr(points.tolist())
spheremesh.CreateFaceVertexCountsAttr([4] * len(faces))
spheremesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
spheremesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)

# Extent for viewport
extent = [
    (-radius, -radius, -radius),
    (radius, radius, radius)
]
spheremesh.CreateExtentAttr(extent)

# Build per-corner UVs to match faceVertexIndices order
u = np.linspace(0.0, 1.0, subdiv_lon)
v = np.linspace(0.0, 1.0, subdiv_lat)

uvs_per_corner = []
for j in range(subdiv_lat - 1):
    for i in range(subdiv_lon - 1):
        # Quad corners (v0, v1, v2, v3) must match faces list
        uvs_per_corner.extend([
            Gf.Vec2f(float(u[i]),     float(1.0 - v[j]    )),  # v0
            Gf.Vec2f(float(u[i+1]),   float(1.0 - v[j]    )),  # v1
            Gf.Vec2f(float(u[i+1]),   float(1.0 - v[j+1]  )),  # v2
            Gf.Vec2f(float(u[i]),     float(1.0 - v[j+1]  )),  # v3
        ])
texCoords = UsdGeom.PrimvarsAPI(spheremesh).CreatePrimvar(
    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
texCoords.Set(uvs_per_corner)


# Make a material
print('Loading Materials...')
material = UsdShade.Material.Define(stage, '/Moon/lunarcolor')
ColorShader = UsdShade.Shader.Define(stage, '/Moon/lunarcolor/colorshader')
ColorShader.CreateIdAttr("UsdPreviewSurface")
ColorShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
ColorShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
#increase sharpness and realism
ColorShader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
ColorShader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)

material.CreateSurfaceOutput().ConnectToSource(ColorShader.ConnectableAPI(), "surface")

# Diffuse texture
stReader = UsdShade.Shader.Define(stage, '/Moon/lunarcolor/stReader')
stReader.CreateIdAttr('UsdPrimvarReader_float2')

diffuseTextureSampler = UsdShade.Shader.Define(stage, '/Moon/lunarcolor/diffuseTexture')
diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(color_map)
diffuseTextureSampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
#Mipmapping and filtering for more detailed close range rendering 
diffuseTextureSampler.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
diffuseTextureSampler.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
diffuseTextureSampler.CreateInput('filter', Sdf.ValueTypeNames.Token).Set('linear')  # or 'linear' for smoother transitions
diffuseTextureSampler.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB') #Test...
ColorShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')


# Primvar connection
stInput = material.CreateInput('frame:stLunarPlane', Sdf.ValueTypeNames.Token)
stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')  

# Bind material
UsdShade.MaterialBindingAPI(spheremesh).Bind(material)

# Save stage
stage.GetRootLayer().Save()
path = os.path.abspath(filename)
print(f'Saved as {filename} at {path}')

