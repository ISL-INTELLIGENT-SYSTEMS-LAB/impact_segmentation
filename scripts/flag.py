import math
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf

# Stage Setup
file_path = "A:/Project/LunarProject/usd_outputfiles/flag.usda"
stage = Usd.Stage.CreateNew(file_path)


flag_width = 5.0
flag_height = 3.0
flag_cols = 20
flag_rows = 10
flag_texture_path = "../_assets/flag.jpg"

# Create Xform container
flag_assembly = UsdGeom.Xform.Define(stage, '/FlagAssembly')
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def create_simple_metal_material(stage, path, color=Gf.Vec3f(0.8,0.6,0.3), metallic=1.0, roughness=0.3):
    material = UsdShade.Material.Define(stage, path)
    surface_shader = UsdShade.Shader.Define(stage, f"{path}/SurfaceShader")
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    surface_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
    return material

def create_textured_material(stage, path, texture_file, roughness=0.65):
    material = UsdShade.Material.Define(stage, path)
    surface_shader = UsdShade.Shader.Define(stage, f"{path}/SurfaceShader")
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    
    tex_shader = UsdShade.Shader.Define(stage, f"{path}/TexShader")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
    tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    
    st_reader = UsdShade.Shader.Define(stage, f"{path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    
    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
    return material

# Create Pole
pole = UsdGeom.Cylinder.Define(stage, '/FlagAssembly/Pole')
pole.GetRadiusAttr().Set(0.1)
pole.GetHeightAttr().Set(12.0)
#UsdGeom.XformCommonAPI(pole).SetRotate(Gf.Vec3f(90, 0, 0))

# Create Flag Mesh
flag_mesh = UsdGeom.Mesh.Define(stage, "/FlagAssembly/FlagMesh")

flag_points = []
for i in range(flag_rows + 1):
    for j in range(flag_cols + 1):
        x = (j / flag_cols) * flag_width
        y = math.sin((j / flag_cols) * math.pi * 2) * 0.2  # waving
        z = (i / flag_rows) * -flag_height
        flag_points.append(Gf.Vec3f(x, y, z))
flag_mesh.CreatePointsAttr(flag_points)

counts = []
indices = []
for i in range(flag_rows):
    for j in range(flag_cols):
        i0 = i * (flag_cols + 1) + j
        i1 = i0 + 1
        i2 = i0 + (flag_cols + 1) + 1
        i3 = i0 + (flag_cols + 1)
        counts.append(4)
        indices += [i0, i1, i2, i3]

flag_mesh.CreateFaceVertexCountsAttr(counts)
flag_mesh.CreateFaceVertexIndicesAttr(indices)
flag_mesh.CreateDoubleSidedAttr(True)

# UVs
uvs = []
for i in range(flag_rows + 1):
    for j in range(flag_cols + 1):
        uvs.append(Gf.Vec2f(j/flag_cols, i/flag_rows))
expanded_uvs = [uvs[i] for i in indices]
primvars_api_flag = UsdGeom.PrimvarsAPI(flag_mesh)
st_primvar_flag = primvars_api_flag.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
st_primvar_flag.Set(expanded_uvs)

# Create Materials
pole_material = create_simple_metal_material(stage, "/FlagAssembly/PoleMaterial", color=Gf.Vec3f(0.8,0.6,0.3), metallic=1.0, roughness=0.3)
flag_material = create_textured_material(stage, "/FlagAssembly/FlagMaterial", flag_texture_path)

# Bind materials
UsdShade.MaterialBindingAPI.Apply(pole.GetPrim())
UsdShade.MaterialBindingAPI(pole.GetPrim()).Bind(pole_material)

UsdShade.MaterialBindingAPI.Apply(flag_mesh.GetPrim())
UsdShade.MaterialBindingAPI(flag_mesh.GetPrim()).Bind(flag_material)

# Transform Flag
UsdGeom.XformCommonAPI(flag_mesh).SetRotate(Gf.Vec3f(0,0,90))
UsdGeom.XformCommonAPI(flag_mesh).SetTranslate(Gf.Vec3d(0,0,-3))

# Scale assembly
UsdGeom.XformCommonAPI(flag_assembly).SetScale(Gf.Vec3f(10,10,10))

stage.SetDefaultPrim(flag_assembly.GetPrim())
stage.Save()
print("USD file created successfully:", file_path)
