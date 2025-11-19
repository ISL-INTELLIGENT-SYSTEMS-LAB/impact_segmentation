from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
import trimesh

def build_asset(obj_path, texture_path, out_path, model_name="Asset", scale=1.0):
    stage = Usd.Stage.CreateNew(out_path)

    # Z-up axis
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)

    # Root prim
    root_path = f"/{model_name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)

    # Scale transform
    xform = UsdGeom.Xform(root)
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale, scale, scale)) #scale

    # Geometry reference
    geom_path = f"{root_path}/Geom"
    geom_prim = stage.DefinePrim(geom_path)
    geom_prim.GetReferences().AddReference(obj_path)

    # Material setup
    looks_path = f"{root_path}/Looks"
    material_path = f"{looks_path}/Mat"
    stage.DefinePrim(looks_path)

    material = UsdShade.Material.Define(stage, material_path)

    tex_shader = UsdShade.Shader.Define(stage, f"{material_path}/UVTexture")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    tex_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("auto")
    tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/Primvar_st")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader_out)

    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex_shader_out = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).ConnectToSource(tex_shader_out)

    material_output = material.CreateSurfaceOutput(UsdShade.Tokens.universalRenderContext)
    pbr_out = pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material_output.ConnectToSource(pbr_out)

    UsdShade.MaterialBindingAPI(geom_prim).Bind(material)

    stage.Save()
    print(f"Wrote USD asset: {out_path}")

# convert .obj file to usd
def obj_to_usd(obj_path, usd_path):
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(root.GetPrim())

    # Minimal mesh prim
    mesh = UsdGeom.Mesh.Define(stage, "/Model/Mesh")

    #pip install trimesh pywavefront
    # Example (using trimesh):
    tm = trimesh.load(obj_path)

    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in tm.vertices])
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(len(f)) for f in tm.faces]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in tm.faces for i in f]))

    #UVs
    if tm.visual.uv is not None:
        uvs = [Gf.Vec2f(float(u[0]), float(u[1])) for u in tm.visual.uv]
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.varying).Set(uvs)

    stage.Save()
    print(f"Converted {obj_path} → {usd_path}")

# -------------------------------
# INPUTS HERE
# ------------------------------

if __name__ == "__main__":
    #MoonRock
    mobj_path = "A:/Project/LunarProject/_assets/OBJ/10017-15_SFM_Full-Resolution-Model_Coordinate-Unregistered.obj"
    mtexture_path = "A:/Project/LunarProject/_assets/moon_rock.tif"
    musd_mesh_path = "A:/Project/LunarProject/_assets/USD/10017-15.usda" #set outputname for usda file after converting
    mout_path = "A:/Project/LunarProject/usd_outputfiles/moon_rock.usda" #set outputname for usda to be used for referencing
    m2out_path = "A:/Project/LunarProject/usd_outputfiles/moon_rock2.usda" 
    #MoonRock1
    m1obj_path = "A:/Project/LunarProject/_assets/OBJ/MoonRock1.obj"
    m1texture_path = "A:/Project/LunarProject/_assets/MoonRock1.tif"
    m1usd_mesh_path = "A:/Project/LunarProject/_assets/USD/m1mesh.usda" 
    m1out_path = "A:/Project/LunarProject/usd_outputfiles/moon_rock1.usda"
    m3out_path = "A:/Project/LunarProject/usd_outputfiles/moon_rock3.usda"
    # Step 1: Convert OBJ → USD mesh
    obj_to_usd(mobj_path, musd_mesh_path)
    obj_to_usd(m1obj_path, m1usd_mesh_path)

    # Step 2: Build asset referencing the USD mesh
    build_asset(musd_mesh_path, mtexture_path, mout_path, model_name="MoonRock", scale=250.0)
    build_asset(m1usd_mesh_path, m1texture_path, m1out_path, model_name="MoonRock1", scale=250.0)
    build_asset(musd_mesh_path, mtexture_path, m2out_path, model_name="MoonRock2", scale=600.0)
    build_asset(m1usd_mesh_path, m1texture_path, m3out_path, model_name="MoonRock3", scale=600.0)