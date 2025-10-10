from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Tf

def create_wall_mesh(stage, room_name, name, width, height, thickness, translation, rotation=None, tile_u=1, tile_v=1):

    wall = UsdGeom.Mesh.Define(stage, f"{room_name}/{name}Mesh")

    # Define 4 corner points of the wall rectangle

    if "wall" in name.lower():
      wall.CreatePointsAttr([
        Gf.Vec3f(-width, -height, 0.0),
        Gf.Vec3f(width, -height, 0.0),
        Gf.Vec3f(width, height, 0.0),
        Gf.Vec3f(-width, height, 0.0),
      ])
    else:
      wall.CreatePointsAttr([
        Gf.Vec3f(-width, 0.0, -thickness),
        Gf.Vec3f(width, 0.0, -thickness),
        Gf.Vec3f(width, 0.0, thickness),
        Gf.Vec3f(-width, 0.0, thickness),
      ])
    # surfaces have 4 faces) 
    wall.CreateFaceVertexCountsAttr([4])
    #define the normal face 
    wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    # Apply transform
    wall.AddTranslateOp().Set(translation)
    if rotation:
        wall.AddRotateXYZOp().Set(rotation)

    # Add UVs
    primvars_api = UsdGeom.PrimvarsAPI(wall.GetPrim())
    uvs = [
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(tile_u, 0.0),
        Gf.Vec2f(tile_u, tile_v),
        Gf.Vec2f(0.0, tile_v),
    ]
    primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    primvar.Set(uvs)

    return wall

def create_shader_textures(stage, room_name, name, image):

  material = UsdShade.Material.Define(stage, f"{room_name}/{name.capitalize()}Material")
  surface_shader = UsdShade.Shader.Define(stage, f"{room_name}/{name.capitalize()}Material/SurfaceShader")
  surface_shader.CreateIdAttr("UsdPreviewSurface")
  
  # Create a texture shader
  tex_shader = UsdShade.Shader.Define(stage, f"{room_name}/{name.capitalize()}Material/TexShader")
  tex_shader.CreateIdAttr("UsdUVTexture")
  tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image)  # Replace with your texture path
  tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
  tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
  tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)

  # Create a primvar reader for UVs
  st_reader = UsdShade.Shader.Define(stage, f"{room_name}/{name.capitalize()}Material/STReader")
  st_reader.CreateIdAttr("UsdPrimvarReader_float2")
  st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
  st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
  
  # Connect UVs to texture shader
  tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(),"result")

  # Connect texture to surface shader
  surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")

  # Connect surface shader to material
  material.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
  return material

# constants
USDA_FILENAME = r"usd_outputfiles\room_with_textured_floor2.usda"
WALL_X = 35.0
WALL_Y = 25.0
WALL_HEIGHT = 10.0
WALL_THICKNESS = 0.1
WALL_TEXTURE_IMAGE = "../_assets/wall-texture-with-white-spots.jpg"
#FLOOR_TEXTURE_IMAGE = "../_assets/AM2512v42-UVCheckerGrid-UnwrapDebugging-TextureMap-2K.png"
FLOOR_TEXTURE_IMAGE = "../_assets/sandcarpet.jpg"
ROOM_NAME = "/Room"


# Create a new USD stage
stage = Usd.Stage.CreateNew(USDA_FILENAME)

# Create a root Xform
room = UsdGeom.Xform.Define(stage, ROOM_NAME)

# define rotations
wall_rotation_left = Gf.Vec3f(0, 90, 0)
wall_rotation_right = Gf.Vec3f(0, 270, 0)
wall_rotation_front = Gf.Vec3f(0, 180, 0)
floor_rotation = Gf.Vec3f(180, 0, 0)

#define translations
wall_back_trans = Gf.Vec3f(0, WALL_HEIGHT , -WALL_Y)
wall_front_trans = Gf.Vec3f(0, WALL_HEIGHT , WALL_Y)
wall_left_trans = Gf.Vec3f(-WALL_X, WALL_HEIGHT , 0)
wall_right_trans = Gf.Vec3f(WALL_X, WALL_HEIGHT , 0)
floor_trans = Gf.Vec3f(0, WALL_THICKNESS , 0)

# Create four walls
create_wall_mesh(stage, ROOM_NAME, "Wall_Back", WALL_X, WALL_HEIGHT, WALL_THICKNESS, wall_back_trans)
create_wall_mesh(stage, ROOM_NAME, "Wall_Front", WALL_X, WALL_HEIGHT, WALL_THICKNESS, wall_front_trans, wall_rotation_front)
create_wall_mesh(stage, ROOM_NAME, "Wall_Left", WALL_Y, WALL_HEIGHT, WALL_THICKNESS, wall_left_trans, wall_rotation_left)
create_wall_mesh(stage, ROOM_NAME, "Wall_Right", WALL_Y, WALL_HEIGHT, WALL_THICKNESS, wall_right_trans, wall_rotation_right)

# Create floor
create_wall_mesh(stage, ROOM_NAME, "Floor", WALL_X, WALL_THICKNESS, WALL_Y, floor_trans, floor_rotation)

# Create material with a texture
floor_material = create_shader_textures(stage, ROOM_NAME, "Floor", FLOOR_TEXTURE_IMAGE)
wall_material = create_shader_textures(stage, ROOM_NAME, "Wall", WALL_TEXTURE_IMAGE)


# Bind material to surface
for primname in ["Wall_Back", "Wall_Front", "Wall_Left", "Wall_Right", "Floor"]:
    prim = stage.GetPrimAtPath(f"{ROOM_NAME}/{primname}Mesh")
    UsdShade.MaterialBindingAPI.Apply(prim)
    if "wall".capitalize() in primname:
      UsdShade.MaterialBindingAPI(prim).Bind(wall_material)
    else:
      UsdShade.MaterialBindingAPI(prim).Bind(floor_material)


# Save the stage
stage.GetRootLayer().Save()
print(f"USD room created: {USDA_FILENAME}")
