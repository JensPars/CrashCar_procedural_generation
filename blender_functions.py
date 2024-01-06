import bpy
import mathutils
import random
import json
import glob
import numpy as np
from math import cos, sin, pi
from collections import defaultdict
from pathlib import Path


#from utils import *

import bpy
import mathutils
import random
import json
import glob
import numpy as np
from math import cos, sin, pi
import math


import bpy
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def move_camera(dist=2, theta=0, delta_theta=0):
    delta_theta = np.random.uniform(-delta_theta, delta_theta)
    #delta_theta = pi/4
    theta += delta_theta
    cam = bpy.data.objects['Camera']
    cam.rotation_mode = 'XYZ'
    cam.location[0] = cos(theta)*dist
    cam.location[1] = sin(theta)*dist
    cam.rotation_euler[-1] = pi/2 + theta
    return np.array([cos(theta)*dist, sin(theta)*dist])

def adjust_cam(bump_pos, cam_pos):
    w = cam_pos - bump_pos
    yaw_theta = angle_between(w, cam_pos)
    cam = bpy.data.objects['Camera']
    cam.rotation_euler[-1] += (yaw_theta)

def set_car_paint(c):
    r,g,b = c
    r,g,b = r/255,g/255,b/255
    paint = bpy.data.materials["damage_texture"].node_tree.nodes["Principled BSDF"]
    paint.inputs[0].default_value = (r, g, b, 1)


def set_hdri(path):
    C = bpy.context
    scn = C.scene

    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(path) # Relative path
    node_environment.location = -300,0

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
    
def point_at(target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)      
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)

    obj = bpy.data.objects['Camera']
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'),'Z') if obj.type=='CAMERA' else (('X', 'Z'),'Y') 
    #because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, rotator)

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

def link_to_collection(coll_name:str):       
    for ob in bpy.data.scenes['Scene'].collection.objects:
        if ob.type != "CAMERA":
            ob.users_collection[0].objects.unlink(ob)
            bpy.data.collections[coll_name].objects.link(ob)

def clear_objects():
    #for o in bpy.data.objects:
    #    if o.type != "CAMERA":
    #        o.select_set(True)
    #    else:
    #        o.select_set(False)
    #bpy.ops.object.delete()
    #bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)

def fix_materials():
    for ob in bpy.data.objects:
        if ob.type == "MESH" and ob.name != "joined":
            name = ob.name
            bpy.context.view_layer.objects.active = ob
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
            ob.active_material_index = 1
            if len(bpy.data.objects[name].material_slots) > 1:
                for _ in range(len(bpy.data.objects[name].material_slots)-1):
                    bpy.ops.object.material_slot_remove()

def fix_normals():
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name != "joined":
            try:
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                # go edit mode
                bpy.ops.object.mode_set(mode='EDIT')
                # select al faces
                bpy.ops.mesh.select_all(action='SELECT')
                # recalculate outside normals 
                bpy.ops.mesh.normals_make_consistent(inside=False)
                # go object mode again
                bpy.ops.object.editmode_toggle()
            except:
                print("failed")


class blender_car():
    def __init__(self, output_dir, mesh_names_file, car_name, root, imp, dir):
        self.output_dir = Path(output_dir)
        self.car_name = car_name
        self.root = Path(dir) / root
        self.car_n = 0
        self.imp  = imp
        self.bump_keys = ["back_bumper",
                        "front_bumper",
                        "back_left_door",
                        "back_right_door",
                        #"back_license_plate",
                        "front_right_door",
                        "front_left_door",
                        "left_frame",
                        "right_frame",
                        "trunk",
                        "roof",
                        "hood"]#,
                        #"front_license_plate"]
        self.window_keys = ["back_left_window",
                          "back_right_window",
                          "front_left_window",
                          "front_right_window",
                          "left_quarter_window",
                          "right_quarter_window",
                          "front_windshield",
                          "back_windshield"]

        # loading parts list from .txt
        mesh_names_file = Path(dir) / mesh_names_file
        with open(mesh_names_file) as f:
            data = f.read()
        self.parts = json.loads(data)
        #self.meshes = self.parts.values()
        self.meshes = []
        self.weights = []
        #[item for sublist in self.parts for item in sublist]
        for key in self.parts:
            if key in self.bump_keys:
                n = len(self.parts[key])
                self.meshes += [mesh for mesh in self.parts[key]]
                self.weights += [1/n for _ in range(n)]
        # loading vehicle-colors list from .json
        col_path = Path(dir) / "data/vehicle-colors.json"
        with open(col_path) as f:
            col = json.load(f)    
        self.col_ls = [[int(val) for val in el['RGB'].split(',')] for el in col] 
        # fetching paths of HDRIs
        self.hdri_ls  = glob.glob('data/HDRI/*.exr')
        #select GPU for rendering
        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'
        self.setup()

    def import_parts(self):
        path_dir = self.root / self.car_name / Path("models/*.obj")
        path_dir = str(path_dir)
        for file in glob.glob(path_dir):
            bpy.ops.import_scene.obj(filepath=file, use_split_groups=True)
        
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            fix_materials()
            fix_normals()
        
        # Joining all other parts for subsequent raycast
        #  - ensuring bump does not go through other car parts

    def join_obj(self):
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            parts =  [item for sublist in self.parts.values() for item in sublist]
            flag = True
            bpy.ops.object.select_all(action='DESELECT')
            for ob in bpy.data.objects:
                if ob.type == "MESH" and ob.name not in parts:
                #if ob.type == "MESH":
                    #ob.select_set(True)
                    bpy.context.view_layer.objects.active = ob
                    bpy.ops.node.new_geometry_nodes_modifier()
                    ob.modifiers['GeometryNodes'].node_group = bpy.data.node_groups['car_bump']
                    #ob.modifiers["GeometryNodes"]["Input_14"] = 0 

                    #if flag:
                        #bpy.context.view_layer.objects.active = ob
                        #ob.name = "joined"
                        #ob.data.name = "joined"
                        #flag = False
            #bpy.ops.object.join()
            #bpy.ops.node.new_geometry_nodes_modifier()
            #bpy.data.objects['joined'].modifiers['GeometryNodes'].node_group = bpy.data.node_groups['car_bump']
            #bpy.data.objects['joined'].modifiers["GeometryNodes"]["Input_14"] = 0 


    def setup(self, bump = True):
        clear_objects()
        if self.imp:
            self.import_parts()
            link_to_collection("car_2")
        self.join_obj()
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            for idx,key in enumerate(self.parts):
                for mesh in self.parts[key]:
                    part = bpy.data.objects[mesh]
                    # setting pass index for part segmentation
                    part.pass_index = idx + 1
                    #applying geonodes and setting up input/output
                    if bump and len(bpy.data.objects[mesh].modifiers) == 0 and key in self.bump_keys:
                            bpy.context.view_layer.objects.active = part
                            # add empty geonodes
                            bpy.ops.node.new_geometry_nodes_modifier()
                            part.modifiers['GeometryNodes'].node_group = bpy.data.node_groups['car_bump']
                            part.modifiers['GeometryNodes']['Output_11_attribute_name'] = 'dam_val'
                            part.modifiers['GeometryNodes']["Output_3_attribute_name"] = 'norm'
                            part.modifiers["GeometryNodes"]["Output_10_attribute_name"] = "mask"
                            #bpy.context.object.modifiers["GeometryNodes"]["Input_14"] = 1
                            for _ in range(len(part.material_slots)):
                                bpy.ops.object.material_slot_remove()
                            part.data.materials.append(bpy.data.materials["damage_texture"])
                            part.active_material_index = 1 
                    ## setting raycast target        
                    #bpy.data.node_groups["car_bump"].nodes["Object Info"].inputs[0].default_value = bpy.data.objects["joined"]
                    
                    if key in self.window_keys:
                        bpy.context.view_layer.objects.active = part
                        for _ in range(len(part.material_slots)):
                                bpy.ops.object.material_slot_remove()
                        part.data.materials.append(bpy.data.materials["glass"])
                        part.active_material_index = 2

        
    def randomize_damage(self):
        part = random.choices(self.meshes, self.weights, k=1)[0]
        print(part)
        # sampling random vertex as centre
        verts = [vert.co for vert in bpy.data.objects[part].data.vertices]
        mid = random.sample(verts,1)[0]
        print(mid)
        ## select type of damage
        #mid = np.array([mid[0],mid[2],mid[1]])
        #mid_shade = np.array([-mid[0],-mid[2],mid[1]])
        mid_shade = np.array([-mid[0],-mid[1],-mid[2]])
        self.randomize_scratch(mid_shade)
        self.randomize_crack(mid_shade)
        #print(mid)
        

        #mid = np.array([mid[0],-mid[2],mid[1]])
        cam = bpy.data.objects['Camera']
        #rot_axis = np.array([0,0,1])
        #theta = np.random.uniform(-pi/6,pi/6)
        #mid = np.dot(rotation_matrix(rot_axis, theta), mid)
        mid = np.array([mid[0],-mid[2],mid[1]])
        distance = np.random.uniform(0.4,1.5)
        displacement = mid * 0.2
        cam.location = 3*mid
        point_at(mid)

            
        ## code placing mid of selected damage
        ## place camera from mid of main damage
        ## code adding more damage with raycast check
        
        return

    def randomize_bump(self, shape_noise=True, vec_noise=True):
        part = random.choices(self.meshes, self.weights, k=1)[0]
        # sampling random vertex as centre
        verts = [vert.co for vert in bpy.data.objects[part].data.vertices]
        mid = random.sample(verts,1)[0]
        bump = bpy.data.node_groups["car_bump"]
        bump.nodes["bump_centre"].vector = mid
        bump.nodes["shape_noise"].boolean = shape_noise
        bump.nodes["Boolean"].boolean = vec_noise
        # randomize bump shape
        bump_area = bump.nodes["bump_area"]
        bump_area.outputs[0].default_value = np.random.uniform(0.02,0.08)
        shape_scale = bump.nodes["shape_scale"]
        shape_scale.outputs[0].default_value = np.random.uniform(4,12)
        shape_seed = bump.nodes["shape_seed"]
        shape_seed.outputs[0].default_value = np.random.uniform(-1000,1000)
        # randomize bump depth
        bump_depth = bump.nodes["depth"]
        bump_depth.outputs[0].default_value =  np.random.uniform(-0.025,-0.07)
        # randomize displacement vector 
        vec_shape = bump.nodes["vec_scale"]
        vec_shape.outputs[0].default_value =  np.random.uniform(20,60)
        vec_seed = bump.nodes["vec_seed"]
        vec_seed.outputs[0].default_value = np.random.uniform(-1000,1000)
        vec_smooth = bump.nodes["vec_smooth"]
        vec_smooth.outputs[0].default_value = np.random.uniform(0.3,0.6)
        # getting normal from nodes
        fetch = bpy.data.objects[part]\
                .evaluated_get(bpy.context.evaluated_depsgraph_get())\
                .data.attributes 
        normal = fetch['norm'].data[0].vector
        bpy.data.node_groups["car_bump"].nodes["Vector"].vector = normal

        # rearranging coordinates between car object and camera object (y,z) are flipped and y negated
        mid = np.array([mid[0],-mid[2],mid[1]])
        #normal = np.array([normal[0],-normal[2],normal[1]])
        cam = bpy.data.objects['Camera']
        cam.location = 3*mid
        point_at(mid)
        
        #return mid

    
    def randomize_scratch(self, mid):
        tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        tex_nodes["scratch_select"].outputs[0].default_value = 1
        tex_nodes["x_scr"].outputs[0].default_value = mid[0]
        tex_nodes["y_scr"].outputs[0].default_value = mid[1]
        tex_nodes["z_scr"].outputs[0].default_value = mid[2]
    
    def randomize_crack(self, mid):
        tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        tex_nodes["scratch_select"].outputs[0].default_value = 1
        tex_nodes["x_crk"].outputs[0].default_value = mid[0]
        tex_nodes["y_crk"].outputs[0].default_value = mid[1]
        tex_nodes["z_crk"].outputs[0].default_value = mid[2]


    #def randomize_crack(self):
    
    #def randomize_shatter(self):
    

    def setup_cam(self, normal, mid):
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = np.random.uniform(-pi/6,pi/6)
        normal = np.dot(rotation_matrix(rot_axis, theta), normal)
        distance = np.random.uniform(0.4,1.5)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        if cam.location[-1] < 0:
            cam.location[-1] = np.random.uniform(-0.03,0.15)
        
        if cam.location[-1] > 0.3:
            cam.location[-1] = np.random.uniform(0.2,0.3)

    def render(self):
        # settiing file output path for damage segmentation and damage
        for out in ["image","damage"]:#["damage", "segmentation", "image"]:        
            bpy.data.scenes["Scene"]\
            .node_tree.nodes[out]\
            .base_path = str(self.output_dir / self.car_name / str(self.car_n))
        # render images and annos
        bpy.ops.render.render(write_still=True)
        self.car_n += 1
        print(f"Rendered {self.car_n}")
    
    def write_bump_degree(self):
        d = defaultdict(lambda:[])
        #d = {}
        for key in self.parts:
            if key in self.bump_keys:
                for part in self.parts[key]:
                    fetch = bpy.data.objects[part]\
                            .evaluated_get(bpy.context.evaluated_depsgraph_get())\
                            .data.attributes
                    #print(key, fetch['dam_val'].data[0].value)
                    d[key] += [fetch['dam_val'].data[0].value]
        
        with open(self.output_dir / self.car_name / str(self.car_n - 1) /'dam_val.json', 'w') as fp:
            json.dump(d, fp)

    def randomize_scene(self):
        # render n images of car
        c = random.sample(self.col_ls, 1)[0]
        set_car_paint(c)
        hdri = random.sample(self.hdri_ls, 1)[0]
        set_hdri(hdri)
        print(self.output_dir)
        #self.write_bump_degree()
        
