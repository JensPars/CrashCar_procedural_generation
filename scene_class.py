import glob
import json
import random
import math
import bpy
import blender_tools
import numpy as np
from pathlib import Path
from numpy.random import choice




def setup_cam(mid):
        mid = np.array([mid[0],-mid[2],mid[1]])
        normal = np.copy(mid)
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = np.random.uniform(-math.pi/6,math.pi/6)
        normal = np.dot(blender_tools.rotation_matrix(rot_axis, theta), mid)
        normal = normal/np.linalg.norm(normal)
        distance = np.random.triangular(0.15,0.2,0.6)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        blender_tools.point_at(mid)
        cam.rotation_euler[-1] += np.random.triangular(-0.1, 0, 0.1)
        cam.rotation_euler[0] += np.random.triangular(-0.1, 0, 0.1)


class Scene():
    def __init__(self, output_dir, mesh_names_file, root, imp, dir, car_name):
        self.output_dir = Path(output_dir)
        self.car = car(mesh_names_file, dir)
        self.root = Path(dir) / root
        self.car_n = 0
        self.imp  = imp
        self.car_name = car_name
        # fetching paths of HDRIs
        self.hdri_ls  = glob.glob('data/HDRI/*')
        #select GPU for rendering
        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'
        self.setup()

    def import_car(self):
        path_dir = self.root / self.car_name / Path("models/*.obj")
        path_dir = str(path_dir)
        for file in glob.glob(path_dir):
            bpy.ops.import_scene.obj(filepath=file, use_split_groups=True)
        
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            blender_tools.fix_materials()
            blender_tools.fix_normals()

    def setup(self, bump = True):
        blender_tools.clear_objects()
        if self.imp:
            self.import_car()
            blender_tools.link_to_collection("car_2")
        self.car.setup()

    def render(self):
        # settiing file output path for damage segmentation and damage
        for out in ["image", "dent", "scratch", "part", "material", "crack"]:#["damage", "segmentation", "image"]:        
            bpy.data.scenes["Scene"]\
            .node_tree.nodes[out]\
            .base_path = str(self.output_dir / self.car_name / str(self.car_n))
        # render images and annos
        bpy.ops.render.render(write_still=True)
        self.car_n += 1
        print(f"Rendered {self.car_n}")

    def randomize_scene(self):
        # render n images of car
        print(len(self.hdri_ls))
        hdri = random.sample(self.hdri_ls, 1)[0]
        blender_tools.set_hdri(hdri)

class car():
    def __init__(self, mesh_names_file, dir):
        self.body_obj = []
        self.body_part = body_part
        self.window_part = window_part
        self.light_part = light_part
        self.body_parts = {}
        self.scratch = []
        self.dent = []
        self.crack = []
        self.setup_cam = []
        self.car_parts = []
        self.window = []
        self.window_ob_d = {}
        self.light = []
        self.light_ob_d = {}
        self.objects = []

        self.bump_keys =["back_bumper",
                         "front_bumper",
                         "back_left_door",
                         "back_right_door",
                         "front_right_door",
                         "front_left_door",
                         "left_frame",
                         "right_frame",
                         "trunk",
                         "roof",
                         "hood"]

        self.window_keys = ["back_left_window",
                            "back_right_window",
                            "front_left_window",
                            "front_right_window",
                            "left_quarter_window",
                            "right_quarter_window",
                            "front_windshield",
                            "back_windshield"]

        self.light_keys = ["left_head_light",
                            "left_tail_light",
                            "right_head_light",
                            "right_tail_light"]
        # loading parts list from .txt
        mesh_names_file = Path(dir) / mesh_names_file
        with open(mesh_names_file) as f:
            data = f.read()
        self.parts = json.loads(data)

        self.window_names = []
        for key in self.window_keys:
            self.window_names += self.parts[key]
        
        self.light_names = []
        for key in self.light_keys:
            self.light_names += self.parts[key]
        
        self.wheel_names = self.parts["wheel"]
        self.damage = True
        # loading vehicle-colors list from .json
        col_path = Path(dir) / "data/vehicle-colors.json"
        with open(col_path) as f:
            col = json.load(f)
        self.col_ls = [[int(val) for val in el['RGB'].split(',')] for el in col] 
        
    def setup(self):
        window = bpy.context.window_manager.windows[0]
        self.join_obj()
        bpy.data.node_groups["car_bump"].nodes["raycat_target"].inputs[0].default_value = bpy.data.objects["joined"]
        with bpy.context.temp_override(window=window):
            for idx,key in enumerate(self.parts):
                if len(self.parts[key]) > 0:
                    body_part = self.body_part(self)
                    for mesh in self.parts[key]:
                        ob = bpy.data.objects[mesh]
                        if key not in self.wheel_names:
                            self.objects.append(ob)
                        # setting pass index for part segmentation
                        ob.pass_index = idx + 1
                        # applying geonodes and setting up input/output
                        if key in self.bump_keys:
                            body_part.setup(ob)
                        if key in self.window_keys:
                            window_part = self.window_part(self, ob, key)
                            window_part.setup(ob)
                            self.window.append(window_part)
                            self.window_ob_d[ob.name] = window_part.shatter
                        if key in self.light_keys:
                            light_part = self.light_part(ob, key)
                            light_part.setup(ob)
                            self.light.append(light_part)
                            self.light_ob_d[ob.name] = light_part.shatter

                    #if key in self.window_keys:
                     #   self.window_ob.append(window_part)
                    if key in self.bump_keys and self.damage:    
                        self.body_parts[key] = body_part
                        self.scratch.append(body_part.randomize_scratch)
                        self.crack.append(body_part.randomize_crack)
                        self.dent.append(body_part.randomize_dent)
                        #self.setup_cam.append(body_part.setup_cam)
                        self.body_obj.append(ob)
                    if key != "roof" and key in self.bump_keys:
                        self.car_parts.append(body_part.randomize_parts)

        
    def randomize_car(self):
        c = random.sample(self.col_ls, 1)[0]
        blender_tools.set_car_paint(c)
        random.sample(self.dent, 1)[0]()

    
    def randomize_damage(self):
        for window in self.window:
            window.reset()
        for light in self.light:
            light.reset()
        bpy.data.node_groups["car_bump"].nodes["bump_0"].outputs[0].default_value = 0
        # Apply main damage
        dmg = [ self.scratch,
                self.crack,
                self.dent,
                list(self.window_ob_d.values()),
                list(self.light_ob_d.values())]
        dmg_s = [self.scratch,
                self.crack]
        # sample random damage
        dmg_func = random.sample(random.sample(dmg, 1)[0],1)[0]
        dmg_func(mid=None)
        # Determine whether secondry data should be applied
        if random.random() > 0.5:
            # use raycast to determine visibility
            vis_verts = []
            for obj in self.objects:
                vis = blender_tools.vis_verts(obj)
                if len(vis) > 5:
                    vis_verts.append((obj, vis))
            if len(vis_verts) > 0:
                obj, vis = random.sample(vis_verts, 1)[0]
                mid = random.sample(vis, 1)[0]
                if obj.name in self.light_names:
                    self.light_ob_d[obj.name](mid)
                if obj.name in self.window_names:
                    self.window_ob_d[obj.name](mid)
                else:
                    dmg_func = random.sample(random.sample(dmg, 1)[0],1)[0]
                    dmg_func(mid=mid)
                    # keep adding damages with a 0.1 prob
                    while random.random() < 0.2:
                        if obj.name in self.light_names:
                            self.light_ob_d[obj.name](mid)
                        if obj.name in self.window_names:
                            self.window_ob_d[obj.name](mid)
                        else:
                            dmg_func = random.sample(random.sample(dmg, 1)[0],1)[0]
                            dmg_func(mid=mid)

    def join_obj(self):
        window = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=window):
            parts =  [item for sublist in self.parts.values() for item in sublist]
            flag = True
            bpy.ops.object.select_all(action='DESELECT')
            for ob in bpy.data.objects:
                if ob.type == "MESH" and ob.name not in parts:
                    #bpy.context.view_layer.objects.active = ob
                    ob.select_set(True)
                    if flag:
                        bpy.context.view_layer.objects.active = ob
                        ob.name = "joined"
                        ob.data.name = "joined"
                        flag = False
            bpy.ops.object.join()
            bpy.context.view_layer.objects.active.name = "joined"
            


    
    def randomize_part_seg(self):
        c = random.sample(self.col_ls, 1)[0]
        blender_tools.set_car_paint(c)
    
    def reset_dent(self):
        bpy.data.node_groups["car_bump"].nodes["bump_1"].outputs[0].default_value = 0
        bpy.data.node_groups["car_bump"].nodes["bump_2"].outputs[0].default_value = 0
        
        


class body_part():
    def __init__(self, car):
        self.verts = []
        self.tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        self.scratch = bpy.data.node_groups["scratches"]
        self.car = car

    def setup(self, ob):
        D = bpy.data
        bpy.context.view_layer.objects.active = ob
        bpy.ops.node.new_geometry_nodes_modifier()
        ob.modifiers['GeometryNodes'].node_group = bpy.data.node_groups['car_bump']
        ob.modifiers["GeometryNodes"]["Output_10_attribute_name"] = "mask"
        for _ in range(len(ob.material_slots)):
            bpy.ops.object.material_slot_remove()
        ob.data.materials.append(D.materials["damage_texture"])
        ob.active_material_index = 0
        self.verts += [vert.co for vert in ob.data.vertices]
        assert len(self.verts) > 0, "ob.name"

    def select_vert(self):
        return random.sample(self.verts,1)[0]

    def randomize_scratch(self, mid=None):
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)
        mid_shade = np.array([mid[0],-mid[2],mid[1]])
        scratch = bpy.data.node_groups["scratches"]
        scratch.nodes["x_scr_0"].outputs[0].default_value = mid[0]
        scratch.nodes["y_scr_0"].outputs[0].default_value = mid[1]
        scratch.nodes["z_scr_0"].outputs[0].default_value = mid[2]
        node = bpy.data.node_groups["scratches"].nodes["scratch"].inputs
        group = bpy.data.node_groups["scratch"].inputs
        blender_tools.randomize_node(group=group, node=node)

    def randomize_crack(self, mid=None):
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)
        crack = bpy.data.node_groups["cracks"]
        mid_shade = np.array([-mid[0],-mid[1],-mid[2]])
        tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        crack.nodes["x_crk"].outputs[0].default_value = mid[0]
        crack.nodes["y_crk"].outputs[0].default_value = mid[1]
        crack.nodes["z_crk"].outputs[0].default_value = mid[2]
        node = bpy.data.node_groups["cracks"].nodes["crack"].inputs
        group = bpy.data.node_groups["crack"].inputs
        blender_tools.randomize_node(group=group, node=node)

    def randomize_dent(self, mid=None,shape_noise=True, vec_noise=True):
        bpy.data.node_groups["car_bump"].nodes["bump_0"].outputs[0].default_value = 1
        # sampling random vertex as centre
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)
        bump = bpy.data.node_groups["car_bump"]
        bump.nodes["bump_centre_0"].vector = mid
        num_bumps = np.random.randint(3, size=1)[0]
        group = bpy.data.node_groups["dent"].inputs
        for i in range(3):
            node = bpy.data.node_groups["car_bump"].nodes[f"dent_{i}"].inputs
            blender_tools.randomize_node(group=group, node=node)

    def randomize_parts(self):
        mid = self.select_vert()
        self.setup_cam_parts(mid)

    def setup_cam(self, mid):
        #mid = self.select_vert()
        mid = np.array([mid[0],-mid[2],mid[1]])
        normal = np.copy(mid)
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = np.random.uniform(-math.pi/6,math.pi/6)
        normal = np.dot(blender_tools.rotation_matrix(rot_axis, theta), mid)
        normal = normal/np.linalg.norm(normal)
        distance = np.random.triangular(0.15,0.2,0.6)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        blender_tools.point_at(mid)
        cam.rotation_euler[-1] += np.random.triangular(-0.05, 0, 0.05)
        cam.rotation_euler[0] += np.random.triangular(-0.05, 0, 0.05)
    
    def setup_cam_parts(self, mid):
        mid = np.array([mid[0],-mid[2],mid[1]])
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        disp_vec = np.array([mid[0],mid[1],0])
        disp_vec = mid/np.linalg.norm(mid)
        distance = np.random.triangular(0.5,0.9,1.3)
        displacement = disp_vec * distance
        cam.location = mid + displacement
        cam.location[2] = np.random.uniform(-0.05,0.3)
        # making sure image is taken correctly
        blender_tools.point_at((0,0,0))
        cam.rotation_euler[-1] += np.random.triangular(-0.05, 0, 0.05)
        cam.rotation_euler[0] += np.random.triangular(-0.05, 0, 0.05)

class window_part():
    def __init__(self, car, ob, key):
        self.verts = []
        self.tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        self.scratch = bpy.data.node_groups["scratches"]
        self.car = car
        self.ob = ob
        self.key = key
    
    def setup(self, ob):
        D = bpy.data
        bpy.context.view_layer.objects.active = ob
        for _ in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove()
        ob.data.materials.append(bpy.data.materials["glass"])
        ob.active_material_index = 0
        self.verts += [vert.co for vert in ob.data.vertices]
        assert len(self.verts) > 0, "ob.name"
    
    def select_vert(self):
        return random.sample(self.verts,1)[0]

    def shatter(self, mid):
        print("shatter")
        self.ob.material_slots[0].material =  bpy.data.materials["shatter"]
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)
        bpy.data.materials["shatter"].node_tree.nodes["x"].outputs[0].default_value = -mid[0]
        bpy.data.materials["shatter"].node_tree.nodes["y"].outputs[0].default_value = -mid[1]
        bpy.data.materials["shatter"].node_tree.nodes["z"].outputs[0].default_value = -mid[2]

    def setup_cam(self, mid):
        #mid = self.select_vert()
        mid = np.array([mid[0],-mid[2],mid[1]])
        normal = np.copy(mid)
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = 0 #np.random.uniform(-math.pi/4,math.pi/4)
        normal = np.dot(blender_tools.rotation_matrix(rot_axis, theta), mid)
        normal = normal/np.linalg.norm(normal)
        normal = mid/np.linalg.norm(mid)
        distance = 0.15  #np.random.triangular(0.15,0.2,0.6)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        blender_tools.point_at(mid)
        #cam.rotation_euler[-1] += np.random.triangular(-0.1, 0, 0.1)
        #cam.rotation_euler[0] += np.random.triangular(-0.1, 0, 0.1)
    
    def reset(self):
        self.ob.material_slots[0].material = bpy.data.materials["glass"]


class window_part():
    def __init__(self, car, ob, key):
        self.verts = []
        self.tex_nodes = bpy.data.materials["damage_texture"].node_tree.nodes
        self.scratch = bpy.data.node_groups["scratches"]
        self.car = car
        self.ob = ob
        self.key = key
    
    def setup(self, ob):
        D = bpy.data
        bpy.context.view_layer.objects.active = ob
        for _ in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove()
        ob.data.materials.append(bpy.data.materials["glass"])
        ob.active_material_index = 0
        self.verts += [vert.co for vert in ob.data.vertices]
        assert len(self.verts) > 0, "ob.name"
    
    def select_vert(self):
        return random.sample(self.verts,1)[0]

    def shatter(self, mid=None):
        print("shatter")
        self.ob.material_slots[0].material =  bpy.data.materials["shatter"]
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)

        bpy.data.materials["shatter"].node_tree.nodes["x"].outputs[0].default_value = mid[0]
        bpy.data.materials["shatter"].node_tree.nodes["y"].outputs[0].default_value = mid[1]
        bpy.data.materials["shatter"].node_tree.nodes["z"].outputs[0].default_value = mid[2]

    def setup_cam(self, mid):
        #mid = self.select_vert()
        mid = np.array([mid[0],-mid[2],mid[1]])
        normal = np.copy(mid)
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = np.random.uniform(-math.pi/4,math.pi/4)
        normal = np.dot(blender_tools.rotation_matrix(rot_axis, theta), mid)
        normal = normal/np.linalg.norm(normal)
        distance = np.random.triangular(0.15,0.2,0.6)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        blender_tools.point_at(mid)
        cam.rotation_euler[-1] += np.random.triangular(-0.1, 0, 0.1)
        cam.rotation_euler[0] += np.random.triangular(-0.1, 0, 0.1)
    def reset(self):
        self.ob.material_slots[0].material = bpy.data.materials["glass"]

class light_part():
    def __init__(self, ob, key):
        self.verts = []
        self.ob = ob
        self.key = key
        if self.key in ["left_head_light", "right_head_light"]:
            self.dam_tex = "light_broken"
            self.tex = "head_light"
        else:
            self.dam_tex = "broken_tail_light"
            self.tex = "tail_light"
    
    def setup(self, ob):
        D = bpy.data
        bpy.context.view_layer.objects.active = ob
        for _ in range(len(ob.material_slots)):
                bpy.ops.object.material_slot_remove()
        ob.data.materials.append(bpy.data.materials[self.tex])
        ob.active_material_index = 0
        self.verts += [vert.co for vert in ob.data.vertices]
        assert len(self.verts) > 0, "ob.name"
    
    def select_vert(self):
        return random.sample(self.verts,1)[0]

    def shatter(self, mid=None):
        print("shatter_lamp")
        self.ob.material_slots[0].material =  bpy.data.materials[self.dam_tex]
        if mid == None:
            mid = self.select_vert()
            setup_cam(mid)
        node = bpy.data.materials["broken_tail_light"].node_tree.nodes["broken_tail_light"].inputs[2:]
        group = bpy.data.node_groups["light_broken"].inputs[2:]
        blender_tools.randomize_node(group=group, node=node)
        node = bpy.data.materials["light_broken"].node_tree.nodes["light_broken"].inputs[2:]
        blender_tools.randomize_node(group=group, node=node)
        bpy.data.materials[self.dam_tex].node_tree.nodes["x"].outputs[0].default_value = mid[0]
        bpy.data.materials[self.dam_tex].node_tree.nodes["y"].outputs[0].default_value = mid[1]
        bpy.data.materials[self.dam_tex].node_tree.nodes["z"].outputs[0].default_value = mid[2]

    def setup_cam(self, mid):
        #mid = self.select_vert()
        mid = np.array([mid[0],-mid[2],mid[1]])
        normal = np.copy(mid)
        cam = bpy.data.objects['Camera']
        rot_axis = np.array([0,0,1])
        theta = 0 #np.random.uniform(-math.pi/4,math.pi/4)
        normal = np.dot(blender_tools.rotation_matrix(rot_axis, theta), mid)
        normal = normal/np.linalg.norm(normal)
        distance = 0.15  #np.random.triangular(0.15,0.2,0.6)
        displacement = normal * distance
        cam.location = mid + displacement
        # making sure image is taken correctly
        blender_tools.point_at(mid)
        #cam.rotation_euler[-1] += np.random.triangular(-0.1, 0, 0.1)
        #cam.rotation_euler[0] += np.random.triangular(-0.1, 0, 0.1)
    
    def reset(self):
        self.ob.material_slots[0].material = bpy.data.materials[self.tex]
