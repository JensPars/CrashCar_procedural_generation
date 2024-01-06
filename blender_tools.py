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
    bpy.data.materials["damage_texture"].node_tree.nodes["RGB.001"].outputs[0].default_value = (r, g, b, 1)
    bpy.data.materials["damage_texture.001"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (r, g, b, 1)



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


def jitter_camera(dist=1, theta=0):
    cam = bpy.data.objects['Camera']
    cam.rotation_mode = 'XYZ'
    dist = np.random.uniform(0.1,1.5)
    theta = theta + np.random.uniform(0,2*math.pi)
    cam.location[0] = cos(theta)*dist
    cam.location[1] = sin(theta)*dist
    cam.location[2] = np.random.uniform(-0.05,0.3)


import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view




def BVHTreeAndVerticesInWorldFromObj( obj ):
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld @ v.co for v in obj.data.vertices]
    vs = [v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in obj.data.polygons] )

    return bvh, vertsInWorld, vs

def DeselectEdgesAndPolygons( obj ):
    for p in obj.data.polygons:
        p.select = False
    for e in obj.data.edges:
        e.select = False
    for e in obj.data.vertices:
        e.select = False

def vis_verts_ray(obj):
    scene = bpy.context.scene
    vis_ls = []
    cam = bpy.data.objects['Camera']
    DeselectEdgesAndPolygons(obj)
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices, vs = BVHTreeAndVerticesInWorldFromObj(obj)
    limit = 0.0001
    for i, v in enumerate( vertices ):
        # Get the 2D projection of the vertex
        co2D = world_to_camera_view( scene, cam, v )
        # By default, deselect it
        obj.data.vertices[i].select = False
        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            # Try a ray cast, in order to test the vertex visibility from the camera
            location, normal, index, distance = bvh.ray_cast( cam.location, (v - cam.location).normalized() )
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            if location and (v - location).length < limit and distance < 0.3:
                obj.data.vertices[i].select = True
                vis_ls.append(vs[i])
    del bvh
    return vis_ls



def vis_verts_ray(obj):
    scene = bpy.context.scene
    vis_ls = []
    cam = bpy.data.objects['Camera']
    DeselectEdgesAndPolygons(obj)
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices, vs = BVHTreeAndVerticesInWorldFromObj(obj)
    limit = 0.0001
    for i, v in enumerate( vertices ):
        # Get the 2D projection of the vertex
        co2D = world_to_camera_view( scene, cam, v )
        # By default, deselect it
        obj.data.vertices[i].select = False
        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            # Try a ray cast, in order to test the vertex visibility from the camera
            location, normal, index, distance = bvh.ray_cast( cam.location, (v - cam.location).normalized() )
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            if location and (v - location).length < limit and distance < 0.3:
                obj.data.vertices[i].select = True
                vis_ls.append(vs[i])
    del bvh
    return vis_ls




def vis_verts(obj):
    scene = bpy.context.scene
    vis_ls = []
    cam = bpy.data.objects['Camera']
    DeselectEdgesAndPolygons(obj)
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices, vs = BVHTreeAndVerticesInWorldFromObj(obj)
    limit = 0.5
    for i, v in enumerate( vertices ):
        co2D = world_to_camera_view( scene, cam, v )
        # By default, deselect it
        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            # Try a ray cast, in order to test the vertex visibility from the camera
            #print((v - cam.location).length)
            if (v - cam.location).length < limit:
                obj.data.vertices[i].select = True
                vis_ls.append(vs[i])
    del bvh
    return vis_ls




import random
def randomize_node(node, group):
    c = 0
    for n,g in zip(node, group):
        if not n.is_linked:
            val = random.uniform(g.min_value, g.max_value)
            n.default_value = val
        #if c == 5:
        #    print(val)
        #c += 1



def set_node_group_values(material, node_group):
    # Run through all its inputs
    for i, node_input in enumerate(material.get(node_group).inputs):
        # If the parameter is
        if not node_input.is_linked:
            # Get the min/max values from node_group properties
            rnd_min = bpy.data.node_groups[node_group].inputs[i].min_value
            rnd_max = bpy.data.node_groups[node_group].inputs[i].max_value
            
            # Set the random value
            if isinstance(node_input.default_value, float):
                node_input.default_value = random.uniform(rnd_min, rnd_max)
            else:
                for j in range(len(node_input.default_value)): # Added for vectors
                    node_input.default_value[j] = random.uniform(rnd_min, rnd_max)
