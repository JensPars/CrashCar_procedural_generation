import sys
import os
import bpy
import time
import json
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
from glob import glob
from pathlib import Path
from scene_class import Scene



def main(device, resolution, n_imgs, part_type, damage_type):
    # set device used for rendering
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = device
    if device=="CUDA" or device=="METAL":
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    # set resolution
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution


    cars = Path("data") / Path("accepted") / Path("*.txt")
    cars = str(cars)
    cars = glob(cars)
    def f(x): return x.split(".")[0].split("/")[-1]
    cars = [f(el) for el in cars]
    cars = sorted(cars)
    root = "data/shapenet/"
    output_dir = "dataset/"
    bpy.context.scene.view_layers["scratch_anno"].use = True
    bpy.context.scene.view_layers["crack_anno"].use = True
    bpy.context.scene.view_layers["dent_anno"].use = True
    for car in cars:
        mesh_names = Path("data/accepted") / (car + ".txt") 
        dir = os.path.dirname(bpy.data.filepath)
        if not dir in sys.path:
            sys.path.append(dir)
        try:
            scene = Scene(output_dir, mesh_names, root, imp = True, dir=dir, car_name=car)
            for _ in range(n_imgs):
                scene.randomize_scene()
                scene.car.randomize_car()
                scene.car.randomize_damage()
                scene.render()
                scene.car.reset_dent()
            print("Completed Rendering")
        except:
            pass


if __name__ == "__main__":
        # Read the contents of the config.json file
    with open('config.json', 'r') as file:
        config_data = json.load(file)
    # Access the data from the config.json file
    print(config_data)
    # randomize and render
    main(**config_data)