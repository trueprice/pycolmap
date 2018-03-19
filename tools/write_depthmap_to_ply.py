import sys
sys.path.append("..")

import imageio
import numpy as np
import os

from plyfile import PlyData, PlyElement
from pycolmap import SceneManager


#-------------------------------------------------------------------------------

def main(args):
    suffix = ".photometric.bin" if args.photometric else ".geometric.bin"

    image_file = os.path.join(args.dense_folder, "images", args.image_filename)
    depth_file = os.path.join(
        args.dense_folder, "stereo", "depth_maps", args.image_filename + suffix)
    normals_file = os.path.join(
        args.dense_folder, "stereo", "normal_maps",
        args.image_filename + suffix)

    # load camera intrinsics from the COLMAP reconstruction
    scene_manager = SceneManager(os.path.join(args.dense_folder, "sparse"))
    scene_manager.load_cameras()
    scene_manager.load_images()

    image_id, image = scene_manager.get_image_from_name(args.image_filename)
    camera = scene_manager.cameras[image.camera_id]

    # load image, depth map, and normal map
    image = imageio.imread(image_file)

    with open(depth_file, "rb") as fid:
        w = int("".join(iter(lambda: fid.read(1), "&")))
        h = int("".join(iter(lambda: fid.read(1), "&")))
        c = int("".join(iter(lambda: fid.read(1), "&")))
        depth_map = np.fromfile(fid, np.float32).reshape(h, w)

    with open(normals_file, "rb") as fid:
        w = int("".join(iter(lambda: fid.read(1), "&")))
        h = int("".join(iter(lambda: fid.read(1), "&")))
        c = int("".join(iter(lambda: fid.read(1), "&")))
        normals = np.fromfile(
            fid, np.float32).reshape(c, h, w).transpose([1, 2, 0])

    # create 3D points
    xmin = -camera.cx / camera.fx
    xmax = (camera.width - 1. - camera.cx) / camera.fx
    ymin = -camera.cy / camera.fy
    ymax = (camera.height - 1. - camera.cy) / camera.fy
    x, y = np.meshgrid(np.linspace(xmin, xmax, camera.width),
                       np.linspace(ymin, ymax, camera.height))
    points3D = np.dstack((x, y, np.ones_like(x)))
    points3D *= depth_map[:,:,np.newaxis]

    # save
    points3D = points3D.astype(np.float32).reshape(-1, 3)
    normals = normals.astype(np.float32).reshape(-1, 3)
    image = image.reshape(-1, 3)
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255.).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    vertices = np.rec.fromarrays(
        tuple(points3D.T) + tuple(normals.T) + tuple(image.T),
        names="x,y,z,nx,ny,nz,red,green,blue")
    vertices = PlyElement.describe(vertices, "vertex")
    PlyData([vertices]).write(args.output_filename)


#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dense_folder")
    parser.add_argument("image_filename")
    parser.add_argument("output_filename")

    parser.add_argument("--photometric", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
