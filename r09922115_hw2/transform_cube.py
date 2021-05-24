import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import pandas as pd
import os
from copy import deepcopy


def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines = o3d.utility.Vector2iVector(
        [[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # R, G, B
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate(
        [scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def update_cube():
    global cube, cube_vertices, R_euleexitr, t, scale

    transform_mat = get_transform_mat(R_euler, t, scale)

    transform_vertices = (transform_mat @ np.concatenate([
        cube_vertices.transpose(),
        np.ones([1, cube_vertices.shape[0]])
    ], axis=0)).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1:  # key down
        shift_pressed = True
    elif action == 0:  # key up
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()


def draw_camera(vis, data):
    p = data['position']
    r = data['rotation']
    model = o3d.geometry.LineSet()
    model.points = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
    model.lines = o3d.utility.Vector2iVector(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
    color = np.array([1, 0, 0])
    model.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
    model.scale(0.05, np.zeros(3))
    model.rotate(r)
    model.translate(p)
    vis.add_geometry(model)
    return model

# if len(sys.argv) != 2:
#     print('[Usage] python3 transform_cube.py /PATH/TO/points3D.txt')
#     sys.exit(1)


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

position_df = pd.read_pickle(
    "camera_position.pkl")
print(len(position_df))
print("start drawing...")

for i in range(len(position_df)):
    tmp = position_df.iloc[i]
    draw_camera(vis, tmp)

print("finish drawing...")

# load point cloud
points3D_df = pd.read_pickle(
    "data/points3D.pkl")
pcd = load_point_cloud(points3D_df)
vis.add_geometry(pcd)

# load axes
axes = load_axes()
vis.add_geometry(axes)

# load cube
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()
vis.add_geometry(cube)

R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale = 1.0
update_cube()

# just set a proper initial camera view
vc = vis.get_view_control()
vc_cam = vc.convert_to_pinhole_camera_parameters()
initial_cam = get_transform_mat(
    np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
initial_cam[-1, -1] = 1.
setattr(vc_cam, 'extrinsic', initial_cam)
vc.convert_from_pinhole_camera_parameters(vc_cam)

# set key callback
shift_pressed = False
vis.register_key_action_callback(340, toggle_key_shift)
vis.register_key_action_callback(344, toggle_key_shift)
vis.register_key_callback(ord('A'), update_tx)
vis.register_key_callback(ord('S'), update_ty)
vis.register_key_callback(ord('D'), update_tz)
vis.register_key_callback(ord('Z'), update_rx)
vis.register_key_callback(ord('X'), update_ry)
vis.register_key_callback(ord('C'), update_rz)
vis.register_key_callback(ord('V'), update_scale)

print('[Keyboard usage]')
print('Translate along X-axis\tA / Shift+A')
print('Translate along Y-axis\tS / Shift+S')
print('Translate along Z-axis\tD / Shift+D')
print('Rotate    along X-axis\tZ / Shift+Z')
print('Rotate    along Y-axis\tX / Shift+X')
print('Rotate    along Z-axis\tC / Shift+C')
print('Scale                 \tV / Shift+V')

vis.run()
vis.destroy_window()

np.save('cube_transform_mat.npy',
        get_transform_mat(R_euler, t, scale))
np.save('cube_vertices.npy',
        np.asarray(cube.vertices))


'''
print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
print('Translation vector:\n{}'.format(t))
print('Scale factor: {}'.format(scale))
'''


def make_points(cube_vertices, index, color):
    ratio = 12
    points = []
    o = cube_vertices[index[0]]
    v1 = cube_vertices[index[1]]-cube_vertices[index[0]]
    v2 = cube_vertices[index[2]]-cube_vertices[index[0]]
    for i in range(ratio):
        for j in range(ratio):
            point = o+(i/ratio)*v1+(j/ratio)*v2
            point = point.tolist()
            point.append(color)
            points.append(point)

    return points


cube_vertices = np.load(
    "cube_vertices.npy")
# print(cube_vertices)
face_index = [[0, 1, 2, 3], [0, 2, 4, 6], [4, 0, 5, 1],
              [7, 3, 5, 1], [6, 4, 7, 5], [2, 3, 6, 7]]
points_pool = []
for i, index in enumerate(face_index):
    points_pool.extend(make_points(cube_vertices, index, i))

# print(points_pool)


def draw_one_point(img, point, c):
    p = tuple([int(i) for i in point])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    cv.circle(img, p, 5, colors[c], -1)

    return img


def draw_points(rimg, transform_matrix, position, points_pool):
    position = np.array(position)
    ordered_points = sorted(points_pool, key=lambda point: np.linalg.norm(
        np.array(point[:3])-position, 2), reverse=True)

    for point in ordered_points:
        color = point[3]
        p = deepcopy(point)
        p[3] = 1
        pos = transform_matrix@np.array(p)
        pos /= pos[2]
        # print(pos)
        rimg = draw_one_point(rimg, pos[:2], color)

    return rimg

images_df = pd.read_pickle("data/images.pkl")

for i,index in enumerate(range(164,293)): #293
    # Load quaery image
    position=position_df.iloc[i]['position']
    idx = index
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    rimg = cv.imread("data/frames/"+fname)

    ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values

    r_matrix=R.from_quat(rotq_gt).as_matrix().reshape(3,3)
    t_matrix = tvec_gt.reshape(3,1)

    R_T=np.concatenate((r_matrix, t_matrix), axis=1)

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    transform_matrix=cameraMatrix@R_T

    rimg=draw_points(rimg,transform_matrix,position,points_pool)
    cv.imwrite("new_images/"+str(index)+".jpg",rimg)

    print("new_images/"+str(index)+".jpg")
    print("saved")
