import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import math
import numpy as np
import scipy.linalg as linalg
from second.core import box_np_ops
import cv2
import mayavi.mlab
# import pcl
# import pcl.pcl_visualization


def get_rotation_matrix(yaw):
    axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis_z / linalg.norm(axis_z) * yaw))
    return rot_matrix


def get_rotation_matrix1(yaw):
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]
                    ])
    return R_z


def compute_3dbox_corners1(box):
    '''
    x0y0z0,
    x0y1z0,
    x1y1z0,
    x1y0z0,
    x0y0z1,
    x0y1z1,
    x1y1z1,
    x1y0z1
    where
    x0 < x1, y0 < y1, z0 < z1
    '''
    R = get_rotation_matrix1(box[6])

    # 3d bounding box dimensions
    w = box[3]
    l = box[4]
    h = box[5]

    # 3d bounding box corners, xyz(wlh) are in camera coordinates
    x_corners = [-w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, w/2]
    y_corners = [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    # rotate and translate 3d bounding box by yaw angle
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + box[0]  # all x coordinates for 8 bb_corners
    corners_3d[1, :] = corners_3d[1, :] + box[1]  # all y coordinates for 8 bb_corners
    corners_3d[2, :] = corners_3d[2, :] + box[2]  # all z coordinates for 8 bb_corners

    return corners_3d
    # return np.transpose(corners_3d)  # corners_3d^T, axis aligned


def compute_3dbox_corners(box):
    centers = box[:3]
    dims = box[3:6]
    angles = np.array([box[6]])
    corners_3d = box_np_ops.center_to_corner_box3d(centers.reshape([1, 3]),
                                                   dims.reshape([1, 3]), angles,
                                                   origin=(0.5, 0.5, 0.5), axis=2)
    result = np.squeeze(corners_3d)
    # print(result)
    # return result
    return np.transpose(result)


def pcd_show_detections(points, detections):
    cloud = pcl.PointCloud(points[:, :3])
    visualcolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 0, 255, 0)
    vs = pcl.pcl_visualization.PCLVisualizering
    vss1 = pcl.pcl_visualization.PCLVisualizering()
    vs.AddPointCloud_ColorHandler(vss1, cloud, visualcolor, id=b'cloud', viewport=0)
    boxes = detections['box3d_lidar'].detach().cpu().numpy()
    classes_index = detections['label_preds'].detach().cpu().numpy()
    scores = detections['scores'].detach().cpu().numpy()
    count = boxes.shape[0]

    # vs.remove_all_pointclouds()
    # vs.remove_all_shapes()

    for i in range(count):
        box = boxes[i]
        class_index = classes_index[i]
        score = scores[i]
        corners_3d = compute_3dbox_corners(box)
        box_id = "box_%d" % i
        vs.AddCube(vss1, corners_3d[0][0], corners_3d[7][0], corners_3d[0][1],
                   corners_3d[7][1], corners_3d[0][2], corners_3d[7][2], 255, 0, 0, box_id)
        txt_id = "txt_%d" % i
        txt = "%d %f" % (class_index, score)
        vs.add_text3D(vss1, txt, box[:3], 2, 255, 0, 0, id=txt_id, viewport=0)
    while not vs.WasStopped(vss1):
        vs.Spin(vss1)


def plot3Dbox(corner, color=(0.23, 0.6, 1)):
    idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
    x = corner[0, idx]
    y = corner[1, idx]
    z = corner[2, idx]
    mayavi.mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral',
                       representation='wireframe', line_width=1)


def show_roi_box(range=(-40, -80, -3, 40, 40, 1)):
    center_x = (range[0] + range[3]) / 2.0
    center_y = (range[1] + range[4]) / 2.0
    center_z = (range[2] + range[5]) / 2.0
    width = range[3] - range[0]
    length = range[4] - range[1]
    height = range[5] - range[2]
    roi_box = np.array([center_x, center_y, center_z, width, length, height, 0])
    corners_3d = compute_3dbox_corners1(roi_box)
    plot3Dbox(corners_3d, (0, 0, 0))


def mayavi_show_3dbox(points, boxes, gt_names):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    # r = lidar[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(x, y, z,
                         d,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig)

    show_roi_box()

    count = boxes.shape[0]
    for i in range(count):
        box = boxes[i]
        mayavi.mlab.text3d(box[0], box[1], box[2], "%s" % (gt_names[i]),
                           color=(1, 0, 0), scale=(1, 1, 1))
        corners_3d = compute_3dbox_corners(box)
        plot3Dbox(corners_3d)
    mayavi.mlab.show()
    mayavi.mlab.close(all=True)


def mayavi_show_detection(points, detections):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    # r = lidar[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(x, y, z,
                         d,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig)

    show_roi_box()

    boxes = detections['box3d_lidar'].detach().cpu().numpy()
    classes_index = detections['label_preds'].detach().cpu().numpy()
    scores = detections['scores'].detach().cpu().numpy()
    count = boxes.shape[0]
    for i in range(count):
        score = scores[i]
        # print(score)
        if score >= 0.3:
            box = boxes[i]
            class_index = classes_index[i]
            mayavi.mlab.text3d(box[0], box[1], box[2], "%d %.2f %.3f" % (class_index, box[1], score),
                               color=(1, 0, 0), scale=(1, 1, 1))
            corners_3d = compute_3dbox_corners(box)
            plot3Dbox(corners_3d)
    mayavi.mlab.show()
    mayavi.mlab.close(all=True)


