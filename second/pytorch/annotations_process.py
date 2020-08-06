import os
import sys
sys.path.insert(0, os.getcwd() + "/.")
import math
import numpy as np
import json
# import pcl
from second.pytorch.show_3dbox import mayavi_show_3dbox


def getFileData(dataFilePath):
    with open(dataFilePath, 'r') as file:
        for line in file:
            if line.strip():
                yield line.strip()
    return


def read_pcd_points(pcd_path):
    pcl_points = pcl.load_XYZI(pcd_path)
    points = []
    for point in pcl_points:
        x = point[0]
        y = point[1]
        z = point[2]
        if abs(x) < 1e-1 and abs(y) < 1e-1 and abs(z) < 1e-1:
            continue
        points.append([point[0], point[1], point[2], point[3]])
    numpy_points = np.array(points)
    return numpy_points


def read_bin_points(bin_path):
    points = np.fromfile(
        str(bin_path), dtype=np.float32,
        count=-1).reshape([-1, 4])
    return points


def read_annotations_data(annotation_path):
    my_file = open(annotation_path, encoding='utf-8')
    result = json.load(my_file)
    object_list = result['objects']['rect3DObject']
    box_names = []
    box_locs = []
    for box_value in object_list:
        if box_value['class'].strip() != 'DontCare':
            yaw = -box_value['yaw']  # inverse clockwise
            box = [box_value['centerX'],
                   box_value['centerY'],
                   box_value['centerZ'],
                   box_value['width'],
                   box_value['length'],
                   box_value['height'],
                   yaw]
            if (box[0] >= -1.5) and (box[0] <= 1.5) and \
                    (box[1] >= -2.5) and (box[1] <= 2.5):
                continue
            if (box[0] >= -41) and (box[0] <= 41) and \
                    (box[1] >= -81) and (box[1] <= 41):
                box_names.append(box_value['class'].strip())
                box_locs.append(box)
    gt_boxes = np.array(box_locs).astype(np.float32)
    gt_names = np.array(box_names)
    return gt_boxes, gt_names


def get_image_and_label_list(train_path):
    result = []
    annotation_post = ".json"
    path, _ = os.path.split(train_path)
    pcd_dir = os.path.join(path, "../pcds")
    annotation_dir = os.path.join(path, "../Annotations")
    for filename_and_post in getFileData(train_path):
        filename, post = os.path.splitext(filename_and_post)
        annotation_filename = filename + annotation_post
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        pcd_path = os.path.join(pcd_dir, filename_and_post)
        # print(pcd_path)
        if os.path.exists(annotation_path) and \
                os.path.exists(pcd_path):
            result.append((annotation_path, pcd_path))
        else:
            print("%s or %s not exist" % (annotation_path, pcd_path))
    return result


def show_annotations(info_path):
    cloud_and_label_list = get_image_and_label_list(info_path)
    print("remain number of infos:", len(cloud_and_label_list))
    for annotation_path, pcd_path in cloud_and_label_list:
        print(pcd_path)
        points = read_bin_points(pcd_path)
        gt_boxes, gt_names = read_annotations_data(annotation_path)
        mayavi_show_3dbox(points, gt_boxes, gt_names)


if __name__ == '__main__':
    show_annotations("/home/lpj/github/data/my_point_cloud/ali_dataset/ImageSets/Pedestrian_train.txt")
