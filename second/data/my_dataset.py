import os
from pathlib import Path
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.utils.eval import get_my_eval_result
from second.data.dataset import Dataset, register_dataset, get_dataset_class
from second.utils.progress_bar import progress_bar_iter as prog_bar
import json
# import pcl


@register_dataset
class MyDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None):
        self._root_path = Path(root_path)
        self.annotation_post = ".json"
        self.cloud_and_label_list = self.get_image_and_label_list(info_path)

        print("remain number of infos:", len(self.cloud_and_label_list))
        self._class_names = class_names
        self._prep_func = prep_func

    def __len__(self):
        return len(self.cloud_and_label_list)

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        annos = []
        for i in range(len(detection)):
            det = detection[i]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 100, 100]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def convert_gt_to_kitti_annos(self):
        annos = []
        for i in range(len(self.cloud_and_label_list)):
            annotation_path, _ = self.cloud_and_label_list[i]
            gt_boxes, gt_names = self.read_annotations_data(annotation_path)
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = gt_boxes
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 100, 100]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])

                anno["name"].append(gt_names[j])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(0)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = {'image_idx': i}
        return annos

    def evaluation(self, detections, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """

        gt_annos = self.convert_gt_to_kitti_annos()
        dt_annos = self.convert_detection_to_kitti_annos(detections)
        # firstly convert standard detection to kitti-format dt annos
        # KITTI camera format use y as regular "z" axis.
        # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        z_axis = 2
        z_center = 0.5
        result_official_dict = get_my_eval_result(
            gt_annos,
            dt_annos,
            self._class_names,
            z_axis=z_axis,
            z_center=z_center)

        return {
            "results": {
                "official": result_official_dict["result"]
            },
            "detail": {
                "eval.kitti": {
                    "official": result_official_dict["detail"]
                }
            },
            "mAp": result_official_dict["mAp"]
        }

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
        annotation_path, pcd_path = self.cloud_and_label_list[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_idx": idx
            }
        }
        points = self.read_bin_points(pcd_path)
        # points = self.read_pcd_points(pcd_path)
        gt_boxes, gt_names = self.read_annotations_data(annotation_path)
        res["lidar"]["points"] = points
        res["lidar"]["annotations"] = {'boxes': gt_boxes,
                                       'names': gt_names}
        return res

    def read_pcd_points(self, pcd_path):
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

    def read_bin_points(self, bin_path):
        points = np.fromfile(
            str(bin_path), dtype=np.float32,
            count=-1).reshape([-1, self.NumPointFeatures])
        return points

    def read_annotations_data(self, annotation_path):
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

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        pcd_dir = os.path.join(path, "../pcds")
        annotation_dir = os.path.join(path, "../Annotations")
        for filename_and_post in self.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            annotation_filename = filename + self.annotation_post
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            pcd_path = os.path.join(pcd_dir, filename_and_post)
            # print(pcd_path)
            if os.path.exists(annotation_path) and \
                    os.path.exists(pcd_path):
                result.append((annotation_path, pcd_path))
            else:
                print("%s or %s not exist" % (annotation_path, pcd_path))
        return result

    def getFileData(self, dataFilePath):
        with open(dataFilePath, 'r') as file:
            for line in file:
                if line.strip():
                    yield line.strip()
        return


def create_my_groundtruth_database(dataset_class_name,
                                   data_path,
                                   info_path=None,
                                   used_classes=None,
                                   database_save_path=None,
                                   db_info_save_path=None,
                                   relative_path=True):
    dataset = get_dataset_class(dataset_class_name)(
        info_path=info_path,
        root_path=data_path,
    )
    root_path = Path(data_path)
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = Path(database_save_path)
    if db_info_save_path is None:
        db_info_save_path = root_path / "my_dbinfos_train.pkl"
    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    group_counter = 0
    for j in prog_bar(list(range(len(dataset)))):
        image_idx = j
        sensor_data = dataset.get_sensor_data(j)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]
        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        group_dict = {}
        group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)

        num_obj = gt_boxes.shape[0]
        point_indices = None
        if num_obj > 0:
            point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            filename = "{}_{}_{}.bin".format(image_idx,
                                             names[i],
                                             i)
            filepath = database_save_path / filename
            gt_points = points[point_indices[:, i]]
            if gt_points.shape[0] >= 5:
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
                if (used_classes is None) or names[i] in used_classes:
                    if relative_path:
                        db_path = str(database_save_path.stem + "/" + filename)
                    else:
                        db_path = str(filepath)
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "image_idx": image_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                    }
                    local_group_id = group_ids[i]
                    if local_group_id not in group_dict:
                        group_dict[local_group_id] = group_counter
                        group_counter += 1
                    db_info["group_id"] = group_dict[local_group_id]
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print("load {} {} database infos".format(len(v), k))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
