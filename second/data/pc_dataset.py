import os
import glob
import time
import numpy as np
from second.data.dataset import Dataset
from second.core import box_np_ops
# import pcl


class PointCloudDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self,
                 data_dir,
                 voxel_generator,
                 anchor_cache,
                 max_voxels):
        self.data_dir = data_dir
        self.voxel_generator = voxel_generator
        self.anchor_cache = anchor_cache
        self.max_voxels = max_voxels
        self.anchor_area_threshold = 1
        self.pc_files = self.get_dir_files(data_dir, '*.bin')

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc_file = self.pc_files[idx]
        points = self.read_bin_points(pc_file)
        metrics = {}
        t1 = time.time()
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        # [352, 400]
        res = self.voxel_generator.generate(points, self.max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        metrics["voxel_gene_time"] = time.time() - t1
        anchors = self.anchor_cache["anchors"]
        anchors_bv = self.anchor_cache["anchors_bv"]
        # anchors_dict = self.anchor_cache["anchors_dict"]
        # matched_thresholds = self.anchor_cache["matched_thresholds"]
        # unmatched_thresholds = self.anchor_cache["unmatched_thresholds"]
        example = {
            'pc_file': pc_file,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            "num_voxels": num_voxels,
            "metrics": metrics,
            "anchors": anchors
        }
        if self.anchor_area_threshold >= 0:
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > self.anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
            example['anchors_mask'] = anchors_mask
        return example

    def read_pcd_points(self, pcd_path):
        pcl_points = pcl.load_XYZI(pcd_path)
        points = []
        for point in pcl_points:
            x = point[0]
            y = point[1]
            z = point[2]
            if abs(x) < 1e-1 and abs(y) < 1e-1 and abs(z) < 1e-1:
                continue
            points.append([point[0], points[1], points[2], points[3]])
        numpy_points = np.array(points)
        return numpy_points

    def read_bin_points(self, bin_path):
        points = np.fromfile(
            str(bin_path), dtype=np.float32,
            count=-1).reshape([-1, self.NumPointFeatures])
        return points

    def get_dir_files(self, data_dir, file_post="*.*"):
        result = []
        if os.path.isdir(data_dir):
            image_path_pattern = os.path.join(data_dir, file_post)
            for filePath in glob.iglob(image_path_pattern):
                result.append(filePath)
        return result
