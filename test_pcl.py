import pcl
import pcl.pcl_visualization
import numpy as np


def read_pcd(path_file):
    pcd = pcl.load_XYZI(path_file)
    return pcd


if __name__ == '__main__':
    pcd = read_pcd("/home/lpj/github/data/my_point_cloud/Ali100/pcds/003487_cloud.pcd")
    for point in pcd:
        x = point[0]
        y = point[1]
        z = point[2]
        if abs(x) < 1e-1 and abs(y) < 1e-1 and abs(z) < 1e-1:
            continue
        print(point)
    numpy_points = np.array(pcd)
    # print(numpy_points)
    # cloud = pcl.PointCloud_PointXYZRGB()
    # cloud.from_array(numpy_points)
    # print(cloud[0])
    # visual = pcl.pcl_visualization.CloudViewing()
    # visual.ShowColorCloud(cloud)
    # v = True
    # while v:
    #     v = not (visual.WasStopped())
    cloud = pcl.PointCloud(numpy_points[:, :3])
    visualcolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 0, 255, 0)
    vs = pcl.pcl_visualization.PCLVisualizering
    vss1 = pcl.pcl_visualization.PCLVisualizering()  # 初始化一个对象，这里是很重要的一步
    vs.AddPointCloud_ColorHandler(vss1, cloud, visualcolor, id=b'cloud', viewport=0)
    v = True
    while not vs.WasStopped(vss1):
        vs.Spin(vss1)
