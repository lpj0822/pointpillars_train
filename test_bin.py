import pcl
import pcl.pcl_visualization
import numpy as np

if __name__ == '__main__':
    numpy_points = np.fromfile(
        str("/home/lpj/Desktop/002976_cloud.bin"), dtype=np.float32,
        count=-1).reshape([-1, 4])
    print(numpy_points)
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