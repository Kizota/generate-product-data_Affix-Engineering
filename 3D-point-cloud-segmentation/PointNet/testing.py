import open3d as o3
import os
from shapenet_dataset import ShapenetDataset

ROOT = os.getcwd() +"\\ShapenetDataset"
sample_dataset = train_dataset = ShapenetDataset(ROOT, npoints = 20000, split='train',classification=False,normalize = False )
point, seg = sample_dataset[4000]

# pcd = o3.geometry.PointCloud()
# pcd.points = o3.utility.Vector3dVector(points)
# pcd.colors = o3.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))

# o3.visualization.draw_plotly([pcd])