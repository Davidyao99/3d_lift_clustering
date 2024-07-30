# # from core.engine_scannet import Engine_Scannet
from core.pcd_laplacian_point import Engine_Laplacian_Point
import argparse
import sys

# Define the argparse parser
parser = argparse.ArgumentParser(description='scannet_laplacian_clustering')

# Add arguments to the parser
parser.add_argument('--base_dir', type=str, required=True)
parser.add_argument('--seg_type', type=str, default="deva_ram_gsam_window")
parser.add_argument('--depth_thresh', type=float, default=0.2)
parser.add_argument('--vis_pred', action='store_true', default=False)
parser.add_argument('--divider', type=int, default=1)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--experiment_name', type=str, default="test")
parser.add_argument('--experiment_run', type=str, default="test")
parser.add_argument('--radius_voxel', type=float, default=0.3)
parser.add_argument('--radius_super', type=float, default=1.0)
parser.add_argument('--voxel_size', type=float, default=0.1)
parser.add_argument('--ncut_point_thres', type=float, required=True)
parser.add_argument('--frames_used', type=float, default=0.2)
parser.add_argument('--clip_weight', type=float, default=0)
parser.add_argument('--dino_weight', type=float, default=0)
parser.add_argument('--deva_weight', type=float, default=0)
parser.add_argument('--norm_weight', type=float, default=0)
parser.add_argument('--dst_weight', type=float, default=0)
parser.add_argument('--ram_weight', type=float, default=0)
parser.add_argument('--tgt_weight', type=float, default=0)
parser.add_argument('--tgt_type', type=str, default="deva")
parser.add_argument('--gt_pos', type=float, default=0)
parser.add_argument('--gt_neg', type=float, default=0)
parser.add_argument('--norm_mean', type=float, default=0.5)
parser.add_argument('--norm_std', type=float, default=0.7)

args = parser.parse_args()

engine = Engine_Laplacian_Point(args)
engine.start()