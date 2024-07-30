import cv2
import numpy as np
import os
import json

from core.util import *
from core.constants import VALID_CLASS_IDS_SCANNET
from core.norm_cut import CutEngine

import scipy

import scipy.spatial as spatial
from scipy.sparse.csgraph import connected_components
from scipy import sparse
from skimage import graph
import networkx as nx

import open3d as o3d
import torch
import clip
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import time

class Engine_Laplacian_Point:

    def __init__(self, args) -> None:

        self.config = vars(args)
        for key, value in self.config.items():
            setattr(self, key, value)

        self.reset()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"{os.getpid()}: Using GPU")
        else:
            self.device = torch.device("cpu")
            print(f"{os.getpid()}: Using CPU")

        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large', force_download=True).to(self.device)

    def prepare_paths(self, scan):

        """
        Prepare paths for saving and loading data
        """

        self.work_dir = os.path.join(self.base_dir, "scans", scan)
        self.work_dir_fusion = os.path.join(self.work_dir, "fusion_results")

        self.save_dir = os.path.join(self.work_dir_fusion, self.experiment_name, self.experiment_run, "result")
        self.save_vis_dir = os.path.join(self.base_dir, "vis", scan, self.experiment_name, self.experiment_run)

        os.makedirs(self.save_dir, exist_ok=True)
        if self.vis_pred:
            os.makedirs(self.save_vis_dir, exist_ok=True)
        
        self.deva_dir = os.path.join(self.work_dir, self.seg_type)
        self.mask_dir_path = os.path.join(self.deva_dir, "Annotations")
        self.mask_dirs = sorted(os.listdir(self.mask_dir_path))
        self.tracklets_json_path = os.path.join(self.deva_dir, "tracklets.json")

        self.frame_dirs = sorted(os.listdir(os.path.join(self.work_dir, "color")))

        with open(f'{self.save_dir}/config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        

    def reset(self):

        self.pcd = None
        self.pcd_coords = None
        self.pcd_KDTree = None

        self.mask_ids = None
        self.clip_feat = None

    def process_tracklets(self):
        """
        Reads in annotations json
        sets:
        self.ids_to_idx_dict: maps id to idx, 0 is bg 
        self.idx_to_ids_dict: maps idx to id, 0 is bg
        self.idx_label_vector: maps idx to label vector
        initializes:
        self.points_deva
        self.points_ram
        """
        id_dict = json.load(open(self.tracklets_json_path, 'r'))

        all_ids = set([int(id) for id in id_dict.keys()])
        all_labels = set()

        print(f"{os.getpid()}: Total of {len(all_ids)} ids processed")

        for id in all_ids:
            for label in id_dict[str(id)]['label']:
                all_labels.add(label)

        self.idx_label_vector = np.zeros((len(all_ids), len(all_labels)), dtype=np.float32)
        label_to_idx_dict = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}

        self.ids_to_idx_dict = {id: idx+1 for idx, id in enumerate(all_ids)}
        self.ids_to_idx_dict[0] = 0
        self.idx_to_ids_dict = {idx: id for id, idx in self.ids_to_idx_dict.items()}

        for id in all_ids:
            for label in id_dict[str(id)]['label']:
                self.idx_label_vector[self.ids_to_idx_dict[id]-1, label_to_idx_dict[label]] += 1
        self.idx_label_vector = self.idx_label_vector / (np.linalg.norm(self.idx_label_vector, axis=-1, keepdims=True) + 1e-16) # normalize

        self.points_deva = np.zeros((self.pcd_coords.shape[0], len(all_ids)), dtype=np.int32) # bins for each points, bg not included
        self.points_ram = np.zeros((self.pcd_coords.shape[0], len(all_labels)), dtype=np.float32) # bins for each points

    def load_camera(self):

        """
        Load camera intrinsics and values from config.json provided by scannet200
        """

        with open(os.path.join(self.work_dir, "config.json"), 'r') as fp:
            cam_config = json.load(fp)
        cam_intr = np.asarray(cam_config['cam_intr'])
        depth_scale = cam_config['depth_scale']

        return cam_intr, depth_scale

    def get_boxes(self, mask):
        '''
        Returns bounding boxes for each instance in the mask in order of ascending instance id, ignoring idx=0 (background)
        '''

        boxes = []
        for id in np.unique(mask):
            if id == 0:
                continue
            mask_id = mask == id
            y,x = np.where(mask_id)
            boxes.append([x.min(), y.min(), x.max()+1, y.max()+1])

        return boxes

    def get_hidden_states_dino(self, image):

        img = Image.fromarray(image)
        size = image.shape[:2]
        resize_size = ((size[0]//14)*14, (size[1]//14)*14) # resize to multiple of 14 as each patch is 14x14
        transform = dino_transforms(resize_size)
        inputs = {
            "pixel_values": transform(img)[None,:,:,:].to(self.device)
        }

        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            last_hidden_states = outputs.last_hidden_state[:,1:,:]

            last_hidden_states = last_hidden_states.reshape(-1, size[0]//14, size[1]//14, 1024).permute(0,3,1,2)
            dino_feats = torch.nn.functional.interpolate(last_hidden_states, size=image.shape[:2], mode="bilinear", align_corners=False)
            dino_feats = dino_feats.squeeze().permute(1,2,0).cpu().detach().numpy()

            torch.cuda.empty_cache()

        return dino_feats

    def get_hidden_states_clip(self, image, boxes):

        crops = []
        h,w = image.shape[:2]
        for i,box in enumerate(boxes):
            crop = image[max(0,int(box[1])):min(h,int(box[3])), max(0,int(box[0])): min(w,int(box[2]))]
            crop_image = Image.fromarray(crop)
            crop = self.clip_preprocess(crop_image)
            crops.append(crop)

        crops_torch = torch.from_numpy(np.stack(crops)).to(self.device)

        with torch.no_grad():
            logits = self.clip_model.encode_image(crops_torch)

        crop_feats = torch.nn.functional.normalize(logits, dim=-1)
        torch.cuda.empty_cache()
        return crop_feats.cpu().numpy()

    def get_orig_pred_mask(self, ts_idx):

        '''
        Generates original prediction mask and maps id to idx so as to use int16
        0 is saved for background
        returns
        idx_mask: (h,w) matrix where each entry is the idx of the instance
        '''

        # mask_dir = os.path.join(self.work_dir, self.tgt_type, "Annotations")
        
        if self.tgt_type == "semsam":
            mask_dir = os.path.join(self.work_dir, "semantic_sam", "mask")
            mask_path = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith('.npy')][ts_idx//5]
            mask = np.load(mask_path)
        elif self.tgt_type == "gsam":
            mask_dir = os.path.join(self.work_dir, "ram_gsam_window", "mask")
            mask_path = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith('.png')][ts_idx]
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.int32)
            mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256

        idx_mask = np.zeros(mask.shape, np.int16)
        ids = sorted(np.unique(mask))

        if 0 not in ids: # make sure 0 is background
            ids = [0] + ids

        for i,id in enumerate(ids):
            idx_mask[mask==id] = i

        return idx_mask

    def get_pred_mask(self, ts_idx):

        '''
        Generates prediction mask and maps id to idx so as to use int16
        returns
        idx_mask: (h,w) matrix where each entry is the idx of the instance
        mask_idx_map: (n_instances) array where each entry is the idx of the instance id, 0 is bg
        '''

        mask_path = self.mask_dirs[ts_idx]
        mask = np.load(os.path.join(self.mask_dir_path, mask_path))
        idx_mask = np.zeros(mask.shape, np.int16)
        ids = sorted(np.unique(mask))

        if 0 not in ids:
            ids = [0] + ids

        mask_idx_map = [] # 0 always maps to 0

        for i,id in enumerate(ids):
            idx_mask[mask==id] = i
            mask_idx_map.append(self.ids_to_idx_dict[id])

        mask_idx_map = np.array(mask_idx_map, dtype=int)

        return idx_mask, mask_idx_map

    def get_frame_rgb(self, ts_idx):

        rgb_path = self.frame_dirs[ts_idx]
        rgb = cv2.imread(os.path.join(self.work_dir, "color", rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        return rgb

    def read_seg_json(self, json_path):

        with open(json_path, 'r') as fp:
            seg_dict = json.load(fp)

        indicies = seg_dict["segIndices"]

        inst = np.array(indicies)

        inst_copy = np.zeros(inst.shape, dtype=np.int32)
        for i, ins in enumerate(np.unique(inst)):
            inst_copy[inst==ins] = i

        return inst_copy

    def prepare_pcds(self, video_name):

        """
        Load in point clouds, downsample, and create point cloud features
        """

        pcd_path = os.path.join(self.work_dir, f"{video_name}_vh_clean_2.ply")
        seg_json_path = os.path.join(self.work_dir, f"{video_name}_vh_clean_2.0.010000.segs.json")

        self.pcd = o3d.io.read_point_cloud(pcd_path)
        self.pcd_coords_orig = np.asarray(self.pcd.points)

        self.pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.2, max_nn = 30))
        # self.pcd.orient_normals_to_align_with_direction(np.array([1,1,1]))
        self.points_normals = np.array(self.pcd.normals)
        self.pcd_coords = np.asarray(self.pcd.points)
        self.pcd_KDTree = spatial.KDTree(self.pcd_coords)

        self.pcd_super = self.read_seg_json(seg_json_path)

        self.points_clip = np.zeros((self.pcd_coords.shape[0], 768))
        self.points_dino = np.zeros((self.pcd_coords.shape[0], 1024))

        points_kdtree = scipy.spatial.KDTree(self.pcd_coords)    
        points_neighbors = points_kdtree.query(self.pcd_coords, 8)[1]  #(n,k)
        seg_num = len(np.unique(self.pcd_super))
        
        self.seg_direct_neighbors = np.zeros((seg_num,seg_num),dtype=bool) # binary matrix, "True" indicating the two superpoints are neighboring
        for id in np.unique(self.pcd_super):
            members = self.pcd_super==id
            neighbors = points_neighbors[members] 
            neighbor_seg_ids = self.pcd_super[neighbors]
            self.seg_direct_neighbors[id][neighbor_seg_ids] = 1
        self.seg_direct_neighbors[np.eye(seg_num,dtype=bool)] = 0 #exclude self
        self.seg_direct_neighbors[self.seg_direct_neighbors.T] = 1 #make neighboring matrix symmetric

        self.super_pos = np.zeros((seg_num, 3))

        for i,cluster in enumerate(np.unique(self.pcd_super)):
            self.super_pos[i] = np.mean(self.pcd_coords[self.pcd_super==cluster], axis=0)

    def visualize_features(self, features, image, idx, feat_type):

        feat_rgb = visualize_features(features) * 255
        vis_frame = np.hstack((image, feat_rgb))
        if not os.path.exists(os.path.join(self.work_dir_fusion, feat_type)):
            os.makedirs(os.path.join(self.work_dir_fusion, feat_type))
        vis_frame = cv2.cvtColor(vis_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.work_dir_fusion, feat_type, self.frame_dirs[idx]), vis_frame)

    def array_to_occ(self, arr):

        """
        Convert array to occupancy binary matrix
        """

        ids = np.unique(arr)
        res = np.zeros((len(ids), len(arr)))
        for i, id in enumerate(ids):
            res[i] = arr == id
        return res

    def populate_super_feats(self, save=True):

        cam_intr, depth_scale = self.load_camera()
        frame_ids = [f.split('-')[0] for f in sorted(os.listdir(os.path.join(self.work_dir, self.seg_type, "Annotations")))]
        
        all_points_world = np.array(self.pcd_coords)
        all_points_world_homo = np.hstack((all_points_world, np.ones((all_points_world.shape[0], 1))))

        self.super_tgt = np.zeros((len(np.unique(self.pcd_super)), len(np.unique(self.pcd_super))), dtype=np.float32)

        super_ids = np.unique(self.pcd_super)

        super_vis = torch.zeros((len(super_ids), len(super_ids)), dtype=torch.float64).to(self.device) # count of total visible points between each super point
        super_cooccur = torch.zeros((len(super_ids), len(super_ids)), dtype=torch.float64).to(self.device) # count of total visible points between each super point that were seen in same mask

        use_every = int(1/self.frames_used)

        super_occ_all = torch.from_numpy(self.array_to_occ(self.pcd_super)).to(self.device)
        super_counts_all = torch.sum(super_occ_all, axis=1) # number of points in each primitive

        super_bg = torch.zeros((len(super_ids))).to(self.device) # points in each primitive in backgroud
        super_total = torch.zeros((len(super_ids))).to(self.device) # points in each primitive that was seen
        
        for i, frame_id in enumerate(frame_ids):

            if i % use_every != 0:
                continue

            print(f"Coloring frame {i} out of {len(frame_ids)} frames", end="\r", flush=True)

            if self.tgt_type == "deva":
                pred_mask, _ = self.get_pred_mask(i)
            else:
                pred_mask = self.get_orig_pred_mask(i)
            
            cam_pose = np.loadtxt(os.path.join(self.work_dir, 'poses', f"{frame_id}-pose.txt"))
            depth_im_path = os.path.join(self.work_dir, 'depth', f"{frame_id}-depth.png")
            depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
            
            all_points_cam_homo = np.linalg.inv(cam_pose) @ all_points_world_homo.T
            all_points_cam = (all_points_cam_homo / all_points_cam_homo[3,:])[:3, :].T

            proj_pts = compute_projected_pts(all_points_cam, cam_intr) # x, y
            visibility_mask = compute_visibility_mask(all_points_cam, proj_pts, depth_im, self.config["depth_thresh"]) # which points are visible in self.pcd
            selected_idxes = np.where(visibility_mask)[0]
            
            super_seen = self.pcd_super[visibility_mask]
            super_seen_ids = np.unique(super_seen) # seen primitive ids
            super_frame_indices = torch.from_numpy(np.where(np.in1d(super_ids, super_seen_ids))[0]).to(self.device) # index of ids in super_ids in super_seen_ids
            points_mask_id = torch.from_numpy(pred_mask[proj_pts[selected_idxes, 1], proj_pts[selected_idxes, 0]]).to(self.device) # should bg be treated as a segment?
            super_occ = torch.from_numpy(self.array_to_occ(super_seen)).to(self.device) # super_ids x binary occupancy

            super_mask_hist = []

            pred_mask_non_bg = np.unique(pred_mask)
            pred_mask_non_bg = pred_mask_non_bg[pred_mask_non_bg!=0] # remove background histogram

            for i in pred_mask_non_bg:
                
                points_mask = points_mask_id == i
                super_occ_mask = super_occ * points_mask
                hist_super = torch.sum(super_occ_mask, axis=1) # for each super_id, how many points were seen in this mask
                super_mask_hist.append(hist_super)

            points_mask_bg = points_mask_id == 0
            super_occ_bg = super_occ * points_mask_bg
            super_bg[super_frame_indices] += torch.sum(super_occ_bg, axis=1) # total in bg
            super_total[super_frame_indices] += torch.sum(super_occ, axis=1) # total seen

            if len(super_mask_hist) == 0:
                print(f"frame {i} out of {len(frame_ids)} frames has no objects")
                continue

            super_mask_hist = torch.stack(super_mask_hist, dim=1)
            super_mask_hist_norm = torch.nn.functional.normalize(super_mask_hist, dim=-1)

            count_super = torch.sum(super_occ, axis=1)
            visibility = count_super / super_counts_all[super_frame_indices] # get fraction of points seen in each super point
            visibility_mtx = visibility[:,None] * visibility[None, :]

            super_cooccur[super_frame_indices[:,None], super_frame_indices] += visibility_mtx * (super_mask_hist_norm @ super_mask_hist_norm.T)
            super_vis[super_frame_indices[:,None], super_frame_indices] += visibility_mtx # times points from both were seen tgt
            torch.cuda.empty_cache()

        self.super_tgt = (super_cooccur / (super_vis + 1e-16)).detach().cpu().numpy()
        self.super_bg = (super_bg / (super_total + 1e-16)).detach().cpu().numpy()

        torch.cuda.empty_cache()

        if save:
            np.save(f'{self.work_dir_fusion}/super_tgt_vis_weighted_{self.tgt_type}_mask_hist_{self.frames_used}.npy', self.super_tgt)
            np.save(f'{self.work_dir_fusion}/super_bg_{self.tgt_type}_{self.frames_used}.npy', self.super_bg)

    def populate_point_feats(self, scan):

        '''
        Generates the required mask and visibility matrices
        self.points_clip: (n_points, 768) matrix where each entry is the normalized clip feature
        self.points_dino: (n_points, 1024) matrix where each entry is the normalized dino feature
        self.points_deva: (n_points, n_instances) matrix where each entry is the normalized deva feature
        self.points_ram: (n_points, n_labels) matrix where each entry is the normalized ram feature
        '''

        if not os.path.exists(f'{self.work_dir_fusion}/points_clip_{self.frames_used}.npy'):


            cam_intr, depth_scale = self.load_camera()
            frame_ids = [f.split('-')[0] for f in sorted(os.listdir(os.path.join(self.work_dir, 'color')))]
            
            all_points_world = np.array(self.pcd_coords)
            all_points_world_homo = np.hstack((all_points_world, np.ones((all_points_world.shape[0], 1))))

            use_every = int(1/self.frames_used)

            for i, frame_id in enumerate(frame_ids):

                if i % use_every != 0:
                    continue

                print(f"{os.getpid()}: Coloring frame {i} out of {len(frame_ids)} frames", end="\r", flush=True)

                pred_mask, mask_idx_map = self.get_pred_mask(i)

                boxes = self.get_boxes(pred_mask)
                if len(boxes) == 0:
                    continue
                image = self.get_frame_rgb(i)
                dino_features = self.get_hidden_states_dino(image)
                
                clip_features = self.get_hidden_states_clip(image, boxes)

                cam_pose = np.loadtxt(os.path.join(self.work_dir, 'poses', f"{frame_id}-pose.txt"))
                depth_im_path = os.path.join(self.work_dir, 'depth', f"{frame_id}-depth.png")
                depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale

                all_points_cam_homo = np.linalg.inv(cam_pose) @ all_points_world_homo.T
                all_points_cam = (all_points_cam_homo / all_points_cam_homo[3,:])[:3, :].T

                proj_pts = compute_projected_pts(all_points_cam, cam_intr) # x, y
                visibility_mask = compute_visibility_mask(all_points_cam, proj_pts, depth_im, self.depth_thresh)
                selected_idxes = np.where(visibility_mask)[0] # bool mask for visible points
                
                sel_idx_mask_id = pred_mask[proj_pts[selected_idxes, 1], proj_pts[selected_idxes, 0]]
                non_bg_sel_idx = selected_idxes[sel_idx_mask_id != 0] # bool mask for visible + not bg pixels
                sel_idx_mask_id_no_bg = pred_mask[proj_pts[non_bg_sel_idx, 1], proj_pts[non_bg_sel_idx, 0]]
                
                self.points_clip[non_bg_sel_idx] += clip_features[sel_idx_mask_id_no_bg - 1]
                self.points_dino[non_bg_sel_idx] += dino_features[proj_pts[non_bg_sel_idx, 1], proj_pts[non_bg_sel_idx, 0]]
                self.points_ram[non_bg_sel_idx] += self.idx_label_vector[mask_idx_map[sel_idx_mask_id_no_bg]-1]
                self.points_deva[non_bg_sel_idx, mask_idx_map[sel_idx_mask_id_no_bg]-1] += 1

            self.points_clip = self.points_clip / (np.linalg.norm(self.points_clip, axis=-1, keepdims=True) + 1e-16)
            self.points_dino = self.points_dino / (np.linalg.norm(self.points_dino, axis=-1, keepdims=True) + 1e-16)
            self.points_deva = self.points_deva / (np.linalg.norm(self.points_deva, axis=-1, keepdims=True) + 1e-16)
            self.points_ram = self.points_ram / (np.linalg.norm(self.points_ram, axis=-1, keepdims=True) + 1e-16)

            np.save(f'{self.work_dir_fusion}/points_clip_{self.frames_used}.npy', self.points_clip)
            np.save(f'{self.work_dir_fusion}/points_dino_{self.frames_used}.npy', self.points_dino)
            np.save(f'{self.work_dir_fusion}/points_deva_{self.frames_used}.npy', self.points_deva)
            np.save(f'{self.work_dir_fusion}/points_ram_{self.frames_used}.npy', self.points_ram)

        else:

            self.points_clip = np.load(f'{self.work_dir_fusion}/points_clip_{self.frames_used}.npy')
            self.points_dino = np.load(f'{self.work_dir_fusion}/points_dino_{self.frames_used}.npy')
            self.points_deva = np.load(f'{self.work_dir_fusion}/points_deva_{self.frames_used}.npy')
            self.points_ram = np.load(f'{self.work_dir_fusion}/points_ram_{self.frames_used}.npy')

            self.points_dino_vis = visualize_features(self.points_dino)
            self.points_clip_vis = visualize_features(self.points_clip)
            self.points_deva_vis = visualize_features(self.points_deva)
            self.points_ram_vis = visualize_features(self.points_ram)
            self.points_norm_vis = visualize_features(self.points_normals)

            os.makedirs(os.path.join(self.base_dir, "vis", scan), exist_ok=True)

            np.save(os.path.join(self.base_dir, "vis", scan, "pcd_dino_rgb_pca.npy"), self.points_dino_vis)
            np.save(os.path.join(self.base_dir, "vis", scan, "pcd_clip_rgb_pca.npy"), self.points_clip_vis)
            np.save(os.path.join(self.base_dir, "vis", scan, "pcd_deva_rgb_pca.npy"), self.points_deva_vis)
            np.save(os.path.join(self.base_dir, "vis", scan, "pcd_ram_rgb_pca.npy"), self.points_ram_vis)
            np.save(os.path.join(self.base_dir, "vis", scan, "pcd_norms_rgb_pca.npy"), self.points_normals)
        
        clustering_points = self.pcd_super
        clusters = np.unique(clustering_points)
        n_super = len(clusters)

        self.super_clip = np.zeros((n_super, self.points_clip.shape[1]))
        self.super_dino = np.zeros((n_super, self.points_dino.shape[1]))
        self.super_pos = np.zeros((n_super, 3))
        self.super_deva = np.zeros((n_super, self.points_deva.shape[1]))
        self.super_ram = np.zeros((n_super, self.points_ram.shape[1]))

        for i,cluster in enumerate(clusters):

            self.super_clip[i] = np.sum(self.points_clip[clustering_points==cluster], axis=0)
            self.super_dino[i] = np.sum(self.points_dino[clustering_points==cluster], axis=0)
            self.super_pos[i] = np.mean(self.pcd_coords[clustering_points==cluster], axis=0)
            self.super_deva[i] = np.sum(self.points_deva[clustering_points==cluster], axis=0)
            self.super_ram[i] = np.sum(self.points_ram[clustering_points==cluster], axis=0)

        self.super_clip = self.super_clip / (np.linalg.norm(self.super_clip, axis=-1, keepdims=True) + 1e-16)
        self.super_dino = self.super_dino / (np.linalg.norm(self.super_dino, axis=-1, keepdims=True) + 1e-16)
        self.super_deva = self.super_deva / (np.linalg.norm(self.super_deva, axis=-1, keepdims=True) + 1e-16)
        self.super_ram = self.super_ram / (np.linalg.norm(self.super_ram, axis=-1, keepdims=True) + 1e-16) 

    def normalize_features_clipped(self, weights, mean_res, std_res):

        """
        Adjust weights to have mean=mean_res and std=std_res
        """
        mean = torch.mean(weights)
        std = torch.std(weights)

        new_weights = (std_res / std) * (weights - mean) + mean_res

        new_weights[new_weights>1] = 1 #  clip to 0 and 1
        new_weights[new_weights<=0] = 1e-16

        return new_weights, mean, std

    def gen_superpoint_features(self, clustering_points=None, use_saved=False):

        '''
        Consolidates the following featurse for each super point formed by clustering
        Consolidation is done by summing and normalizing
        self.super_clip: (n_super, 768) matrix where each entry is the clip feature of the super point
        self.super_dino: (n_super, 1024) matrix where each entry is the dino feature of the super point
        self.super_pos: (n_super, 3) matrix where each entry is the position of the super point
        self.super_deva: (n_super, n_deva_instances) matrix where each entry is the number of points in the super point belonging to that deva instance
        '''
        if use_saved:
            self.super_clip = np.load(f'{self.work_dir_fusion}/super_clip_{self.frames_used}.npy')
            self.super_dino = np.load(f'{self.work_dir_fusion}/super_dino_{self.frames_used}.npy')
            self.super_deva = np.load(f'{self.work_dir_fusion}/super_deva_{self.frames_used}.npy')
            self.super_ram = np.load(f'{self.work_dir_fusion}/super_ram_{self.frames_used}.npy')

        if os.path.exists(f'{self.work_dir_fusion}/super_tgt_vis_weighted_{self.tgt_type}_mask_hist_{self.frames_used}.npy'):
            self.super_tgt = np.load(f'{self.work_dir_fusion}/super_tgt_vis_weighted_{self.tgt_type}_mask_hist_{self.frames_used}.npy')
        else:
            raise ValueError("super_tgt not found")

        if clustering_points is None:
            return

        clusters = np.unique(clustering_points)
        clusters = clusters[clusters!=-1] # remove bg
        n_super = len(clusters)

        super_clip_updated = np.zeros((len(clusters), self.super_clip.shape[1]))
        super_dino_updated = np.zeros((len(clusters), self.super_dino.shape[1]))
        super_deva_updated = np.zeros((len(clusters), self.super_deva.shape[1]))
        super_ram_updated = np.zeros((len(clusters), self.super_ram.shape[1]))
        super_pos_updated = np.zeros((len(clusters), self.super_pos.shape[1]))

        for i,cluster in enumerate(clusters):

            super_clip_updated[i] = np.sum(self.super_clip[clustering_points==cluster], axis=0)
            super_dino_updated[i] = np.sum(self.super_dino[clustering_points==cluster], axis=0)
            super_pos_updated[i] = np.mean(self.super_pos[clustering_points==cluster], axis=0)
            super_deva_updated[i] = np.sum(self.super_deva[clustering_points==cluster], axis=0)
            super_ram_updated[i] = np.sum(self.super_ram[clustering_points==cluster], axis=0)

        self.super_clip = (super_clip_updated / (np.linalg.norm(super_clip_updated, axis=-1, keepdims=True) + 1e-16))
        self.super_dino = (super_dino_updated / (np.linalg.norm(super_dino_updated, axis=-1, keepdims=True) + 1e-16))
        self.super_deva = (super_deva_updated / (np.linalg.norm(super_deva_updated, axis=-1, keepdims=True) + 1e-16))
        self.super_ram = (super_ram_updated / (np.linalg.norm(super_ram_updated, axis=-1, keepdims=True) + 1e-16))
        self.super_pos = super_pos_updated

    def gen_super_adj_mtx(self):

        '''
        Generates adjacency matrix for super points
        '''
        super_clip_torch = torch.from_numpy(self.super_clip).double().to(self.device)
        super_dino_torch = torch.from_numpy(self.super_dino).double().to(self.device)
        super_deva_torch = torch.from_numpy(self.super_deva).double().to(self.device)
        super_ram_torch = torch.from_numpy(self.super_ram).double().to(self.device)
        
        super_clip_torch = torch.nn.functional.normalize(super_clip_torch, dim=-1)
        super_dino_torch = torch.nn.functional.normalize(super_dino_torch, dim=-1)
        super_deva_torch = torch.nn.functional.normalize(super_deva_torch, dim=-1)
        super_ram_torch = torch.nn.functional.normalize(super_ram_torch, dim=-1)

        super_clip_relation = super_clip_torch @ super_clip_torch.T
        super_dino_relation = super_dino_torch @ super_dino_torch.T
        super_deva_relation = super_deva_torch @ super_deva_torch.T
        super_ram_relation = super_ram_torch @ super_ram_torch.T

        # self.super_tgt = self.c(self.super_tgt, self.norm_mean, self.norm_std)[0]

        dst_matrix = spatial.distance_matrix(self.super_pos, self.super_pos)

        vis_mask = dst_matrix < self.radius_super

        #--------------------- Normalization ---------------------
        super_clip_relation[super_clip_relation != 0], mean_clip, std_clip = self.normalize_features_clipped(super_clip_relation[super_clip_relation != 0], self.norm_mean, self.norm_std)
        super_dino_relation[super_dino_relation != 0], mean_dino, std_dino = self.normalize_features_clipped(super_dino_relation[super_dino_relation != 0], self.norm_mean, self.norm_std)
        # --------------------------------------------------------   

        adj_matrix = (super_clip_relation).cpu().numpy() * self.clip_weight + \
                    (super_dino_relation).cpu().numpy() * self.dino_weight + \
                    (super_deva_relation).cpu().numpy() * self.deva_weight + \
                    (super_ram_relation).cpu().numpy() * self.ram_weight + \
                    self.super_tgt * self.tgt_weight + 1e-12

        np.fill_diagonal(adj_matrix, 0)

        self.bg = np.where(np.sum(adj_matrix, axis=-1) < 1)[0]

        rbf_kernel = np.exp(-dst_matrix**2/(2*0.5**2))
        adj_matrix = adj_matrix * rbf_kernel

        torch.cuda.empty_cache()

        adj_matrix = adj_matrix * vis_mask

        adj_matrix = adj_matrix / (self.clip_weight + self.dino_weight + self.deva_weight + self.ram_weight + self.tgt_weight + 1e-16) # normalize

        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix

    def finalize_clustering(self, cluster_super):

        final_cluster = np.zeros(self.pcd_coords.shape[0], dtype=np.int32) - 1 # -1 for background

        super_ids = np.unique(self.pcd_super)

        cluster_super_no_bgs = np.unique(cluster_super)[np.unique(cluster_super) != -1] # remove bgs

        for super_idx in cluster_super_no_bgs:

            super_ids_cluster = super_ids[cluster_super == super_idx]

            final_cluster[np.isin(self.pcd_super, super_ids_cluster)] = super_idx

        return final_cluster

    def ids_to_matrix(self, pcd_labels, remove_bg=True):

        '''
        Inputs:
        pcd_labels -> list of label for each point in the scene (unsorted). 
        remove_bg -> whether to remove background class (0) or not
        Returns:
        matrix -> (n_instances, n_points) matrix where each entry is 1 if the point belongs to that instance, 0 otherwise
        labels -> list of instance ids that is the row index of the matrix
        '''   

        labels = np.unique(pcd_labels)
        if remove_bg:
            labels = labels[labels != 0]
        matrix = np.zeros((len(labels), len(pcd_labels)), dtype=np.float32)

        for i, label in enumerate(labels):
            matrix[i] = pcd_labels == label

        return matrix, labels

    def check_disconnect(self, cluster, adj_matrix_point):

        """
        Checks if the cluster is disconnected
        """

        cluster_ids = np.unique(cluster)

        disconnects = 0

        for cluster_id in cluster_ids:
            
            sel_pcd_idx = np.where(cluster == cluster_id)[0]
            sel_pcd = self.pcd_coords[sel_pcd_idx]

            dst_matrix = spatial.distance_matrix(sel_pcd, sel_pcd)
            adj_matrix = dst_matrix <= self.radius_voxel
            n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

            if n_components != 1:
                cluster1 = sel_pcd_idx[np.where(labels == 0)[0]]
                cluster2 = sel_pcd_idx[np.where(labels == 1)[0]]
                a = adj_matrix_point[cluster2]
                a = a[:,cluster1].A
                disconnects += 1

        print(f"{os.getpid()}: Total of {disconnects} disconnects found")

        return disconnects

    def oracle_merge(self, clustering_points, is_sem=False):

        """
        Merges clusters based on oracle using majority voting with gt
        Final cluster idx is 0 if background, otherwise it is the idx of gt inst/sem id without bg + 1
        """

        instance_ids = (np.load(os.path.join(self.work_dir, "instance_gt.npy")))

        mask = np.isin(instance_ids//1000, VALID_CLASS_IDS_SCANNET)
        gt_ids = np.where(mask, instance_ids, 0)

        if is_sem:
            gt_ids = gt_ids//1000 # get classes

        gt_matrix, gt_matrix_labels = self.ids_to_matrix(gt_ids, False)
        pred_matrix, pred_matrix_labels = self.ids_to_matrix(clustering_points, False)

        pred_gt_vote = pred_matrix @ gt_matrix.T # (n_pred, n_gt) showing number of votes for each pred to each gt

        assignment = np.argmax(pred_gt_vote, axis=1) # (n_pred, ) showing which gt each pred is assigned to

        final_cluster = np.zeros(len(clustering_points), dtype=np.int32)

        for i, cluster_idx in enumerate(pred_matrix_labels):

            final_cluster[clustering_points == cluster_idx] = gt_matrix_labels[assignment[i]]

        return final_cluster
        
    def merge_normalized_cut(self, matrix):

        labels = {x:{"labels": [x]} for x in range(matrix.shape[0])}
        G = nx.from_numpy_array(matrix)
        nx.set_node_attributes(G, labels)
        tmp = np.zeros(1, dtype=np.int32)+matrix.shape[0]
        try:
            cluster_vis_tmp = graph.cut_normalized(tmp, G, max_edge=np.max(matrix), thresh=self.ncut_point_thres, rng=1)
        except Exception as e:
            print(f"{os.getpid()}: Error in normalized cut: {e}")
            return None
        
        cluster_vis = np.zeros((len(cluster_vis_tmp)+1, matrix.shape[0]), dtype=np.int32)
        for i, partition in enumerate(cluster_vis_tmp):
            cluster_vis[i+1] = cluster_vis[i]
            cluster_vis[i+1][partition] = i+1

        clustering = []

        for n, d in G.nodes(data=True):
            clustering.append(d['ncut label'])

        return clustering, cluster_vis

    def cluster(self, adj_matrix):

        """
        Used to cluster super points
        """

        test_connectivity = adj_matrix.copy()
        test_connectivity[test_connectivity > 0] = 1                                                                                                                           
        n_components, clustering = connected_components(csgraph=test_connectivity, directed=False, return_labels=True)

        print(f"{os.getpid()}: Number of connected components {n_components}")

        cluster_engine = CutEngine(self.ncut_point_thres, 1)

        clustering, clustering_vis = cluster_engine.normalized_cut(adj_matrix, self.super_pos, self.vis_pred)

        if clustering is None:
            return None, None

        print(f"{os.getpid()}: Number of clusters {len(np.unique(clustering))}")

        # ------------------------ deal with bg primitives ------------------------
        clustering = np.array(clustering)
        clustering[self.bg] = -1 # set bg to -1

        if self.vis_pred:
            clustering_vis[:, self.bg] = 0 # set bg to 0
            indexes = []
            for i in range(len(clustering_vis)-1):
                if np.sum(clustering_vis[i] != clustering_vis[i+1]) == 0:
                    indexes.append(i)
            clustering_vis = np.delete(clustering_vis, indexes, axis=0)

        return clustering, clustering_vis

    def save_res(self, clustering, clustering_vis, video_name, vis=False):

        """
        Generates outputs for both upsampled and downsampled
        """
        clustering = self.finalize_clustering(clustering) # upscale to points

        np.save(f"{self.save_dir}/pred_inst.npy", clustering)
        np.save(f"{self.save_dir}/pred_features.npy", self.super_clip)

        assert(len(np.unique(clustering[clustering != -1])) == self.super_clip.shape[0])

        if vis:

            print(f"Saving visualization for {video_name} to {self.save_vis_dir}")

            gt_inst = np.load(f"{self.work_dir}/instance_gt.npy")

            os.makedirs(self.save_vis_dir, exist_ok=True)

            np.save(f"{self.save_vis_dir}/pred_features.npy", self.super_clip)
            np.save(f"{self.save_vis_dir}/clustering_vis.npy", np.array(clustering_vis))
            np.save(f"{self.save_vis_dir}/pred_super.npy", self.pcd_super)
            np.save(f"{self.save_vis_dir}/pred_inst.npy", clustering)
            np.save(f"{self.save_vis_dir}/gt_inst.npy", gt_inst)

            o3d.io.write_point_cloud(f"{self.save_vis_dir}/{video_name}_vh_clean_2.ply", self.pcd)

    def start(self): # entry point

        scans = sorted(os.listdir(os.path.join(self.base_dir, "scans")))

        avg_clusters = 0

        for i, scan in enumerate(scans):

            
            if scan[-4:] == ".pkl":
                continue

            if i % self.divider != self.mode:
                continue

            self.prepare_paths(scan)

            print(f"{os.getpid()}: Working on {i}:{scan} out of {len(scans)} videos", flush=True)

            self.prepare_pcds(scan)

            print("done preparing")

            self.process_tracklets()

            self.populate_point_feats(scan)

            self.populate_super_feats(True)

            super_point_adj_matrix = self.gen_super_adj_mtx()
            start = time.time()
            final_cluster, clustering_vis = self.cluster(super_point_adj_matrix)
            end = time.time()
            print(f"Time taken for clustering: {end-start} seconds")

            non_bg_clusters = len(np.unique(final_cluster)[np.unique(final_cluster) != -1])
            print(non_bg_clusters)
            avg_clusters += non_bg_clusters

            self.gen_superpoint_features(final_cluster, False)

            self.save_res(final_cluster, clustering_vis, scan, True)

            self.reset()

        self.config["Average Clusters"] = avg_clusters / len(scans)

        with open(f'{self.save_dir}/config.json', 'w') as f: # save with Average clusters for last scan
            json.dump(self.config, f, indent=4)