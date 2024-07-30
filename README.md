# 3d_lift_clustering

Given the latest advancements in 2D foundation models such as DINO, CLIP, SAM, can they be used to assist 3D tasks? This exploratory project aims to leverage 2D foundation models in the task of point cloud instance segmentation. Given posed RGB-D images along with a scene point cloud, our aim is to segment points that  correspondence to object instances. Various 2D features can be extracted using 2D foundation models and fused with point clouds using camera projection. Normalized graph cut is used to cluster points that have high affinity, and separate clusters with low affinity. Qualitative and Quantitative results show the application of 2D foundation models for 3D tasks is a promising future direction.

# Pipeline

