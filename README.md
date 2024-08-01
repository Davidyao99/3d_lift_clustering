# 3D_lift_clustering

Given the latest advancements in 2D foundation models such as DINO, CLIP, SAM, can they be used to assist 3D tasks? This exploratory project aims to leverage 2D foundation models in the task of point cloud instance segmentation. Given posed RGB-D images along with a scene point cloud, our aim is to segment points that correspond to object instances. Various 2D features can be extracted using 2D foundation models and fused with point clouds using camera projection. Normalized graph cut is used to cluster points that have high affinity, and separate clusters with low affinity. Qualitative and Quantitative results show the application of 2D foundation models for 3D tasks is a promising future direction.

# Pipeline

Inputs to our algorithm are camera poses, camera intrinsics, color frames, depth frames, and scene point clouds. 

During preprocessing, we want to generate accurate and temporally consistent 2D segmentations of objects present in the video. We utilize Recognize-Anything Model to obtain semantic labels for prominent object in each frame. These labels are fed to Grounding-DINO to generate bounding boxes, which can be used to prompt Segment-Anything for fine grained segmentation. In order to improve temporal consistency, we use DEVA to obtain segmentation tracklets for each object. We now have segmentation tracklets for each prominent object in our video, along with its semantic labels (eg. chair, table).



# Preprocessing

Preprocessing is performed in the submodule `Tracking-Anything-with-DEVA`. Remember to initialize it with the following command:

```
git submodule update --init --recursive
```

##

```

```
