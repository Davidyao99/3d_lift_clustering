# 3D_lift_clustering

<p align="center">
  <img src="https://github.com/user-attachments/assets/f7583223-c303-45ae-a41c-e82500218ef9" alt="Image 2" width="370" />
  <img src="https://github.com/user-attachments/assets/520e3837-7d91-4983-9c60-78db145080b1" alt="Image 1" width="310" />
</p>

Given the latest advancements in 2D foundation models such as DINO, CLIP, SAM, can they be used to assist 3D tasks? This exploratory project aims to leverage 2D foundation models in the task of point cloud instance segmentation. Given posed RGB-D images along with a scene point cloud, our aim is to segment points that correspond to object instances. Various 2D features can be extracted using 2D foundation models and fused with point clouds using camera projection. Normalized graph cut is used to cluster points that have high affinity, and separate clusters with low affinity. Qualitative and Quantitative results show the application of 2D foundation models for 3D tasks is a promising future direction.
# Pipeline

Inputs to our algorithm are camera poses, camera intrinsics, color frames, depth frames, and scene point clouds. 

During preprocessing, we want to generate accurate and temporally consistent 2D segmentations of objects present in the video. We utilize Recognize-Anything Model to obtain semantic labels for prominent object in each frame. These labels are fed to Grounding-DINO to generate bounding boxes, which can be used to prompt Segment-Anything for fine grained segmentation. In order to improve temporal consistency, we use DEVA to obtain segmentation tracklets for each object. We now have segmentation tracklets for each prominent object in our video, along with its semantic labels (eg. chair, table).

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/ebda8c30-7e07-441c-9322-b232722d43eb" alt="Image 1" width="200" />
      <div>Frame 26 (RGB)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/c8119eb7-23b7-4ed7-8398-7405fcc3f9e8" alt="Image 2" width="200" />
      <div>Frame 27 (RGB)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/b05c3ccf-c7c5-42f1-82d1-2e53058c07a3" alt="Image 3" width="200" />
      <div>Frame 28 (RGB)</div>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/b13cbdd8-725d-4b63-83ae-265070650404" alt="Image 4" width="200" />
      <div>Frame 26 (RAM + GDINO + SAM)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/38d1f4f6-192f-4f81-b03f-410898223c6b" alt="Image 5" width="200" />
      <div>Frame 27 (RAM + GDINO + SAM)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/5c71978d-fdfb-480a-994a-bae9afcf2d6f" alt="Image 6" width="200" />
      <div>Frame 28 (RAM + GDINO + SAM)</div>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/bf150492-5f8f-46e2-9fbf-0364acbd7d79" alt="Image 7" width="200" />
      <div>Frame 26 (DEVA)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/1aea1290-b3ea-4829-9916-e2ccc1b2c1ce" alt="Image 8" width="200" />
      <div>Frame 27 (DEVA)</div>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/0fabd7b7-f735-4814-8611-eb3e23ee5be2" alt="Image 9" width="200" />
      <div>Frame 28 (DEVA)</div>
    </td>
  </tr>
</table>

Our main clustering algorithm is the normalized graph cut algorithm found [here](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf). The algorithm performs partitions on a graph, forming clusters of nodes with high-valued edges and separating nodes with low-valued edges. The normalization is due to the cluster value being formulated as a fraction of the total edges with the entire graph, as opposed to simply a sum of edge values within the cluster. Representing each point as a node in the graph is too computationally expensive. Therefore, a simpler but more efficient clustering is performed to cluster points based on similar geometric features to form super points. These super points will become the nodes of our graphs.

![superpoints](https://github.com/user-attachments/assets/cc35e4d2-2080-4051-8dd5-9c4ade766e6e)


# Preprocessing

Preprocessing is performed in the submodule `Tracking-Anything-with-DEVA`. Remember to initialize it with the following command:

```
git submodule update --init --recursive
```

##

```

```
