from numba import njit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import torchvision.transforms as transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

@njit
def compute_projected_pts(pts, cam_intr): # project 3D points to 2D image plane
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts

@njit
def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.1):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool_)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    return visibility_mask

def visualize_features(features):
    '''
    Inputs:
    features (np.array): (... , dim_of_features) array of features to be converted to rgb for visualization in last dimension
    '''

    orig_shape = features.shape[:-1]
    features = features.reshape((-1, features.shape[-1]))
    pipeline = Pipeline([('scaling', StandardScaler(with_mean=True, with_std=False)), ('pca', PCA(n_components=3))])

    rgb_features = pipeline.fit_transform(features)
    rgb_features = (rgb_features - rgb_features.min()) / (rgb_features.max() - rgb_features.min())

    rgb_features = rgb_features.reshape((*orig_shape, 3))

    return rgb_features

def sort_paths(dir):
    """
    returns files in a directory sorted by their names numerically
    """

    files = os.listdir(dir)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    return files

def dino_transforms(
    resize_size,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean = IMAGENET_DEFAULT_MEAN,
    std = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


    