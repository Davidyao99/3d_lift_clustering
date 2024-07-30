from scipy.sparse import lil_matrix, dia_matrix
import scipy.spatial as spatial

from sklearn.cluster import spectral_clustering
from scipy.sparse.csgraph import connected_components

import numpy as np

import pdb

class CutEngine:

    def __init__(self, thres=0.3, size_limit=50, eigen_solver='lobpcg', eigen_tol=1e-12):

        self.next_cls_idx = 1
        self.norm_cut_thres = thres
        self.size_limit = size_limit
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol

        np.random.seed(1)

    # def perf_norm_cut(self):

    def get_diag(self, W):

        '''
        Creates diagonal matrix with values = col sum
        '''
        
        entries = W.sum(axis=0)
        D = dia_matrix((entries, 0), shape=W.shape).tocsc()

        return D

    def calc_cost(self, W, D, cluster):

        idxes1 = np.where(cluster == 0)[0]
        idxes2 = np.where(cluster == 1)[0]

        cut_cost = W[idxes1, :][:, idxes2].sum()

        assoc1 = D[idxes1, idxes1].sum()
        assoc2 = D[idxes2, idxes2].sum()

        if assoc1 == 0 or assoc2 == 0: # isolated points, cut them out
            return 0

        cost = cut_cost / assoc1 + cut_cost / assoc2

        return cost

    def normalized_cut(self, adj_matrix, pcd_coords, vis=False):

        """
        Entry point for performing normalized cut
        """
        adj_matrix = adj_matrix
        self.cluster = np.zeros(adj_matrix.shape[0], dtype=np.int32)
        self.cluster_vis = []
        self.pcd_coords = pcd_coords
        self._norm_cut(adj_matrix, np.arange(adj_matrix.shape[0]), 0, vis)
        
        return self.cluster, self.cluster_vis

    # def check_disconnect(self, cluster, idxes, W):

    #     print("checking disconnect")

    #     cluster_ids = np.unique(cluster)

    #     for cluster_id in cluster_ids:
            
    #         sel_pcd_idx = np.where(cluster == cluster_id)[0]
    #         sel_pcd = self.pcd_coords[idxes[sel_pcd_idx]]

    #         dst_matrix = spatial.distance_matrix(sel_pcd, sel_pcd)
    #         adj_matrix = dst_matrix <= 0.3
    #         n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

    #         if n_components != 1:
    #             cluster1 = sel_pcd_idx[np.where(labels == 0)[0]]
    #             cluster2 = sel_pcd_idx[np.where(labels == 1)[0]]
    #             a = W[cluster2]
    #             a = a[:,cluster1].A
    #             print(a) # check that the adjacency matrix is correct
    #         else:
    #             return None


    def _norm_cut(self, W, idxes, cls_idx, vis=False):

        """
        Performs normalized cut on matrix into 2 parttions. One partition remains as cls_idx while other takes on self.next_cls_idx
        Cost of cut is checked after partition and if below threshold, then parttion is accepted. Else partition is not done
        W: affinity matrix
        idxes: indices of points that W came from original matrix
        cls_idx: original cluster idx
        """
        if W.shape[0] <= self.size_limit:
            return 

        D = self.get_diag(W)

        clustering = spectral_clustering(W, n_clusters=2, eigen_solver=self.eigen_solver, eigen_tol=self.eigen_tol, assign_labels="discretize", random_state=0)

        cost = self.calc_cost(W, D, clustering)

        n_components, clustering_con = connected_components(csgraph=W, directed=False, return_labels=True)

        if cost < self.norm_cut_thres: # perform cluster and try subclustering only if cost < threshold

            idxes1 = np.where(clustering == 0)[0]
            idxes2 = np.where(clustering == 1)[0]

            self.cluster[idxes[idxes1]] = cls_idx
            self.cluster[idxes[idxes2]] = self.next_cls_idx

            self.cluster_vis.append(self.cluster.copy())

            # print(f"Number of clusters {self.next_cls_idx}", end='\r', flush=True)
            
            self.next_cls_idx += 1

            next_idxes2 = idxes[idxes2]
            self._norm_cut(W[idxes2, :][:, idxes2], next_idxes2, self.next_cls_idx-1, vis)

            next_idxes1 = idxes[idxes1]
            self._norm_cut(W[idxes1, :][:, idxes1], next_idxes1, cls_idx, vis)
