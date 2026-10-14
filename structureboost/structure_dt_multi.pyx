# cython: profile=False
# cython: language_level=3

"""Multiclass Decision Tree based on Discrete Graph structure"""
import graphs
import copy
import random
import warnings
import numpy as np
import scipy as sp
import pandas as pd
from libc.math cimport log as clog
from libc.math cimport isnan
from structure_dt import StructureDecisionTree
import structure_dt as stdt
from pdf_group import team_size
cimport numpy as np
np.import_array()
cimport cython
from cython.parallel cimport prange


class StructureDecisionTreeMulti(StructureDecisionTree):
    """Multi class Decision Tree using graphical structure.

    Uses Newton steps based on first and second derivatives of loss fn.
    """

    def __init__(self, feature_configs, feature_graphs, num_classes,
                 min_size_split=2,
                 max_depth=3, gamma=0, feat_sample_by_node=1,
                 reg_lambda=1):
        super().__init__(feature_configs=feature_configs,
                         min_size_split=min_size_split,
                         max_depth=max_depth, gamma=gamma,
                         feat_sample_by_node=feat_sample_by_node,
                         reg_lambda=reg_lambda,
                         feature_graphs=feature_graphs)
        self.num_classes = num_classes

    def _process_tree_node(self, curr_node):
        """StructureDecisionTree._process_tree_node with the numerical split search batched across features.

        Every random draw happens in the original order (features in order: a numerical feature's split points and
        its NA coin flip, a categorical feature's spanning trees), so results are unchanged. Only the per-row work
        of the numerical features, binning each row among the candidate splits and summing its gradients per bin,
        runs in one parallel pass over the features (numerical_bin_sums); the scoring that follows is the original
        NumPy code on those sums."""
        mask = curr_node['mask']
        X_train_node = self.X_train[mask]
        g_h_train_node = self.g_h_train[mask]
        num_dp = g_h_train_node.shape[0]
        curr_node['num_data_points'] = num_dp
        curr_node['node_summary_val'] = self._node_summary_gh(g_h_train_node)
        curr_node['node_index'] = self.node_counter
        self.node_counter += 1
        wrap_up_now = self._check_stopping_condition(num_dp, curr_node['depth'], g_h_train_node)
        if wrap_up_now:
            self._wrap_up_node(curr_node, g_h_train_node.shape[0])
            return None

        features_to_search = self._get_features_to_search()
        results, jobs = [], []
        for feature in features_to_search:
            fc = self.feature_configs[feature]
            if fc['feature_type'] == 'numerical':
                fvec = X_train_node[feature].to_numpy()
                interp_mode = fc.get('interp_mode', 'random')
                splits = self._get_numerical_splits(fvec, self.uv_dict[feature], interp_mode=interp_mode)
                if len(splits) == 0:
                    results.append(stdt._initialize_best_split_dict())
                    continue
                splits = np.ascontiguousarray(stdt._subset_splits(splits, fc), dtype=np.float64)
                has_na = bool(np.isnan(splits[-1]))
                coin = None if (has_na and len(splits) > 2) else int(random.random() < .5)
                jobs.append((len(results), np.ascontiguousarray(fvec, dtype=np.float64), splits, coin))
                results.append(None)
            else:
                results.append(self.evaluate_feature(fc, curr_node['feature_graphs'], feature, X_train_node,
                                                     g_h_train_node, self.uv_dict))
        if jobs:
            g_h = np.ascontiguousarray(g_h_train_node, dtype=np.float64)
            sums = numerical_bin_sums([j[1] for j in jobs], [j[2] for j in jobs], g_h, self.num_classes)
            for (slot, _, splits, coin), (g_sum_bins, h_sum_bins) in zip(jobs, sums):
                best_loss, best_spl_val, na_left, na_rnd = self._score_numerical_splits(g_sum_bins, h_sum_bins, splits, coin)
                results[slot] = {'loss_score': best_loss, 'left_split': best_spl_val, 'feature_type': 'numerical',
                                 'na_left': na_left, 'na_dir_random': na_rnd}

        best_split_dict = stdt._initialize_best_split_dict()
        for feature, best_split_for_feature in zip(features_to_search, results):
            if best_split_for_feature:
                best_split_for_feature['split_feature'] = feature
                if best_split_for_feature['loss_score'] < best_split_dict['loss_score']:
                    best_split_dict = best_split_for_feature
        if best_split_dict['loss_score'] < np.inf:
            self._execute_split(curr_node, best_split_dict, curr_node['feature_graphs'])
        else:
            self._wrap_up_node(curr_node, g_h_train_node.shape[0])

    def _score_numerical_splits(self, g_sum_bins, h_sum_bins, split_vec, coin):
        """The scoring half of _evaluate_numerical_splits, from precomputed bin sums; coin is the NA coin flip drawn
        in the original order (None when the NA branch below decides the side)."""
        has_na_vals = np.isnan(split_vec[-1])
        g_sum_total, g_sum_left, g_sum_right = get_left_right_sums_mc(g_sum_bins)
        h_sum_total, h_sum_left, h_sum_right = get_left_right_sums_mc(h_sum_bins)
        score_vec = (-1)*_get_gh_score_array_mc(g_sum_left, g_sum_right, h_sum_left, h_sum_right, self.gamma, self.reg_lambda)
        best_loss, best_split_val = stdt._get_best_vals(score_vec, split_vec)
        if has_na_vals and (len(split_vec) > 2):
            g_sum_left_nal = (g_sum_left + (g_sum_total - g_sum_left[-1,:]))[:-1,:]
            h_sum_left_nal = (h_sum_left + (h_sum_total - h_sum_left[-1,:]))[:-1,:]
            g_sum_right_nal = g_sum_total - g_sum_left_nal
            h_sum_right_nal = h_sum_total - h_sum_left_nal
            score_vec_nal = (-1)*_get_gh_score_array_mc(g_sum_left_nal, g_sum_right_nal, h_sum_left_nal, h_sum_right_nal,
                                                     self.gamma, self.reg_lambda)
            best_loss_nal, best_split_val_nal = stdt._get_best_vals(score_vec_nal, split_vec)
            if best_loss_nal < best_loss:
                return best_loss_nal, best_split_val_nal, 1, 0
            else:
                return best_loss, best_split_val, 0, 0
        else:
            return best_loss, best_split_val, coin, 1

    def _node_summary_gh(self, g_h_mat):
        if (g_h_mat.shape[0] == 0):
            return np.zeros(self.num_classes)
        else:
            g_vec = np.sum(g_h_mat[:,:self.num_classes], axis=0)
            h_vec = np.sum(g_h_mat[:,self.num_classes:2*self.num_classes], axis=0)
            out_val = -(g_vec/(h_vec+self.reg_lambda))
            return(out_val)

    def _evaluate_numerical_splits(self, feature_vec, g_h_mat,
                                   split_vec):

        has_na_vals = np.isnan(split_vec[-1])
        bin_result_vec = np.searchsorted(split_vec,
                                         feature_vec,
                                         side='right').astype(np.int32)
        g_sum_bins, h_sum_bins = get_bin_sums_c_mc(g_h_mat,
                                                bin_result_vec,
                                                len(split_vec)+1, self.num_classes)
        g_sum_total, g_sum_left, g_sum_right = get_left_right_sums_mc(g_sum_bins)
        h_sum_total, h_sum_left, h_sum_right = get_left_right_sums_mc(h_sum_bins)
        score_vec = (-1)*_get_gh_score_array_mc(g_sum_left, g_sum_right,
                                             h_sum_left, h_sum_right,
                                             self.gamma, self.reg_lambda)

        best_loss, best_split_val = stdt._get_best_vals(score_vec, split_vec)
        if has_na_vals and (len(split_vec) > 2):
            g_sum_left_nal = (g_sum_left + (g_sum_total - g_sum_left[-1,:]))[:-1,:]
            h_sum_left_nal = (h_sum_left + (h_sum_total - h_sum_left[-1,:]))[:-1,:]
            g_sum_right_nal = g_sum_total - g_sum_left_nal
            h_sum_right_nal = h_sum_total - h_sum_left_nal

            score_vec_nal = (-1)*_get_gh_score_array_mc(g_sum_left_nal,
                                                     g_sum_right_nal,
                                                     h_sum_left_nal,
                                                     h_sum_right_nal,
                                                     self.gamma, self.reg_lambda)
            best_loss_nal, best_split_val_nal = stdt._get_best_vals(score_vec_nal,
                                                               split_vec)
            if best_loss_nal < best_loss:
                # NAs go left by design
                return best_loss_nal, best_split_val_nal, 1, 0
            else:
                # NAs go right by design
                return best_loss, best_split_val, 0, 0
        else:
            # If no NAs at training, randomly choose side for NAs at this node
            # adjust here if we want coin flip to be other than .5
            return best_loss, best_split_val, int(random.random() < .5), 1


    def get_gh_val_mat(self, feature_vec_node,g_h_train_node,
                       max_num_vertices):
        g_h_val_arr = np.zeros((max_num_vertices,2*self.num_classes))
        g_h_val_arr = get_g_h_feature_sum_matrix(
                                        feature_vec_node.astype(np.int32),
                                        g_h_train_node,
                                        g_h_val_arr, self.num_classes)
        return g_h_val_arr

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def update_g_h_accum(self, np.ndarray[double, ndim=2] g_h_accum_array,
        long lni, long lvi):
        ## TODO: make this more efficient (cython)
        cdef long j
        cdef long limit = 2*self.num_classes
        for j in range(2*self.num_classes):
            g_h_accum_array[lni,j] +=  g_h_accum_array[lvi,j]
        return g_h_accum_array

    def get_loss_in_span_tree(self, feature_vec_node, g_h_train_node,
                              g_h_accum_array, g_h_sum, leaf_vertex_ind,
                              left_feat_values, feature_type):
        g_left = g_h_accum_array[leaf_vertex_ind,:self.num_classes]
        h_left = g_h_accum_array[leaf_vertex_ind,self.num_classes:2*(self.num_classes)]
        g_sum = g_h_sum[:self.num_classes]
        h_sum = g_h_sum[self.num_classes:2*self.num_classes]
        curr_loss = _get_gh_score_num_mc(g_left, g_sum-g_left,
                                      h_left, h_sum-h_left,
                                      self.gamma, self.reg_lambda)
        return(curr_loss)


    def get_score_of_split(self, g_h_train_node, mask_left, g_h_sum):
        g_h_masked = g_h_train_node[mask_left,:]
        g_h_masked_sum = np.sum(g_h_masked, axis=0)
        g_left = g_h_masked_sum[:self.num_classes]
        h_left = g_h_masked_sum[self.num_classes:(2*self.num_classes)]
        g_sum = g_h_sum[:self.num_classes]
        h_sum = g_h_sum[self.num_classes:(2*self.num_classes)]
        curr_loss = _get_gh_score_num_mc(g_left, g_sum-g_left,
                                      h_left, h_sum-h_left,
                                      self.gamma, self.reg_lambda)
        return(curr_loss)


    def predict(self, X_test):
        col_list = list(X_test.columns)
        column_to_int_dict = {col_list[i]: i for i in range(len(col_list))}
        return self.get_prediction(self.dec_tree, X_test.to_numpy(),
                              column_to_int_dict)


    def get_prediction(self, tree_node, X_te, dict col_to_int_dict):
        cdef np.ndarray[np.int32_t] ind_subset_left, ind_subset_right
        cdef long vec_len, lsize
        cdef np.ndarray[double,ndim=2] next_vec

        if tree_node['node_type'] == 'leaf':
            return stdt.get_node_response_leaf(X_te.shape[0], tree_node)
        else:
            split_bool = stdt.get_node_response_df_val(X_te, tree_node, col_to_int_dict)
            vec_len = len(split_bool)
            next_vec = np.zeros((vec_len,self.num_classes))
            ind_subset_left = np.empty(vec_len, dtype=np.int32)
            ind_subset_right = np.empty(vec_len, dtype=np.int32)
            ind_subset_left, ind_subset_right, lsize = stdt.separate_indices(
                                                    ind_subset_left,
                                                    ind_subset_right,
                                                    split_bool.astype(np.int32),
                                                    vec_len)
            ind_subset_left = ind_subset_left[:lsize]
            ind_subset_right = ind_subset_right[:(vec_len-lsize)]

            if lsize > 0:
                next_vec[ind_subset_left,:] = self.get_prediction(tree_node['left_child'],
                                                           X_te[
                                                           ind_subset_left, :],
                                                           col_to_int_dict)

            if lsize < vec_len:
                next_vec[ind_subset_right,:] = self.get_prediction(
                                                tree_node['right_child'],
                                                X_te[ind_subset_right, :],
                                                col_to_int_dict)
            return next_vec


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_g_h_feature_sum_matrix(np.ndarray[np.int32_t] feature_vec_node,
                               np.ndarray[double, ndim=2] g_h_train_node,
                               np.ndarray[double, ndim=2] g_h_val_arr,
                               long num_classes):
    cdef long i, ind
    cdef long array_size = len(feature_vec_node)
    for i in range(array_size):
        for j in range(2*num_classes):
            ind = feature_vec_node[i]
            g_h_val_arr[ind,j] += g_h_train_node[i,j]
    return g_h_val_arr


def _get_gh_score_num_mc(np.ndarray[double] g_left,
                      np.ndarray[double] g_right,
                      np.ndarray[double] h_left,
                      np.ndarray[double] h_right,
                      double gamma, double reg_lambda, double tol=1e-12):
    loss_val = -1.0 * np.sum(.5*(((g_left*g_left)/(h_left+reg_lambda)) +
                       ((g_right*g_right)/(h_right+reg_lambda)) -
                   (((g_left + g_right)*(g_left + g_right)) /
                    (h_left + h_right+reg_lambda)))-gamma)
    if loss_val >= -tol:
        loss_val = np.inf
    return(loss_val)


def _get_gh_score_array_mc(np.ndarray[double, ndim=2] g_left,
                        np.ndarray[double, ndim=2] g_right,
                        np.ndarray[double, ndim=2] h_left,
                        np.ndarray[double, ndim=2] h_right,
                        double gamma, double reg_lambda):
    return(np.sum(.5*(((g_left*g_left)/(h_left+reg_lambda)) +
               ((g_right*g_right)/(h_right+reg_lambda)) -
               (((g_left+g_right) * (g_left+g_right)) /
                (h_left+h_right+reg_lambda)))-gamma, axis=1))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_bin_sums_c_mc(np.ndarray[double, ndim=2] g_h_mat,
                   np.ndarray[np.int32_t] bin_result_vec,
                   long out_vec_size, long num_classes):
    cdef int i,j
    cdef int m = bin_result_vec.shape[0]

    cdef np.ndarray[double, ndim=2] g_sum_bins = np.zeros((out_vec_size, num_classes))
    cdef np.ndarray[double, ndim=2] h_sum_bins = np.zeros((out_vec_size, num_classes))

    for i in range(m):
        for j in range(num_classes):
            g_sum_bins[bin_result_vec[i],j] += g_h_mat[i,j]
            h_sum_bins[bin_result_vec[i],j] += g_h_mat[i,num_classes+j]
    return g_sum_bins, h_sum_bins

def get_left_right_sums_mc(bin_sums):
    sum_left = np.cumsum(bin_sums, axis=0)
    sum_total = sum_left[-1,:]
    sum_left = sum_left[:-1,:]
    sum_right = sum_total - sum_left
    return sum_total, sum_left, sum_right







@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long _upper_bound_nan(const double[::1] arr, long n, double key) noexcept nogil:
    """np.searchsorted(arr, key, side='right') with NumPy's ordering (NaN sorts after every number): the number of
    elements of the sorted arr that are <= key in that ordering."""
    cdef long lo = 0, hi = n, mid
    cdef double a
    cdef bint key_nan = key != key
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        a = arr[mid]
        # key < a in NumPy's order: a is NaN and key is not, or both numbers and key < a
        if (a != a and not key_nan) or (key < a):
            hi = mid
        else:
            lo = mid + 1
    return lo


@cython.boundscheck(False)
@cython.wraparound(False)
def numerical_bin_sums(list feature_vecs, list split_vecs, double[:, ::1] g_h_mat, long num_classes):
    """For each numerical feature j: bin every row by searchsorted(split_vecs[j], x, side='right') and sum its g and h
    per bin and class, exactly as get_bin_sums_c_mc after np.searchsorted (rows added in order). Features run in
    parallel. Returns [(g_sum_bins, h_sum_bins)] with shape (len(split_vecs[j]) + 1, num_classes) each."""
    cdef long nf = len(feature_vecs), n = g_h_mat.shape[0], j, i, c, b, ns
    cdef np.ndarray[np.int64_t] offs = np.zeros(nf + 1, dtype=np.int64)
    cdef np.ndarray[np.int64_t] soffs = np.zeros(nf + 1, dtype=np.int64)
    for j in range(nf):
        offs[j + 1] = offs[j] + (len(split_vecs[j]) + 1) * num_classes
        soffs[j + 1] = soffs[j] + len(split_vecs[j])
    cdef np.ndarray[double, ndim=2] X = np.empty((nf, n), dtype=np.float64)
    for j in range(nf):
        X[j, :] = feature_vecs[j]
    cdef np.ndarray[double] S = np.concatenate(split_vecs).astype(np.float64) if nf else np.empty(0)
    cdef np.ndarray[double] G = np.zeros(offs[nf], dtype=np.float64)
    cdef np.ndarray[double] H = np.zeros(offs[nf], dtype=np.float64)
    cdef double[:, ::1] Xv = X
    cdef double[::1] Sv = S, Gv = G, Hv = H
    cdef long[::1] ov = offs, sv = soffs
    cdef int nth = team_size(nf * n * num_classes)
    with nogil:
        for j in prange(nf, schedule='dynamic', num_threads=nth):
            for i in range(n):
                b = _upper_bound_nan(Sv[sv[j]:sv[j + 1]], sv[j + 1] - sv[j], Xv[j, i])
                for c in range(num_classes):
                    Gv[ov[j] + b * num_classes + c] += g_h_mat[i, c]
                    Hv[ov[j] + b * num_classes + c] += g_h_mat[i, num_classes + c]
    out = []
    for j in range(nf):
        ns = soffs[j + 1] - soffs[j] + 1
        out.append((G[offs[j]:offs[j + 1]].reshape(ns, num_classes), H[offs[j]:offs[j + 1]].reshape(ns, num_classes)))
    return out
