import numpy as np
from .config import FIDL_RS
from .util import  increment_mask, decrement_mask


class MaskOptimizer:
    def __init__(self, mask_batch_size, mask_shape, flip_size,
                 frac_of_rand_masks=0.5, s_guess=None):
        self.mask_shape = mask_shape
        self.data_size = np.prod(mask_shape)
        self.mask_batch_size = mask_batch_size
        self.frac_of_rand_masks = frac_of_rand_masks
        self.flip_size = flip_size
        self.s_mean = int(self.data_size / 2) if s_guess is None else s_guess
        self.s_mean_frac = 1.0 * self.s_mean / self.data_size

    def get_random_batch_of_masks(self, batch_size):
        s_list = self.get_s(batch_size, mean=self.s_mean)
        m = np.empty((batch_size,)+self.data_size)
        for i,s in enumerate(s_list):
            m[i] = increment_mask(m[i],s)
        return m

    def get_s(self, n_samples, mean=None):
        return FIDL_RS.binomial(self.data_size-self.flip_size,
                                self.s_mean_frac if mean is None else
                                (mean if mean < 1 else 1.0 * mean / self.data_size), n_samples).astype('int')
    @staticmethod
    def advance_mask(importances,m,flip_size):
        flip_idcs = np.argpartition(importances,flip_size)[:flip_size]
        m[flip_idcs] = 1+ np.negative(m[flip_idcs])
        return m

    def get_new_optimal_mask(self, pfs_base_model, step,
                             max_n_features_in_mask=None,
                             return_n_features=False,
                             return_change_hist = False,
                             record_hist_to_prevent_loops = 3,
                             max_iters=100
                             ):
        assert len(pfs_base_model.inputs[0].shape.as_list()) == 2
        max_n_features = pfs_base_model.inputs[0].shape[1]
        if max_n_features_in_mask is None:
            max_n_features_in_mask = max_n_features # no limit
        m_opt = np.zeros((1, max_n_features),dtype='float')
        n_features = 0
        iter = 0
        hist={'masks':[],'chosen_features':[]}
        record_hist = np.zeros((record_hist_to_prevent_loops, max_n_features))
        record_hist_idx = 0
        while True:
            importances = pfs_base_model.predict(m_opt)[0]
            if n_features == max_n_features_in_mask or np.all(importances > 0):
                # found the optimal set
                break

            if n_features + step <= max_n_features_in_mask:
                current_step = step
            else:
                current_step = max_n_features_in_mask - n_features
            best_choices = np.argpartition(importances,current_step)[:current_step] # for minimization
            print(n_features,"Choices: ", best_choices,"imps: ",np.round(importances[best_choices],3))
            m_opt[:,best_choices] = 1.0 - m_opt[:,best_choices] # flips 0->1 and 1->0
            n_added_features = np.count_nonzero(m_opt[:,best_choices])
            n_features += n_added_features
            n_features -= current_step - n_added_features

            if return_change_hist:
                hist['masks'].append(np.copy(m_opt))
                hist['chosen_features'].append(best_choices)

            if iter>=max_iters:
                print("Optimized mask stopped for iter:",iter,"out of",max_iters)
                break
            else:
                iter+=1

            if np.where((record_hist == m_opt[0]).all(axis=1))[0].size>0:
                print(np.where((record_hist == m_opt[0]).all(axis=1))[0])
                print("Optimized mask stopped for hist item:",np.where((record_hist == m_opt[0]).all(axis=1))[0])
                break
            if record_hist_to_prevent_loops > 0:
                #if(len(record_hist)==record_hist_to_prevent_loops):
                #    record_hist.pop()
                #record_hist.insert(0,m_opt)
                record_hist[record_hist_idx] = m_opt
                record_hist_idx = (record_hist_idx + 1) % record_hist_to_prevent_loops
            print("M_opt:",np.nonzero(m_opt[0])[0])

        if return_change_hist is False:
            if return_n_features is True:
                return m_opt,n_features
            else:
                return m_opt
        else:
            if return_n_features is True:
                return m_opt, n_features,hist
            else:
                return m_opt,hist

    @staticmethod
    def flip_masks(m, n_flips, with_repetitions=True):
        n_masks, mask_size = m.shape
        if with_repetitions == True:
            flip_idx_1 = FIDL_RS.randint(0, mask_size, (n_masks, n_flips))
        else:
            # see implementation at:
            # https://stackoverflow.com/questions/35572381/generate-large-number-of-random-card-decks-numpy/35572771#35572771
            raise NotImplementedError("Mask flips with no repetitions not implemented.")
        flip_idx_0 = np.tile(np.arange(n_masks)[:, None], (1, n_flips))
        m[flip_idx_0, flip_idx_1] = np.logical_not(m[flip_idx_0, flip_idx_1])
        return m

    @staticmethod
    def get_random_masks(shape, frac_of_1, dtype='uint'):
        """
        :param shape:
        :param frac_of_1:
        :return: can return masks with bigger size (more 1s) as this is a random function
        """
        return FIDL_RS.binomial(1, frac_of_1, size=np.prod(shape)).reshape(shape).astype(dtype)
