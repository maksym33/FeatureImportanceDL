import numpy as np
import tensorflow as tf


class MaskOptimizer:
    def __init__(self, mask_batch_size, data_shape, unmasked_data_size,perturbation_size,
                 frac_of_rand_masks=0.5, epoch_condition=1000 ):
        self.data_shape = data_shape
        self.unmasked_data_size = unmasked_data_size
        self.data_size = np.zeros(data_shape).size
        self.mask_history = []
        self.raw_mask_history = []
        self.loss_history = []
        self.epoch_counter = 0
        self.mask_batch_size = mask_batch_size
        self.frac_of_rand_masks = frac_of_rand_masks
        self.epoch_condition = epoch_condition
        self.perturbation_size = perturbation_size
        self.max_optimization_iters = 5
        self.step_count_history = []

    def gradient(model, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            # loss_mask_size = (tf.norm(x_tensor,ord=2,axis=1))
            loss_model = model(x_tensor)
            loss = loss_model  # +0.001*loss_mask_size#*loss_mask_size
        return t.gradient(loss, x_tensor).numpy(), loss_model

    def new_get_mask_from_grads(grads, unmasked_size, mask_size):
        m_opt = np.zeros(shape=mask_size)
        top_arg_grad = np.argpartition(grads, -unmasked_size)[-unmasked_size:]
        m_opt[top_arg_grad] = 1
        return m_opt

    def new_get_m_opt(model, unmasked_size):
        input_img = np.ones(shape=model.layers[0].output_shape[0][1:])[None, :] / 2  # define an initial random image
        grad, loss = MaskOptimizer.gradient(model, input_img)
        grad = np.negative(np.squeeze(grad))  # change sign
        m_opt = MaskOptimizer.new_get_mask_from_grads(grad, unmasked_size, model.layers[0].output_shape[0][1:])
        return m_opt

    def new_check_for_opposite_grad(m_opt_grad, m_opt_indexes):
        m_opt_grad_cp = np.copy(m_opt_grad[m_opt_indexes])
        m_opt_arg_opposite_grad = np.argwhere(m_opt_grad_cp < 0)
        return m_opt_indexes[m_opt_arg_opposite_grad]

    def new_check_loss_for_opposite_indexes(model, m_opt, min_index, max_index, opposite_indexes):
        m_opt_changed = False
        m_opt_loss = model.predict(m_opt[None, :])
        for ind in opposite_indexes:
            m_new_opt = np.copy(m_opt)
            m_new_opt[max_index] = 1
            m_new_opt[ind] = 0
            m_new_opt_loss = model.predict(m_new_opt[None, :])
            if m_new_opt_loss < m_opt_loss:
                # print("Changed i "+str(max_index)+" from 0->1 and"+str(ind)+" from 1->0.")
                return True, m_new_opt
        return False, m_opt

    def new_check_for_likely_change(model, m_opt, min_index, max_index, m_opt_grad):
        m_opt_changed = False
        m_opt_loss = np.squeeze(model.predict(m_opt[None, :]))
        not_m_opt_indexes = np.argwhere(m_opt == 0)
        max_index = not_m_opt_indexes[np.argmax(m_opt_grad[not_m_opt_indexes])]
        m_new_opt = np.copy(m_opt)
        m_new_opt[min_index] = 0
        m_new_opt[max_index] = 1
        m_new_opt_loss = np.squeeze(model.predict(m_new_opt[None, :]))
        # print("New proposed likely m_opt:   ")
        # print(str(m_new_opt))
        # print("Losses old/new: "+str(m_opt_loss)+"   "+str(m_new_opt_loss))
        #print(m_new_opt_loss,"  ",m_opt_loss)
        if (m_new_opt_loss < m_opt_loss):
            return True, m_new_opt
        else:
            return False, m_opt

    def get_opt_mask(self, unmasked_size, model, steps=None):
        m_opt = MaskOptimizer.new_get_m_opt(model, unmasked_size)
        repeat_optimization = True
        step_count = 0
        if steps is None:
            steps = self.max_optimization_iters
        while (repeat_optimization == True and step_count < steps):
            # print(step_count)
            # print(np.squeeze(np.argwhere(m_opt==1)))
            step_count += 1
            repeat_optimization = False
            m_opt_grad, m_opt_loss = MaskOptimizer.gradient(model, m_opt[None, :])
            m_opt_grad = -np.squeeze(m_opt_grad)
            m_opt_indexes = np.squeeze(np.argwhere(m_opt == 1))
            # print(m_opt_indexes)
            # print(m_opt_grad[m_opt_indexes])
            # min_index = MaskOptimizer.new_get_min_opt_grad(m_opt_grad,m_opt_indexes)
            min_index = m_opt_indexes[np.argmin(m_opt_grad[m_opt_indexes])]
            not_m_opt_indexes = np.squeeze(np.argwhere(m_opt == 0))
            # print(m_opt_grad[not_m_opt_indexes])
            if (not_m_opt_indexes.size > 1):
                max_index = not_m_opt_indexes[np.argmax(m_opt_grad[not_m_opt_indexes])]
            elif (not_m_opt_indexes.size == 1):
                max_index = not_m_opt_indexes
            # print(min_index)
            # print(max_index)
            opposite_indexes = MaskOptimizer.new_check_for_opposite_grad(m_opt_grad, m_opt_indexes)
            # print("opposite indexes: "+str(opposite_indexes))
            repeat_optimization, m_opt = MaskOptimizer.new_check_loss_for_opposite_indexes(model, m_opt, min_index,
                                                                                           max_index,
                                                                                           opposite_indexes)
            if (repeat_optimization == True):
                # print("Repeating due negative indexes for unmasked inputs")
                continue
            repeat_optimization, m_opt = MaskOptimizer.new_check_for_likely_change(model, m_opt, min_index,
                                                                                   max_index, m_opt_grad)
            if (repeat_optimization == True):
                # print("replacing lowest gradient unmasked index with highest gradient masked index gave better results")
                continue
        self.step_count_history.append(step_count - 1)
        return m_opt

    def check_condiditon(self):
        if (self.epoch_counter >= self.epoch_condition):
            return True
        else:
            return False

    def get_random_masks(self):
        masks_zero = np.zeros(shape=(self.mask_batch_size, self.data_size - self.unmasked_data_size))
        masks_one = np.ones(shape=(self.mask_batch_size, self.unmasked_data_size))
        masks = np.concatenate([masks_zero, masks_one], axis=1)
        masks_permuted = np.apply_along_axis(np.random.permutation, 1, masks)
        return masks_permuted

    def get_perturbed_masks(mask, n_masks, n_times=1):
        masks = np.tile(mask, (n_masks, 1))
        for i in range(n_times):
            masks = MaskOptimizer.perturb_masks(masks)
        return masks

    def perturb_masks(masks):
        def perturb_one_mask(mask):
            where_0 = np.nonzero(mask - 1)[0]
            where_1 = np.nonzero(mask)[0]
            i0 = np.random.randint(0, len(where_0), 1)
            i1 = np.random.randint(0, len(where_1), 1)
            mask[where_0[i0]] = 1
            mask[where_1[i1]] = 0
            return mask

        n_masks = len(masks)
        masks = np.apply_along_axis(perturb_one_mask, 1, masks)
        return masks

    def get_new_mask_batch(self, model, best_performing_mask,  gen_new_opt_mask):
        self.epoch_counter += 1
        random_masks = self.get_random_masks()
        if (gen_new_opt_mask):
            self.mask_opt = self.get_opt_mask(self.unmasked_data_size, model)
            # print("Opt: "+str(np.squeeze(np.argwhere(self.mask_opt==1))))
            # print("Perf: "+str(np.squeeze(np.argwhere(best_performing_mask==1))))
        if (self.check_condiditon() is True):
            index = int(self.frac_of_rand_masks * self.mask_batch_size)

            random_masks[index] = self.mask_opt
            random_masks[index + 1] = best_performing_mask
            random_masks[index + 2:] = MaskOptimizer.get_perturbed_masks(random_masks[index],
                                                                         self.mask_batch_size - (index + 2),
                                                                         self.perturbation_size)
            # print("mask batch_size: "+str(self.mask_batch_size))
            # print("index: "+str(index))
            # print("left: "+str(self.mask_batch_size - (index+1)))
            # [print(np.squeeze(np.argwhere(i==1))) for i in random_masks]
        return random_masks

    def get_mask_weights(self, tiling):
        w = np.ones(shape=self.mask_batch_size)
        index = int(self.frac_of_rand_masks * self.mask_batch_size)
        w[index] = 5
        w[index + 1] = 10
        return np.tile(w, tiling)
