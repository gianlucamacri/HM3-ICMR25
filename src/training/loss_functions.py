import torch
from typing import Optional

class LossContrastive:
    def __init__(self, patience=15, delta=.001, logger=None, use_wandb=False):
        self.train_losses = []
        self.validation_losses = []
        self.counter_patience = 0
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.logger = logger
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb # may be useful to log some values

    def on_epoch_end(self, loss, train=True):
        if train:
            self.train_losses.append(loss)
        else:
            self.validation_losses.append(loss)

    def get_loss_trend(self):
        return self.train_losses, self.validation_losses

    def calculate_loss(self, pairwise_distances, margin=.25, margin_tensor=None, categories=None, within_category_margin:Optional[float]=None, new_positive_samples:Optional[int]=None, allow_sampling_replacement_of_new_positives = False, loss_components_weights=[0.5,0.5], filter_out_hard_negatives=False):
        # categories may be either null or a vector of elements that are all equal
        # within_category_margin is considered only if categories is not None
        # new_positive_samples is used to sample the new positives within a class for the second part of the loss where the positive is sampled from the elements within the same class of the anchor and the negatives are the elements of the batch that belong to a different class
        # note that new_positive_samples is usable only if the within_category_samples is not set (for now)
        # loss_components_weights is a enumerable that contains the weights for the different loss components if more than 2 are required  

        assert not (new_positive_samples is not None and within_category_margin is not None), f'new_positive_samples and within_category_margin cannot both be set as for now'
        assert not new_positive_samples or categories is not None, f'to use new_positive_samples teh categories vector should be provided'

        batch_size = pairwise_distances.shape[0]
        diag = pairwise_distances.diag().view(batch_size, 1)
        pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
        d1 = diag.expand_as(pairwise_distances) # matrix of shape of pairwise_distances with each row having i having all the same elements originally at (i,i)

        # Assuming `categories` is a 1D tensor containing the category labels
        different_cat_fltr = (categories.unsqueeze(0) != categories.unsqueeze(1)).float() if categories is not None else (torch.ones((batch_size,batch_size))-torch.eye(batch_size))
        different_cat_fltr = different_cat_fltr.to(pairwise_distances.device)
        different_class_balancing_denominator = different_cat_fltr.sum()
        if different_class_balancing_denominator == 0:
            different_class_balancing_denominator = 1 # to avoid a non zero division
            self.logger.info(f'the filter matrix is full of zeros, hence the loss for this batch will be 0')

        balancing_denominator = different_class_balancing_denominator
        ### normal triplet loss (or categorical triplet loss when categories is not None) 
        ## s2t (or vice versa)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            original_cost_s = (margin_tensor + pairwise_distances - d1).clamp(min=0)
        else:
            original_cost_s = (margin + pairwise_distances - d1).clamp(min=0)
        if filter_out_hard_negatives:
            keep_fltr_s = (d1 > pairwise_distances).int()
            original_cost_s = original_cost_s*keep_fltr_s # set to 0 all the hard negatives
            balancing_denominator = (different_cat_fltr*keep_fltr_s).sum()
        cost_s = original_cost_s*different_cat_fltr
        cost_s = cost_s.masked_fill(pos_masks, 0)
        cost_s = cost_s / balancing_denominator
        cost_s = cost_s.sum()

        ## t2s (the other one)
        d2 = diag.t().expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            original_cost_d = (margin_tensor + pairwise_distances - d2).clamp(min=0)
        else:
            original_cost_d = (margin + pairwise_distances - d2).clamp(min=0)
        if filter_out_hard_negatives:
            keep_fltr_d = (d2 > pairwise_distances).int()
            original_cost_d = original_cost_d*keep_fltr_d # set to 0 all the hard negatives
            balancing_denominator = (different_cat_fltr*keep_fltr_d).sum()
        cost_d = original_cost_d*different_cat_fltr
        cost_d = cost_d.masked_fill(pos_masks, 0)
        cost_d = cost_d / balancing_denominator
        cost_d = cost_d.sum()

        cost  = (cost_s + cost_d) / 2

        if categories is not None and within_category_margin is not None and loss_components_weights[1] > 0:
            assert sum(loss_components_weights) == 1, f'the loss weights does not sum to 1'

            ### within class loss
            same_cat_fltr = (categories.unsqueeze(0) == categories.unsqueeze(1)).float()

            balancing_denominator_c = same_cat_fltr.sum() - same_cat_fltr.shape[0] # remove elements on the diagonal
            assert (balancing_denominator_c + different_class_balancing_denominator + same_cat_fltr.shape[0]) == (same_cat_fltr.shape[0]**2),f'some rebalancing is wrong'
            if balancing_denominator_c == 0:
                balancing_denominator_c = 1 # to avoid a non zero division
                self.logger.info(f'the filter matrix within the categories is full of zeros, hence the loss for this batch will be 0')
            

            assert margin_tensor is None, f'margin tensor not supported when within_category_margin is provided'
            original_cost_s_c = (within_category_margin + pairwise_distances - d1).clamp(min=0)
            if filter_out_hard_negatives:
                original_cost_s_c = original_cost_s_c*keep_fltr_s # set to 0 all the hard negatives
                balancing_denominator_c = (same_cat_fltr*keep_fltr_s).masked_fill(pos_masks, 0).sum()
            cost_s_c = original_cost_s_c*same_cat_fltr
            cost_s_c = cost_s_c.masked_fill(pos_masks, 0) # remove elements on the diagonal
            cost_s_c = cost_s_c / balancing_denominator_c
            cost_s_c = cost_s_c.sum()

            original_cost_d_c = (within_category_margin + pairwise_distances - d2).clamp(min=0)
            if filter_out_hard_negatives:
                original_cost_d_c = original_cost_d_c*keep_fltr_d # set to 0 all the hard negatives
                balancing_denominator_c = (same_cat_fltr*keep_fltr_d).masked_fill(pos_masks, 0).sum()
            cost_d_c = original_cost_d_c*same_cat_fltr
            cost_d_c = cost_d_c.masked_fill(pos_masks, 0) # remove elements on the diagonal
            cost_d_c = cost_d_c / balancing_denominator_c
            cost_d_c = cost_d_c.sum()

            cost =  cost * loss_components_weights[0] + (cost_s_c + cost_d_c) * loss_components_weights[1] / 2

        elif categories is not None and new_positive_samples is not None:
            assert sum(loss_components_weights) == 1, f'the loss weights does not sum to 1'

            same_cat_fltr = (categories.unsqueeze(0) == categories.unsqueeze(1)).float()
            
            singletons_row_fltr = same_cat_fltr.sum(dim=1) == 1
            singletons_fltr = singletons_row_fltr.expand_as(pairwise_distances)
             
            same_cat_fltr_w_singletons = same_cat_fltr- (pos_masks.float()-singletons_fltr.float()).clamp(min=0) # remove the exact matches to avoid sampling them, but keep the lines with a singleton class
            
            balancing_denominator = different_class_balancing_denominator - singletons_row_fltr.sum()*(batch_size-1) # remove contribution of the singleton rows

            
            if not allow_sampling_replacement_of_new_positives:
                assert min(same_cat_fltr_w_singletons.sum(dim=1).min().item(), 1)  >= new_positive_samples, f'the current batch has some class with less than new_positives_samples' 
            new_positve_idxs = torch.multinomial(same_cat_fltr_w_singletons, new_positive_samples,replacement=allow_sampling_replacement_of_new_positives) # todo check/manage of the elements 
            costs = torch.zeros(new_positive_samples)

            for i in range(new_positive_samples): # maybe this could be improved to use a vectorized operation instead
                indices = new_positve_idxs[:,i].unsqueeze(1)
                ps = pairwise_distances.gather(1, indices).expand_as(pairwise_distances) # distance anchor selected positive

                ## s2t (or vice versa)
                if margin_tensor is not None:
                    margin_tensor = margin_tensor.to(pairwise_distances.device)
                    cost_s = (margin_tensor + pairwise_distances - ps).clamp(min=0)
                else:
                    cost_s = (margin + pairwise_distances - ps).clamp(min=0)
                if filter_out_hard_negatives:
                    keep_fltr = (ps > pairwise_distances).int()
                    cost_s = cost_s*keep_fltr # set to 0 all the hard negatives
                    balancing_denominator = (different_cat_fltr*keep_fltr).sum() - singletons_row_fltr.sum()*(batch_size-1) # remove contribution of the singleton rows
                cost_s = cost_s*different_cat_fltr*singletons_fltr.float()
                cost_s = cost_s / balancing_denominator
                cost_s = cost_s.sum()

                ## t2s (the other one)
                ps = pairwise_distances.t().gather(1, indices).expand_as(pairwise_distances) # distance anchor selected positive
                if margin_tensor is not None:
                    margin_tensor = margin_tensor.to(pairwise_distances.device)
                    cost_d = (margin_tensor + pairwise_distances - ps).clamp(min=0)
                else:
                    cost_d = (margin + pairwise_distances - ps).clamp(min=0)
                if filter_out_hard_negatives:
                    keep_fltr = (ps > pairwise_distances).int()
                    cost_d = cost_d*keep_fltr # set to 0 all the hard negatives
                    balancing_denominator = (different_cat_fltr*keep_fltr).sum() - singletons_row_fltr.sum()*(batch_size-1) # remove contribution of the singleton rows
                cost_d = cost_d*different_cat_fltr*singletons_fltr.float()
                cost_d = cost_d / balancing_denominator
                cost_d = cost_d.sum()

                costs[i] = (cost_s + cost_d)/2

            cost = cost*loss_components_weights[0] + costs.mean()*loss_components_weights[1]
                
        return cost

    def is_val_improving(self):
        score = -self.validation_losses[-1] if self.validation_losses else None

        if score and self.best_score and self.logger:
            self.logger.info(f'epoch: {len(self.validation_losses)} score: {-score} best_score: {-self.best_score} counter: {self.counter_patience}')

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter_patience += 1
            if self.counter_patience >= self.patience:
                return False
        else:
            self.best_score = score
            self.counter_patience = 0
        return True
