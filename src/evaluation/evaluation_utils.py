import numpy as np
import torch
import operator
import os
from src.models.models import EncodersExecutor, EncodersExecutorConfig
def evaluate(
    output_description, # shape: batchSize x featureSize
    output_scene,       # shape: batchSize x featureSize
    scene_classes=None,
    description_classes=None
):
    # Compute similarity matrix (basically equivalent to calling cosine_sim from utils.model_utils)
    similarity_matrix = torch.nn.functional.cosine_similarity(
        output_scene.unsqueeze(1),
        output_description.unsqueeze(0),
        dim=-1
    )  # similarity matrix of shape batchSize x batchSize

    sd_rank_matrix = torch.argsort(similarity_matrix, descending=True, dim=1)  # Scene-to-Description ranks
    ds_rank_matrix = torch.argsort(similarity_matrix.T, descending=True, dim=1)  # Description-to-Scene ranks

    # Calculate ranks as np arrays
    def get_ranks(ranks, num_elements):
        rank_positions = (ranks == torch.arange(num_elements).unsqueeze(1).to(ranks.device)).nonzero(as_tuple=True)[1]
        return rank_positions.cpu().numpy()

    n_scenes, n_descriptions = similarity_matrix.shape
    scene_ranks = get_ranks(sd_rank_matrix, n_scenes)               # ranks of the actual matches
    description_ranks = get_ranks(ds_rank_matrix, n_descriptions)

    # Calculate metrics
    def calculate_metrics(ranks, n_elements):
        r1 = 100 * (ranks < 1).sum() / n_elements
        r5 = 100 * (ranks < 5).sum() / n_elements
        r10 = 100 * (ranks < 10).sum() / n_elements
        medr = np.median(ranks + 1)
        meanr = np.mean(ranks) + 1
        return r1, r5, r10, medr, meanr

    sd_r1, sd_r5, sd_r10, sd_medr, sd_meanr = calculate_metrics(scene_ranks, n_scenes)
    ds_r1, ds_r5, ds_r10, ds_medr, ds_meanr = calculate_metrics(description_ranks, n_descriptions)

    # NDCG
    def calculate_ndcg(ranks):
        discout_t = torch.log2(torch.arange(2,ranks.shape[1]+2))
        dcg = ((ranks == torch.arange(ranks.shape[0]).unsqueeze(1)).float()/discout_t).sum(axis=1)
        ndcg = 100*torch.mean(dcg).item() # ideal_dcg = 1 in the case of a single relevant element
        return ndcg
    
    sd_avg_ndcg = calculate_ndcg(sd_rank_matrix)
    ds_avg_ndcg = calculate_ndcg(ds_rank_matrix)

    # mAP
    def calculate_map(ranks):
        class_matching = (ranks == torch.arange(ranks.shape[0]).unsqueeze(1)).float()
        precisions_at_k = class_matching.cumsum(axis=1)/torch.arange(1,ranks.shape[1]+1)*class_matching
        ap = precisions_at_k.sum(axis=1) # no division needed as there is a single relevant element for each row
        return 100*ap.mean().item()
    
    sd_mAP = calculate_map(sd_rank_matrix)
    ds_mAP = calculate_map(ds_rank_matrix)


    # Class-based metrics (only if class vectors are provided)
    if scene_classes is not None and description_classes is not None:

        # Function to compute Recall@K for class-based metrics
        def calculate_class_precision_recall_at_k(ranks, query_classes, key_classes, k):
            relevant_elements = (key_classes[ranks] == query_classes.unsqueeze(1)).sum(dim=1)
            matches = (key_classes[ranks[:, :k]] == query_classes.unsqueeze(1))
            precision_at_k = matches.sum(dim=1).float().mean().item() * 100 / k  # Precision@K
            recall_at_k = (matches.sum(dim=1).float() / relevant_elements).mean().item() * 100  # Recall@K
            return precision_at_k, recall_at_k

        sd_p1_by_class, sd_r1_by_class = calculate_class_precision_recall_at_k(sd_rank_matrix, scene_classes, description_classes, k=1)
        sd_p5_by_class, sd_r5_by_class = calculate_class_precision_recall_at_k(sd_rank_matrix, scene_classes, description_classes, k=5)
        sd_p10_by_class, sd_r10_by_class = calculate_class_precision_recall_at_k(sd_rank_matrix, scene_classes, description_classes, k=10)

        ds_p1_by_class, ds_r1_by_class = calculate_class_precision_recall_at_k(ds_rank_matrix, description_classes, scene_classes, k=1)
        ds_p5_by_class, ds_r5_by_class = calculate_class_precision_recall_at_k(ds_rank_matrix, description_classes, scene_classes, k=5)
        ds_p10_by_class, ds_r10_by_class = calculate_class_precision_recall_at_k(ds_rank_matrix, description_classes, scene_classes, k=10)

        # Compute NDCG
        def calculate_class_aware_ndcg(ranks, query_classes, key_classes):
            
            discout_t = 1/torch.log2(torch.arange(2,ranks.shape[1]+2))
            class_matching = (key_classes[ranks] == query_classes.unsqueeze(1)).float()
            dcg = (class_matching*discout_t).sum(axis=1)
            ideal_dcg = (class_matching.sort(descending=True, axis=1).values * discout_t).sum(axis=1)
            ndcg = 100* torch.mean(dcg/ideal_dcg).item()
            return ndcg

        # Class-aware mAP
        def calculate_class_map(ranks, query_classes, key_classes):
            class_matching = (key_classes[ranks] == query_classes.unsqueeze(1)).float()
            precisions_at_k = class_matching.cumsum(axis=1)/torch.arange(1,ranks.shape[1]+1)*class_matching
            ap = precisions_at_k.sum(axis=1)/class_matching.sum(axis=1)
            return 100*ap.mean().item()

        sd_avg_ndcg_by_class = calculate_class_aware_ndcg(sd_rank_matrix, scene_classes, description_classes)
        ds_avg_ndcg_by_class = calculate_class_aware_ndcg(ds_rank_matrix, description_classes, scene_classes)
        sd_map_by_class = calculate_class_map(sd_rank_matrix, scene_classes, description_classes)
        ds_map_by_class = calculate_class_map(ds_rank_matrix, description_classes, scene_classes)

    else:
        sd_p1_by_class, sd_r1_by_class = -1, -1
        sd_p5_by_class, sd_r5_by_class = -1, -1
        sd_p10_by_class, sd_r10_by_class = -1, -1
        ds_p1_by_class, ds_r1_by_class = -1, -1
        ds_p5_by_class, ds_r5_by_class = -1, -1
        ds_p10_by_class, ds_r10_by_class = -1, -1
        sd_avg_ndcg_by_class, ds_avg_ndcg_by_class = -1, -1
        sd_map_by_class, ds_map_by_class = -1, -1

    # Prepare results as a dictionary
    results = {
        "t2s_R@1": ds_r1,
        "t2s_R@5": ds_r5,
        "t2s_R@10": ds_r10,
        "t2s_median_rank": ds_medr,
        "t2s_mean_rank": ds_meanr,
        "t2s_class_R@1": ds_r1_by_class,
        "t2s_class_R@5": ds_r5_by_class,
        "t2s_class_R@10": ds_r10_by_class,
        "t2s_class_P@1": ds_p1_by_class,
        "t2s_class_P@5": ds_p5_by_class,
        "t2s_class_P@10": ds_p10_by_class,
        
        "s2t_R@1": sd_r1,
        "s2t_R@5": sd_r5,
        "s2t_R@10": sd_r10,
        "s2t_median_rank": sd_medr,
        "s2t_mean_rank": sd_meanr,
        "s2t_class_R@1": sd_r1_by_class,
        "s2t_class_R@5": sd_r5_by_class,
        "s2t_class_R@10": sd_r10_by_class,
        "s2t_class_P@1": sd_p1_by_class,
        "s2t_class_P@5": sd_p5_by_class,
        "s2t_class_P@10": sd_p10_by_class,

        "t2s_avg_ndcg": ds_avg_ndcg,
        "s2t_avg_ndcg": sd_avg_ndcg,
        "s2t_avg_ndcg_by_class": sd_avg_ndcg_by_class,
        "t2s_avg_ndcg_by_class": ds_avg_ndcg_by_class,

        "s2t_mAP": sd_mAP,
        "t2s_mAP": ds_mAP,
        "s2t_mAP_by_class": sd_map_by_class,
        "t2s_mAP_by_class": ds_map_by_class,
    }

    return results


class BestModelManager():
    """small utility class to log the best model metric based on the chosen strategy
    """


    __best_strategies_comp_and_eval_idx = {
        't2s-r1':(operator.ge, 't2s_R@1'),
        't2s-r5':(operator.ge, 't2s_R@5'),
        't2s-r10':(operator.ge, 't2s_R@10'),
        's2t-r1':(operator.ge, 's2t_R@1'),
        's2t-r5':(operator.ge, 's2t_R@5'),
        's2t-r10':(operator.ge, 's2t_R@10'),
        't2s-medr':(operator.le, 't2s_median_rank'),
        's2t-medr':(operator.le, 's2t_median_rank'),
        't2s-meanr':(operator.le, 't2s_mean_rank'),
        's2t-meanr':(operator.le, 's2t_mean_rank'),

        "t2s_c_r1": (operator.ge, "t2s_class_R@1"),
        "t2s_c_r5": (operator.ge, "t2s_class_R@5"),
        "t2s_c_r10": (operator.ge, "t2s_class_R@10"),
        
        "t2s_c_r1": (operator.ge, "t2s_class_P@1"),
        "t2s_c_r5": (operator.ge, "t2s_class_P@5"),
        "t2s_c_r10": (operator.ge, "t2s_class_P@10"),
        
        "s2t_c_r1": (operator.ge, "s2t_class_R@1"),
        "s2t_c_r5": (operator.ge, "s2t_class_R@5"),
        "s2t_c_r10": (operator.ge, "s2t_class_R@10"),
        
        "s2t_c_r1":(operator.ge, "s2t_class_P@1"),
        "s2t_c_r5":(operator.ge, "s2t_class_P@5"),
        "s2t_c_r10":(operator.ge, "s2t_class_P@10"),
        
        "t2s_ndcg": (operator.ge, "t2s_avg_ndcg"),
        "s2t_ndcg": (operator.ge, "s2t_avg_ndcg"),
        "s2t_c_ndcg": (operator.ge, "s2t_avg_ndcg_by_class"),
        "t2s_c_ndcg": (operator.ge, "t2s_avg_ndcg_by_class"),

        "s2t_mAP": (operator.ge, "s2t_mAP"),
        "t2s_mAP": (operator.ge, "t2s_mAP"),
        "s2t_c_mAP": (operator.ge, "s2t_mAP_by_class"),
        "t2s_c_mAP": (operator.ge, "t2s_mAP_by_class"),
        #'loss':(operator.le, None), # this seems wrong, maybe it is a TODO: ass loss as a metric
    }

    def __init__(self, best_model_strategy):
        strategies = self.get_best_model_strategies()
        assert best_model_strategy in strategies, f"strategy {best_model_strategy} is unavailable, choose among {strategies}"
        self.operator, self.eval_res_idx = __class__.__best_strategies_comp_and_eval_idx[best_model_strategy]
        self.best_value = None
        # saving the best model here as a variable would not make much sense as the model in a complex dynamic object

    @classmethod
    def get_best_model_strategies(cls):
        return list(cls.__best_strategies_comp_and_eval_idx.keys())

    def updateBest(self, evalRes) -> bool:
        """updates the best value based on the strategy chosen at init

        Args:
            evalRes: indexable series of metrics given by the evaluation procedure
            
        Returns True if the new value is the new best value
        """
        def compare(op, a, b):
            return op(a, b)
        
        if self.best_value is None or compare(self.operator, evalRes[self.eval_res_idx], self.best_value):
            self.best_value = evalRes[self.eval_res_idx]
            
            return True
        else:
            return False
    
    def get_best_value(self):
        return self.best_value
