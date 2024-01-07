"""
Code utilities for running various experiments

"""
import torch
from torch_geometric.utils import get_num_hops
from src.attention_analysis_utils import (
    return_edges_in_k_hop,
    generate_att_dict,
    generate_att_dict_sim,
    attattribute,
    attattribute_sim,
    avgatt,
    attattribute_batch,
    attattribute_sim_batch,
)    
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class FaithfulnessExperiment:
    def __init__(self, model, data, device = 'cuda:0') -> None:
        self.model = model.to(device)
        self.data= data.to(device)
        self.device = device
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(data.x, data.edge_index, return_att = True)
            self.att = self.model.att
            
        self.num_hops = get_num_hops(model = self.model)
        
    def set_target_nodes(self, target_nodes: List[int]) -> None:
        self.target_nodes = target_nodes 
        print(f"Target nodes set...")
        print(f"Generating edge lists within {self.num_hops} hops of target nodes...")
        self.edge_lists_dict = {}
        
        for node in target_nodes:
            self.edge_lists_dict[node] = return_edges_in_k_hop(data = self.data, 
                                                            target_idx=node, 
                                                            hop=self.num_hops, 
                                                            self_loops=True)
        print("...Done")
    
    def get_attributions(self, 
                        target_node: Optional[List[int]] = None, 
                        get_attattribute: bool = True, 
                        get_attattribute_sim: bool = True, 
                        get_avgatt: bool = True,
                        verbose: bool = True) -> Dict[str, Any]:
        if target_node is None:
            target_node = self.target_nodes
        else:
            assert target_node in self.target_nodes, "Target node must be in the list of target nodes"
        assert get_attattribute or get_attattribute_sim or get_avgatt, "Must set at least one of attattribute, attattribute_sim, or avgatt to True"
        
        result = {}
        if isinstance(target_node, int):
            target_node = [target_node]
        if verbose:
            print(f"Getting attributions for {len(target_node)} nodes...")
        # Keep appending to the lists while iterating through the target nodes
        # Prep attribute dictionaries
        att_matrix_dict, correction_matrix_dict = self._prep_to_get_attattribute()
        att_matrix_dict_sim, correction_matrix_dict_sim = self._prep_to_get_attattribute_sim()

        for node in target_node:
            edge_lists = self.edge_lists_dict[node]
            # Initialize lists
            if get_attattribute:
                attattribute_list = []
            if get_attattribute_sim:
                attattribute_sim_list = []
            if get_avgatt:
                avgatt_list = []

            for current_edge in edge_lists:
                if get_attattribute:
                    att = attattribute(target_edge=current_edge,
                                    ref_node=node,   
                                    att_matrix_dict=att_matrix_dict,
                                    correction_matrix_dict=correction_matrix_dict)
                    attattribute_list.append(att)
                if get_attattribute_sim:
                    att_sim = attattribute_sim(target_edge=current_edge,
                                    ref_node=node,   
                                    att_matrix_dict=att_matrix_dict,
                                    att_matrix_dict_sim=att_matrix_dict_sim,
                                    correction_matrix_dict=correction_matrix_dict_sim)
                    attattribute_sim_list.append(att_sim)
                if get_avgatt:
                    att_avg = avgatt(target_edge=current_edge,
                                    ref_node=node,   
                                    att_matrix_dict=att_matrix_dict)
                    avgatt_list.append(att_avg)

            if get_attattribute:
                if 'attattribute' in result.keys():
                    result['attattribute'] += attattribute_list
                else:
                    result['attattribute'] = attattribute_list
            if get_attattribute_sim:
                if 'attattribute_sim' in result.keys():
                    result['attattribute_sim'] += attattribute_sim_list
                else:
                    result['attattribute_sim'] = attattribute_sim_list
            if get_avgatt:
                if 'avgatt' in result.keys():
                    result['avgatt'] += avgatt_list
                else:
                    result['avgatt'] = avgatt_list
                        
        if verbose:
            print("...Done")
        return result

            
    def _prep_to_get_attattribute(self) -> Tuple[Dict[int, torch.sparse.FloatTensor], Optional[Dict[int, torch.sparse.FloatTensor]]]:
        att_matrix_dict = generate_att_dict(model=self.model, data=self.data)
        if self.num_hops == 3:
            correction_matrix_dict = {}
            correction_matrix_dict[0] = torch.sparse.mm(att_matrix_dict[2], att_matrix_dict[1])
        else:
            correction_matrix_dict = None
            
        return att_matrix_dict, correction_matrix_dict
    
    def _prep_to_get_attattribute_sim(self) -> Tuple[Dict[int, torch.sparse.FloatTensor], Optional[Dict[int, torch.sparse.FloatTensor]]]:
        att_matrix_dict_sim = generate_att_dict_sim(model=self.model, data=self.data)
        if self.num_hops == 3:
            correction_matrix_dict_sim = {}
            correction_matrix_dict_sim[0] = torch.sparse.mm(att_matrix_dict_sim[2], att_matrix_dict_sim[1])
        else:
            correction_matrix_dict_sim = None
            
        return att_matrix_dict_sim, correction_matrix_dict_sim
    
    @torch.no_grad()
    def model_intervention(self, target_node: Optional[List[int]] = None) -> Dict[str, Any]:
        if target_node is None:
            target_node = self.target_nodes
        else:
            assert target_node in self.target_nodes, "Target node must be in the list of target nodes"
        
        result = {}
        if isinstance(target_node, int):
            target_node = [target_node]
        print(f"Getting model intervention for {len(target_node)} nodes...")
        
        # Keep appending to the lists while iterating through the target nodes
        for ref_node in target_node:
            edge_lists = self.edge_lists_dict[ref_node]
            # Get some baseline values before intervention
            output = self.model(
                x=self.data.x, edge_index=self.data.edge_index, return_att=True, mask_edge=None
            )
            pred = output.argmax(dim=1)[ref_node].item()  # Single integer
            pred_prob = output[ref_node].softmax(dim=0)[pred].item()  # Single float
            prob_vector = output[ref_node].softmax(dim=0)  # 1D vector of floats
            entropy = -(prob_vector * prob_vector.log()).sum().item()  # Single float
            
            # Initialize lists
            pred_list_masked, pred_prob_list_masked, entropy_list_masked = [], [], []

            for current_edge in edge_lists:
                output_masked = self.model(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    return_att=True,
                    mask_edge=current_edge,
                )
                # att_masked = self.model.att
                pred_masked = output_masked.argmax(dim=1)[ref_node].item()
                pred_prob_masked = (
                    output_masked[ref_node].softmax(dim=0)[pred_masked].item()
                )
                prob_vector_masked = output_masked[ref_node].softmax(dim=0)
                entropy_masked = (
                    -(prob_vector_masked * prob_vector_masked.log()).sum().item()
                )
                pred_list_masked.append(pred_masked)
                pred_prob_list_masked.append(pred_prob_masked)
                entropy_list_masked.append(entropy_masked)

            # For pred_list_masked, change into a list of bools indicating whether the prediction has changed
            if 'pred_list_masked_bool' in result.keys():
                # Compare pred_list_masked with
                result['pred_list_masked_bool'] += [pred != curr_pred for curr_pred in pred_list_masked]
            else:
                result['pred_list_masked_bool'] = [pred != curr_pred for curr_pred in pred_list_masked]
            # For pred_prob_list_masked, change into a list of floats indicating the change in prediction probability
            if 'pred_prob_list_masked_float' in result.keys():
                result['pred_prob_list_masked_float'] += [pred_prob - curr_pred_prob for curr_pred_prob in pred_prob_list_masked]
            else:
                result['pred_prob_list_masked_float'] = [pred_prob - curr_pred_prob for curr_pred_prob in pred_prob_list_masked]
            # For entropy_list_masked, change into a list of floats indicating the change in entropy
            if 'entropy_list_masked_float' in result.keys():
                result['entropy_list_masked_float'] += [entropy - curr_entropy for curr_entropy in entropy_list_masked]
            else:
                result['entropy_list_masked_float'] = [entropy - curr_entropy for curr_entropy in entropy_list_masked]
        
        return result
    
class FaithfulnessExperimentBatch:
    def __init__(self, model, data, device = 'cuda:0') -> None:
        self.model = model.to(device)
        self.data= data.to(device)
        self.device = device
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(data.x, data.edge_index, return_att = True)
            self.att = self.model.att
            
        self.num_hops = get_num_hops(model = self.model)
        
    def set_target_nodes(self, target_nodes: List[int]) -> None:
        self.target_nodes = target_nodes 
        print(f"Target nodes set...")
        print(f"Generating edge lists within {self.num_hops} hops of target nodes...")
        self.edge_lists_dict = {}
        
        for node in target_nodes:
            self.edge_lists_dict[node] = return_edges_in_k_hop(data = self.data, 
                                                            target_idx=node, 
                                                            hop=self.num_hops, 
                                                            self_loops=True)
        print("...Done")
    
    def get_attributions(self, 
                        target_node: Optional[List[int]] = None, 
                        get_attattribute: bool = True, 
                        get_attattribute_sim: bool = True, 
                        get_avgatt: bool = True,
                        verbose: bool = True) -> Dict[str, Any]:
        if target_node is None:
            target_node = self.target_nodes
        else:
            assert target_node in self.target_nodes, "Target node must be in the list of target nodes"
        assert get_attattribute or get_attattribute_sim or get_avgatt, "Must set at least one of attattribute, attattribute_sim, or avgatt to True"
        
        result = {}
        if isinstance(target_node, int):
            target_node = [target_node]
        if verbose:
            print(f"Getting attributions for {len(target_node)} nodes...")
        # Keep appending to the lists while iterating through the target nodes
        # Prep attribute dictionaries
        # Unlike FaithfulnessExperiment, we will not use sparse matrices
        att_matrix_dict, correction_matrix_dict = self._prep_to_get_attattribute()
        att_matrix_dict_sim, correction_matrix_dict_sim = self._prep_to_get_attattribute_sim()

        for node in target_node:
            edge_lists = self.edge_lists_dict[node]
            # Initialize lists
            if get_attattribute:
                attattribute_list = []
                self.attattribute_matrix = attattribute_batch(ref_node=node,
                                                        att_matrix_dict=att_matrix_dict,
                                                        correction_matrix_dict=correction_matrix_dict)
            if get_attattribute_sim:
                attattribute_sim_list = []
                self.attattribute_sim_matrix = attattribute_sim_batch(ref_node=node,
                                                        att_matrix_dict=att_matrix_dict,
                                                        att_matrix_dict_sim=att_matrix_dict_sim,
                                                        correction_matrix_dict=correction_matrix_dict_sim)
            if get_avgatt:
                avgatt_list = []
                temp = [att_matrix_dict[key] for key in att_matrix_dict.keys()]
                self.avgatt_list = torch.stack(temp).mean(dim=0).squeeze()

            for current_edge in edge_lists:
                if get_attattribute:
                    attattribute_list.append(self.attattribute_matrix[current_edge[1], current_edge[0]].item())
                if get_attattribute_sim:
                    attattribute_sim_list.append(self.attattribute_sim_matrix[current_edge[1], current_edge[0]].item())
                if get_avgatt:
                    avgatt_list.append(self.avgatt_list[current_edge[1], current_edge[0]].item())

            if get_attattribute:
                if 'attattribute' in result.keys():
                    result['attattribute'] += attattribute_list
                else:
                    result['attattribute'] = attattribute_list
            if get_attattribute_sim:
                if 'attattribute_sim' in result.keys():
                    result['attattribute_sim'] += attattribute_sim_list
                else:
                    result['attattribute_sim'] = attattribute_sim_list
            if get_avgatt:
                if 'avgatt' in result.keys():
                    result['avgatt'] += avgatt_list
                else:
                    result['avgatt'] = avgatt_list
                        
        if verbose:
            print("...Done")
        return result

            
    def _prep_to_get_attattribute(self) -> Tuple[Dict[int, torch.FloatTensor], Optional[Dict[int, torch.FloatTensor]]]:
        att_matrix_dict = generate_att_dict(model=self.model, data=self.data, sparse=False)
        if self.num_hops == 3:
            correction_matrix_dict = {}
            correction_matrix_dict[0] = torch.mm(att_matrix_dict[2], att_matrix_dict[1])
        else:
            correction_matrix_dict = None
            
        return att_matrix_dict, correction_matrix_dict
    
    def _prep_to_get_attattribute_sim(self) -> Tuple[Dict[int, torch.FloatTensor], Optional[Dict[int, torch.FloatTensor]]]:
        att_matrix_dict_sim = generate_att_dict_sim(model=self.model, data=self.data, sparse=False)
        if self.num_hops == 3:
            correction_matrix_dict_sim = {}
            correction_matrix_dict_sim[0] = torch.mm(att_matrix_dict_sim[2], att_matrix_dict_sim[1])
        else:
            correction_matrix_dict_sim = None
            
        return att_matrix_dict_sim, correction_matrix_dict_sim
    
    @torch.no_grad()
    def model_intervention(self, target_node: Optional[List[int]] = None) -> Dict[str, Any]:
        if target_node is None:
            target_node = self.target_nodes
        else:
            assert target_node in self.target_nodes, "Target node must be in the list of target nodes"
        
        result = {}
        if isinstance(target_node, int):
            target_node = [target_node]
        print(f"Getting model intervention for {len(target_node)} nodes...")
        
        # Keep appending to the lists while iterating through the target nodes
        for ref_node in target_node:
            edge_lists = self.edge_lists_dict[ref_node]
            # Get some baseline values before intervention
            output = self.model(
                x=self.data.x, edge_index=self.data.edge_index, return_att=True, mask_edge=None
            )
            pred = output.argmax(dim=1)[ref_node].item()  # Single integer
            pred_prob = output[ref_node].softmax(dim=0)[pred].item()  # Single float
            prob_vector = output[ref_node].softmax(dim=0)  # 1D vector of floats
            entropy = -(prob_vector * prob_vector.log()).sum().item()  # Single float
            
            # Initialize lists
            pred_list_masked, pred_prob_list_masked, entropy_list_masked = [], [], []

            for current_edge in edge_lists:
                output_masked = self.model(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    return_att=True,
                    mask_edge=current_edge,
                )
                # att_masked = self.model.att
                pred_masked = output_masked.argmax(dim=1)[ref_node].item()
                pred_prob_masked = (
                    output_masked[ref_node].softmax(dim=0)[pred_masked].item()
                )
                prob_vector_masked = output_masked[ref_node].softmax(dim=0)
                entropy_masked = (
                    -(prob_vector_masked * prob_vector_masked.log()).sum().item()
                )
                pred_list_masked.append(pred_masked)
                pred_prob_list_masked.append(pred_prob_masked)
                entropy_list_masked.append(entropy_masked)

            if 'pred_list_masked_bool' in result.keys():
                # Compare pred_list_masked with
                result['pred_list_masked_bool'] += [pred != curr_pred for curr_pred in pred_list_masked]
            else:
                result['pred_list_masked_bool'] = [pred != curr_pred for curr_pred in pred_list_masked]
            # For pred_prob_list_masked, change into a list of floats indicating the change in prediction probability
            if 'pred_prob_list_masked_float' in result.keys():
                result['pred_prob_list_masked_float'] += [pred_prob - curr_pred_prob for curr_pred_prob in pred_prob_list_masked]
            else:
                result['pred_prob_list_masked_float'] = [pred_prob - curr_pred_prob for curr_pred_prob in pred_prob_list_masked]
            # For entropy_list_masked, change into a list of floats indicating the change in entropy
            if 'entropy_list_masked_float' in result.keys():
                result['entropy_list_masked_float'] += [entropy - curr_entropy for curr_entropy in entropy_list_masked]
            else:
                result['entropy_list_masked_float'] = [entropy - curr_entropy for curr_entropy in entropy_list_masked]
        
        return result


"""
Some assumptions regarding how different data is named
attribution_dict: Dict[str, Any]
    'attattribute': List[float]
    'attattribute_sim': List[float]
    'avgatt': List[float]
intervention_dict: Dict[str, Any]
    'pred_list_masked': List[int]
    'pred_prob_list_masked': List[float]
    'entropy_list_masked': List[float]
    'pred_list_masked_bool': List[bool]
    'pred_prob_list_masked_float': List[float]
    'entropy_list_masked_float': List[float]
    
We will use the same naming convention unless otherwise specified to avoid confusion
"""
class FaithfulnessExperimentAnalysis:
    def __init__(self, attribution_dict: Dict[str, Any], 
                intervention_dict: Dict[str, Any]) -> None:
        self.attribution_dict = attribution_dict
        self.intervention_dict = intervention_dict
        
        
    def generate_random_baseline(self) -> Dict[str, Any]:
        # Random baseline
        if isinstance(self.attribution_dict['attattribute'], torch.Tensor):
            self.random_attr = torch.rand(self.attribution_dict['attattribute'].shape).to(self.attribution_dict['attattribute'].device)
        elif isinstance(self.attribution_dict['attattribute'], list):
            self.random_attr = torch.rand(len(self.attribution_dict['attattribute']))
        
    def get_full_analysis(self, include_random: bool = True) -> Dict[str, Any]:
        # Get pearsonr, spearmanr, kendalltau, auc for
        # attattribute, attattribute_sim, avgatt, and random (if include_random is True)
        # for each vs. pred_prob_list_masked_float, vs. entropy_list_masked_float, and vs. pred_list_masked_bool ('ΔPC', 'ΔNE', 'ROC_AUC')

        result = {}
        # attattribute
        result['attattribute'] = {}
        result['attattribute']['ΔPC'] = self.get_correlations(attribution=self.attribution_dict['attattribute'], 
                                                            intervention=self.intervention_dict['pred_prob_list_masked_float'])
        result['attattribute']['ΔNE'] = self.get_correlations(attribution=self.attribution_dict['attattribute'], 
                                                            intervention=self.intervention_dict['entropy_list_masked_float'])
        result['attattribute']['ROC_AUC'] = self.get_auc(attribution=self.attribution_dict['attattribute'], 
                                                        intervention=self.intervention_dict['pred_list_masked_bool'])
        
        result['attattribute_sim'] = {}
        result['attattribute_sim']['ΔPC'] = self.get_correlations(attribution=self.attribution_dict['attattribute_sim'], 
                                                                intervention=self.intervention_dict['pred_prob_list_masked_float'])
        result['attattribute_sim']['ΔNE'] = self.get_correlations(attribution=self.attribution_dict['attattribute_sim'], 
                                                                intervention=self.intervention_dict['entropy_list_masked_float'])
        result['attattribute_sim']['ROC_AUC'] = self.get_auc(attribution=self.attribution_dict['attattribute_sim'], 
                                                            intervention=self.intervention_dict['pred_list_masked_bool'])
        
        result['avgatt'] = {}
        result['avgatt']['ΔPC'] = self.get_correlations(attribution=self.attribution_dict['avgatt'], 
                                                        intervention=self.intervention_dict['pred_prob_list_masked_float'])
        result['avgatt']['ΔNE'] = self.get_correlations(attribution=self.attribution_dict['avgatt'], 
                                                        intervention=self.intervention_dict['entropy_list_masked_float'])
        result['avgatt']['ROC_AUC'] = self.get_auc(attribution=self.attribution_dict['avgatt'], 
                                                intervention=self.intervention_dict['pred_list_masked_bool'])
        
        if include_random:
            result['random'] = {}
            result['random']['ΔPC'] = self.get_correlations(attribution=self.random_attr, 
                                                            intervention=self.intervention_dict['pred_prob_list_masked_float'])
            result['random']['ΔNE'] = self.get_correlations(attribution=self.random_attr, 
                                                            intervention=self.intervention_dict['entropy_list_masked_float'])
            result['random']['ROC_AUC'] = self.get_auc(attribution=self.random_attr, 
                                                    intervention=self.intervention_dict['pred_list_masked_bool'])
            
        return result
        
    def get_correlations(self, attribution, intervention) -> Dict[str, Any]:
        # Check nan values, discard them while keeping track of indices
        # Do not remove them from the original lists
        if sum(np.isnan(attribution)) > 0 or sum(np.isnan(intervention)) > 0:
            nan_indices = []
            for i, (att, inter) in enumerate(zip(attribution, intervention)):
                if np.isnan(att) or np.isnan(inter):
                    nan_indices.append(i)
            attribution_ = [att for i, att in enumerate(attribution) if i not in nan_indices]
            intervention_ = [inter for i, inter in enumerate(intervention) if i not in nan_indices]
        else:
            attribution_ = attribution
            intervention_ = intervention
        
        # Attribution vs intervention
        pearsonr_value = pearsonr(attribution_, intervention_)[0]
        spearmanr_value = spearmanr(attribution_, intervention_)[0]
        kendalltau_value = kendalltau(attribution_, intervention_)[0]
        
        # Return as dictionary
        result = {}
        result['pearsonr'] = pearsonr_value
        result['spearmanr'] = spearmanr_value
        result['kendalltau'] = kendalltau_value
        
        return result
    
    def get_auc(self, attribution, intervention) -> float:
        auc = roc_auc_score(y_true=intervention, y_score=attribution)
        return auc
    
    def print_result(self, result: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        for key, value in result.items():
            print(f"{key}:")
            for key2, value2 in value.items():
                print(f"\t{key2}:")
                if key2 == 'ROC_AUC':
                    print(f"\t\t{value2:.4f}")
                else:
                    for key3, value3 in value2.items():
                        print(f"\t\t{key3}: {value3:.4f}")
