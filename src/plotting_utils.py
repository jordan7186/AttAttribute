import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Any, Optional, Union

def faithfulness_experiment_plot(attribution_dict: Dict[str, Any],
                                intervention_dict: Dict[str, Any],
                                dataset_name: str,
                                ) -> None:
    group_plot_intervention_vs_attribution(
        attribution_dict=attribution_dict,
        intervention_dict=intervention_dict['pred_list_masked'],
        intervention_label='ΔPC',
        dataset_name=dataset_name,
    )
    group_plot_intervention_vs_attribution(
        attribution_dict=attribution_dict,
        intervention_dict=intervention_dict['pred_prob_list_masked'],
        intervention_label='ΔNE',
        dataset_name=dataset_name,
    )
    group_plot_intervention_vs_attribution_histogram(
        attribution_dict=attribution_dict,
        intervention_dict=intervention_dict,
        dataset_name=dataset_name,
    )

def group_plot_intervention_vs_attribution(attribution_dict: Dict[str, Any], 
                                    intervention_dict: Dict[str, Any],
                                    intervention_label: str,
                                    dataset_name: str,) -> None:
    # Check whak kind of attribution data is present in the dictionary
    if 'attattribute' in attribution_dict.keys():
        attattribute = True
    if 'attattribute_sim' in attribution_dict.keys():
        attattribute_sim = True
    if 'avgatt' in attribution_dict.keys():
        avgatt = True
    
    num_exp = attattribute + attattribute_sim + avgatt
    curr_idx = 0
    
    # Set canvas size according to the available 
    fig, axs = plt.subplots(1, num_exp, figsize=(5*num_exp + 2, 5))
    if 'attattribute' in attribution_dict.keys():
        plot_intervention_vs_attribution_scatter(
            axs=axs,
            axs_idx=curr_idx,
            color_code='blue',
            intervention_list=intervention_dict,
            attribution_list=attribution_dict['attattribute'],
            intervention_label=f'{intervention_label}',
            attribution_label='Attattribute',
            dataset_name=dataset_name,
        )
        curr_idx += 1
    if 'attattribute_sim' in attribution_dict.keys():
        plot_intervention_vs_attribution_scatter(
            axs=axs,
            axs_idx=curr_idx,
            color_code='red',
            intervention_list=intervention_dict,
            attribution_list=attribution_dict['attattribute_sim'],
            intervention_label=f'{intervention_label}',
            attribution_label='Attattribute_sim',
            dataset_name=dataset_name,
        )
        curr_idx += 1
    if 'avgatt' in attribution_dict.keys():
        plot_intervention_vs_attribution_scatter(
            axs=axs,
            axs_idx=curr_idx,
            color_code='green',
            intervention_list=intervention_dict,
            attribution_list=attribution_dict['avgatt'],
            intervention_label=f'{intervention_label}',
            attribution_label='Avgatt',
            dataset_name=dataset_name,
        )
    plt.show()
    
def group_plot_intervention_vs_attribution_histogram(attribution_dict: Dict[str, Any],
                                                    intervention_dict: Dict[str, Any],
                                                    dataset_name: str,) -> None:
    # Check whak kind of attribution data is present in the dictionary
    if 'attattribute' in attribution_dict.keys():
        attattribute = True
    if 'attattribute_sim' in attribution_dict.keys():
        attattribute_sim = True
    if 'avgatt' in attribution_dict.keys():
        avgatt = True
        
    num_exp = attattribute + attattribute_sim + avgatt
    curr_idx = 0
    
    # Set canvas size according to the available
    fig, axs = plt.subplots(1, num_exp, figsize=(5*num_exp + 2, 5))
    if 'attattribute' in attribution_dict.keys():
        plot_intervention_vs_attribution_histogram(
            axs=axs,
            axs_idx=curr_idx,
            attribution_list=attribution_dict['attattribute'],
            attribution_label='Attattribute',
            pred_list_masked_bool=intervention_dict['pred_list_masked_bool'],
            dataset_name=dataset_name,
        )
        curr_idx += 1
    if 'attattribute_sim' in attribution_dict.keys():
        plot_intervention_vs_attribution_histogram(
            axs=axs,
            axs_idx=curr_idx,
            attribution_list=attribution_dict['attattribute_sim'],
            attribution_label='Attattribute_sim',
            pred_list_masked_bool=intervention_dict['pred_list_masked_bool'],
            dataset_name=dataset_name,
        )
        curr_idx += 1
    if 'avgatt' in attribution_dict.keys():
        plot_intervention_vs_attribution_histogram(
            axs=axs,
            axs_idx=curr_idx,
            attribution_list=attribution_dict['avgatt'],
            attribution_label='Avgatt',
            pred_list_masked_bool=intervention_dict['pred_list_masked_bool'],
            dataset_name=dataset_name,
        )
    plt.show()
                                                
    
    
def plot_intervention_vs_attribution_scatter(axs, 
                                            axs_idx: int, 
                                            color_code: str,
                                            intervention_list: List[float],
                                            attribution_list: List[float],
                                            intervention_label: str,
                                            attribution_label: str,
                                            dataset_name: str,
                                            correlation: bool = True,):
    axs[axs_idx].scatter(
        attribution_list,
        intervention_list,
        alpha=0.5,
        color=color_code,
        marker='x',
        label=f'{intervention_label} vs. {attribution_label}',
    )
    axs[axs_idx].set_title(f'{dataset_name}')
    axs[axs_idx].set_xlabel(f'{attribution_label}')
    axs[axs_idx].set_ylabel(f'{intervention_label}')

    if correlation:
        corr, _ = pearsonr(attribution_list, intervention_list)
        axs[axs_idx].text(
            0.05, 
            0.95, 
            f'Correlation: {corr:.2f}', 
            transform=axs[axs_idx].transAxes,
            fontsize=10,
            verticalalignment='top',
        )
        
def plot_intervention_vs_attribution_histogram(
    axs, 
    axs_idx: int, 
    attribution_list: List[float],
    attribution_label: str,
    pred_list_masked_bool: List[int or bool],
    dataset_name: str,
    ) -> None:
    # pred_list_masked_bool is expected to be intervention_dict['pred_list_masked_bool']
    max_attribute = max(attribution_list).ceil()
    pred_has_remained_mask = torch.logical_not(pred_list_masked_bool)
    pred_has_flipped_mask = pred_list_masked_bool.bool()
    
    bins = torch.linspace(0, max_attribute, 40)
    
    axs[axs_idx].hist(
        attribution_list[pred_has_flipped_mask],
        bins=bins,
        alpha=0.5,
        color="#F24141",
        label='Prediction changed',
    )
    axs[axs_idx].hist(
        attribution_list[pred_has_remained_mask],
        bins=bins,
        alpha=0.5,
        color="#29A5F2" ,
        label='Prediction unchanged',
    )
    axs[axs_idx].set_title(f'{dataset_name}')
    axs[axs_idx].set_xlabel(f'{attribution_label}')
    axs[axs_idx].set_ylabel('Relative frequency')
    axs[axs_idx].legend()