# Import the utility functions
import torch
from src.experiment_utils import FaithfulnessExperiment, FaithfulnessExperimentAnalysis
import torch
import numpy as np
# Fix the seed
np.random.seed(0)

dataset_name = "Arxiv"

device = torch.device("cpu")
dataset_folder = "/workspace/Datasets"

# Load the network data locally (for reproduction)
data = torch.load(f"{dataset_folder}/{dataset_name}.pt").to(device)

test_nodes = (np.random.choice(np.arange(data.num_nodes), size=100, replace=False)).tolist()
device = torch.device("cpu")
dataset_folder = "/workspace/Datasets"
model_folder = "/workspace/Models"
config = "3L1H"

# Load the network data locally (for reproduction)
# data = torch.load(f"{dataset_folder}/{dataset_name}.pt").to(device)
# Load model as a whole
model = torch.load(f"{model_folder}/GAT_{dataset_name}_{config}.pt").to(device)
model.eval()

with torch.no_grad():
    _  = model(data.x, data.edge_index, return_att = True)
    att = list(model.att)
    

# Define the experiment
faithfulness_experiment = FaithfulnessExperiment(
    model = model,
    data = data,
    device = device,
    )

faithfulness_experiment.set_target_nodes(
    test_nodes
)
print("Get attributions")
attribute_dict = faithfulness_experiment.get_attributions()
print("Get interventions")
intervention_dict = faithfulness_experiment.model_intervention()

analysis = FaithfulnessExperimentAnalysis(
    attribution_dict = attribute_dict,
    intervention_dict = intervention_dict,
)
analysis.generate_random_baseline()
result = analysis.get_full_analysis()
analysis.print_result(result=result)

# Save the result
experiment_artifacts_folder = "/workspace/Experimental_Artifacts"
torch.save(attribute_dict, f"{experiment_artifacts_folder}/Faithfulness_GAT_{dataset_name}_{config}_Attributions.pt")
torch.save(intervention_dict, f"{experiment_artifacts_folder}/Faithfulness_GAT_{dataset_name}_{config}_Interventions.pt")
experiments_folder = "/workspace/Experimental_Results"
torch.save(result, f"{experiments_folder}/Faithfulness_GAT_{dataset_name}_{config}.pt")