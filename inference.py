import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from config import params as gnn_params
from data_processing import TemporalGraphDataset, add_jammed_column
from train import initialize_model, validate
from sklearn.metrics import mean_squared_error

def calculate_noise(distance, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing, pl0=32, d0=1):
    # Prevent log of zero if distance is zero by replacing it with a very small positive number
    d = np.where(distance == 0, np.finfo(float).eps, distance)
    # Path loss calculation
    path_loss_db = pl0 + 10 * path_loss_exponent * np.log10(d / d0)
    # Apply shadowing if sigma is not zero
    if shadowing != 0:
        path_loss_db += np.random.normal(0, shadowing, size=d.shape)
    return P_tx_jammer + G_tx_jammer - path_loss_db


def generate_dummy_data(num_nodes, jammer_position, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing):
    a = np.random.uniform(0.5, 1.5)  # Random alignment coefficient
    b = np.random.uniform(0.1, 0.5)  # Random growth rate
    cov = np.random.randint(2, 5)
    theta = np.linspace(0, cov * np.pi, num_nodes)
    r = a + b * theta
    x = r * np.cos(theta) + jammer_position[0]
    y = r * np.sin(theta) + jammer_position[1]
    node_positions = np.vstack((x, y)).T

    # Calculate Euclidean distances from each node to the jammer
    distances = np.sqrt((x - jammer_position[0]) ** 2 + (y - jammer_position[1]) ** 2)

    node_noise = calculate_noise(distances, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing)
    data = pd.DataFrame({
        'node_positions': [node_positions.tolist()],
        'node_noise': [node_noise.tolist()]
    })
    return data

def plot_positions(data, jammer_position, predicted_position):
    node_positions = np.array(data['node_positions'].iloc[0])
    plt.figure(figsize=(8, 6))
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c='blue', label='Nodes', s=1, alpha=0.5)
    plt.scatter(jammer_position[0], jammer_position[1], c='red', marker='x', s=5, label='Jammer')
    plt.scatter(predicted_position[0][0], predicted_position[0][1], c='green', marker='o', s=5, label='Prediction', alpha=0.5)
    plt.title('Node and Jammer Positions')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_rmse(predicted_position, actual_position):
    rmse = np.sqrt(mean_squared_error([actual_position], [predicted_position[0]]))
    return rmse

def gnn(data):
    gnn_params.update({'inference': True})
    # Add jammed column
    data = add_jammed_column(data, threshold=-55)
    # Set the device to use for computations
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # Initialize data loader
    test_dataset = TemporalGraphDataset(data, test=True, discretization_coeff=1.0)
    test_loader = DataLoader(test_dataset, batch_size=gnn_params['batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    # Load trained model
    model_path = 'trained_model_GAT_cartesian_knnfc_minmax_400hybrid_combined.pth'
    model, optimizer, scheduler, criterion = initialize_model(device, gnn_params, len(test_loader))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    # Predict jammer position
    predictions, _, _ = validate(model, test_loader, criterion, device, test_loader=True)
    return predictions

# Example usage
num_samples = 1000
jammer_pos = [50, 50]
jammer_ptx = np.random.uniform(20, 40)
jammer_gtx = np.random.uniform(0, 5)
plexp = 3.5
sigma = np.random.uniform(2, 6)
print(f"Ptx jammer: {jammer_ptx}, Gtx jammer: {jammer_gtx}, PL: {plexp}, Sigma: {sigma}")

data = generate_dummy_data(num_samples, jammer_pos, jammer_ptx, jammer_gtx, plexp, sigma)
predicted_jammer_pos = gnn(data)
# plot_positions(data, jammer_pos, predicted_jammer_pos)
rmse = calculate_rmse(predicted_jammer_pos, jammer_pos)
print("Predicted Jammer Position:", predicted_jammer_pos)
print(f"RMSE: {round(rmse, 2)} m")

