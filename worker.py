import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import nats
import asyncio
import os
import json
import math
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from config import params as gnn_params
from data_processing import TemporalGraphDataset, add_jammed_column
from train import initialize_model, validate
from sklearn.metrics import mean_squared_error

def gps_to_2d(lat, lon, radius=6378137.0):
    """
    Converts GPS (latitude, longitude) to 2D Cartesian coordinates using Mercator Projection.
    """
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Mercator projection formulas
    x = radius * lon_rad
    y = radius * math.log(math.tan(math.pi / 4 + lat_rad / 2))

    return x, y

def twod_to_gps(x, y, radius=6378137.0):
    """
    Converts 2D Cartesian coordinates back to GPS (latitude, longitude) using Mercator Projection.
    """
    lon = math.degrees(x / radius)
    lat = math.degrees(2 * math.atan(math.exp(y / radius)) - math.pi / 2)

    return lat, lon

def gps_to_cartesian(lat, lon, alt=0):
    # Constants for WGS84
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = f * (2 - f)  # Square of eccentricity

    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)

    # Cartesian coordinates
    X = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - e2) * N + alt) * math.sin(lat_rad)

    return X, Y, Z

def cartesian_to_gps(X, Y, Z):
    # WGS84 constants
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = f * (2 - f)  # Square of eccentricity
    b = a * (1 - f)  # Semi-minor axis

    # Longitude
    lon = math.atan2(Y, X)

    # Iterative solution for latitude
    p = math.sqrt(X**2 + Y**2)
    lat = math.atan2(Z, p * (1 - e2))  # Initial guess
    for _ in range(5):  # Iterate to improve accuracy
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N
        lat = math.atan2(Z, p * (1 - e2 * N / (N + h)))

    # Altitude
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    h = p / math.cos(lat) - N

    # Convert radians back to degrees
    lat = math.degrees(lat)
    lon = math.degrees(lon)

    return lat, lon, h

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

def load_gnn():
    model_path = 'trained_model_GAT_cartesian_knnfc_unit_sphere_400hybrid_combined.pth'
    #model_path = 'trained_model_GAT_cartesian_knnfc_minmax_400hybrid_combined.pth'
    model, optimizer, scheduler, criterion = initialize_model('cpu', gnn_params, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    return model, criterion

def gnn(model, criterion, data):
    test_dataset = TemporalGraphDataset(data, test=True, discretization_coeff=1.0)
    test_loader = DataLoader(test_dataset, batch_size=gnn_params['batch_size'], shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    predictions, _, _ = validate(model, test_loader, criterion, 'cpu', test_loader=True)
    return predictions

def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth radius in meters (mean radius)
    R = 6371000.0

    # Calculate distance
    distance = R * c
    return distance

## Example usage
#num_samples = 1000
#jammer_pos = [50, 50]
#jammer_ptx = np.random.uniform(20, 40)
#jammer_gtx = np.random.uniform(0, 5)
#plexp = 3.5
#sigma = np.random.uniform(2, 6)
#print(f"Ptx jammer: {jammer_ptx}, Gtx jammer: {jammer_gtx}, PL: {plexp}, Sigma: {sigma}")
#
#data = generate_dummy_data(num_samples, jammer_pos, jammer_ptx, jammer_gtx, plexp, sigma)
#predicted_jammer_pos = gnn(data)
## plot_positions(data, jammer_pos, predicted_jammer_pos)
#rmse = calculate_rmse(predicted_jammer_pos, jammer_pos)
#print("Predicted Jammer Position:", predicted_jammer_pos)
#print(f"RMSE: {round(rmse, 2)} m")

async def main():
    conn = await nats.connect(servers = os.getenv('NATS_SERVER', 'nats://localhost:4224'))
    print("conn ok")
    sub = await conn.subscribe("telemetry.current_location")

    jammer_ptx = 40 #np.random.uniform(20, 40)
    jammer_gtx = 2 #np.random.uniform(0, 5)
    plexp = 3
    sigma = 2 #np.random.uniform(2, 6)
    print(f"Ptx jammer: {jammer_ptx}, Gtx jammer: {jammer_gtx}, PL: {plexp}, Sigma: {sigma}")

    gnn_params.update({'inference': True})
    model, criterion = load_gnn()

    jammer_pos = None
    all_positions = []
    all_noises = []
    pos_jammed = None

    gotSomeData = 0
    async for msg in sub.messages:
        #print(f"Received a message on '{msg.subject} {msg.reply}': {msg.data.decode()}")
        #{"device_id":"dev_pmc01","ts":"2025-01-20T10:47:55.200333347Z","payload":{"current_position":{"lat":24.436311225969455,"lon":54.61389614281878,"alt":0},"distance":0}}
        location_msg = json.loads(msg.data, object_hook=lambda d: SimpleNamespace(**d))
        #print("location message:", location_msg)

        #pos = gps_to_cartesian(location_msg.payload.current_position.lat, location_msg.payload.current_position.lon, location_msg.payload.current_position.alt)
        pos = gps_to_2d(location_msg.payload.current_position.lat, location_msg.payload.current_position.lon)
        did = location_msg.device_id

        #gotSomeData = False

        if did == "dev_pmc01":
            continue
        elif did == "mercury42f0135a7d9f870":
            jammer_pos = pos
            #print("jammer position set:", jammer_pos)
        else:
            if jammer_pos is not None:
                all_positions += [pos]
                distance = np.sqrt((pos[0] - jammer_pos[0]) ** 2 + (pos[1] - jammer_pos[1]) ** 2)
                noise = calculate_noise(distance, jammer_ptx, jammer_gtx, plexp, sigma)
                all_noises += [noise]
                #print(len(all_positions), len(all_noises))
                #gotSomeData = True
                gotSomeData += 1

        #if gotSomeData:
        if gotSomeData > 15:
            gotSomeData = 0
            try:
                #print(all_noises)
                data = pd.DataFrame({
                    'node_positions': [all_positions],
                    'node_noise': [all_noises]
                })
                data = add_jammed_column(data, threshold=-55)
                if pos_jammed is None:
                    print(data)
                    pos_jammed = data['node_positions'][0][data['jammed_at'][0]]
                    pos_jammed = np.array(pos_jammed)
                predicted_jammer_pos = gnn(model, criterion, data)
                #print("Predicted Jammer Position:", predicted_jammer_pos)
                #predicted_gps = cartesian_to_gps(predicted_jammer_pos[0][0], predicted_jammer_pos[0][1], all_positions[0][2])
                predicted_gps = twod_to_gps(predicted_jammer_pos[0][0], predicted_jammer_pos[0][1])
                print("stuff:", predicted_jammer_pos, predicted_gps, pos_jammed)
                distance = np.linalg.norm(pos_jammed - np.array(predicted_jammer_pos))
                print(predicted_gps, distance)
                print(np.sqrt((predicted_jammer_pos[0][0] - jammer_pos[0]) ** 2 + (predicted_jammer_pos[0][1] - jammer_pos[1]) ** 2))
            except Exception as e:
                print(e)
                continue

            responseb = json.dumps({"predicted": predicted_jammer_pos.tolist(), "predicted_gps": predicted_gps, "distance": int(distance)}).encode()
            await conn.publish("srta.jamlocator.prediction", responseb)


if __name__ == '__main__':
    asyncio.run(main())
    #gpsloc = (24.436879692730688, 54.61409845782048, 21.958504397422075)
    #print(cartesian_to_gps(*gps_to_cartesian(*gpsloc)))
