params = {
    'model': 'GAT',
    'learning_rate': 0.0006703373871168542,
    'weight_decay': 0.00005,
    'test_batch_size': 8,
    'batch_size': 8,
    'dropout_rate': 0.5,
    'num_heads': 4,
    'num_layers': 8,
    'hidden_channels': 128,
    'out_channels': 128,
    'out_features': 3,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'max_epochs': 800,
    '3d': False,
    'coords': 'cartesian',  # opts: 'polar', 'cartesian'
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['mean_noise', 'std_noise', 'range_noise', 'dist_to_centroid', 'sin_azimuth', 'cos_azimuth'],
    'num_neighbors': 'fc',
    'edges': 'knn',  # opts: 'knn', 'proximity'
    'norm': 'unit_sphere',  # opts: 'minmax', 'unit_sphere'
    'activation': False,
    'max_nodes': 400,
    'filtering_proportion': 0.6,
    'grid_meters': 1,
    'ds_method': 'hybrid', # time_window_avg, noise
    'experiments_folder': 'exp_jtx/gat/',
    'dataset_path': 'data/modified_interpolated_controlled_path_data_1000_updated_aoa.pkl',  # combined_fspl_log_distance.csv',  # combined_urban_area.csv
    'test_sets': ['guided_path_data_test_set.csv', 'linear_path_data_test_set.csv'],
    'dynamic': True,
    'train_per_class': False,
    'all_env_data': False,
    'inference': False,
    'reproduce': True,
    'plot_network': False,
    'study': 'dataset',  # dataset, coord_system, feat_engineering, knn_edges, downsampling
    'val_discrite_coeff': 0.25,
    'test_discrite_coeff': 0.1,  # disable discritization -> step_size = 1
    'num_workers': 16,
    'aug': ['crop']
}

