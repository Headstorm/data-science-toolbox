from azure_ml_manager import MLWorkspace, MLComputeCluster, MLExperiment, MLData

CONFIG_FILE_PATH = '.azureml/config.json'
DATA_TARGET_PATH = 'datasets/data'

if __name__ == '__main__':
    # Create Azure ML Workspace
    az_ws = MLWorkspace(CONFIG_FILE_PATH)

    # Create Azure ML Compute Cluster
    az_compute_clus = MLComputeCluster(az_ws.ws)
    az_compute_clus.initiate_compute_cluster()

    # Upload data
    az_data = MLData(az_ws.ws)
    az_data.data_upload(target_path=DATA_TARGET_PATH)

    # Run Experiment
    az_control_script_train = MLExperiment(az_ws.ws)
    az_control_script_train.create_experiment(experiment_name='day1-experiment-train')
    az_control_script_train.get_datastore_experiment(target_path=DATA_TARGET_PATH)
    az_control_script_train.config_experiment(script_name='train.py', compute_cluster_name=az_compute_clus.compute_cluster_name)
    # az_control_script_train.config_automl_experiment(task='classification', X, y, iteration = 30, iteration_timeout_min = 5, metric = "AUC_weighted", n_cross_validation = 5)
    az_control_script_train.set_env_experiment('azure_env', './.azureml/azure_env.yml')
    az_control_script_train.run_experiment()

    # Register and Deploy Model
    az_control_script_train.register_model(model_name='model')
    az_control_script_train.set_inference_config()
    az_control_script_train.set_deploy_config(mode='aci')
    az_control_script_train.deploy_model()

    
