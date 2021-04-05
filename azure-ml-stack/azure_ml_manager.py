import json
from azureml.core import Workspace

class MLWorkspace:
    def __init__(self, config_file_path):
        config_file = open(config_file_path)
        self.config_json = json.load(config_file)
        try:
            self.ws = Workspace.create(name=self.config_json['workspace_name'], # provide a name for your workspace
                        subscription_id=self.config_json['subscription_id'], # provide your subscription ID
                        resource_group=self.config_json['resource_group'], # provide a resource group name
                        create_resource_group=True,
                        location='westus2')
        except:
            print('The workspace already exists. Use existing workspace.')
            self.ws = Workspace.from_config()
    
    def write_config(self):
        self.ws.write_config(path='.azureml')

import datetime
import time
from azureml.core.compute import ComputeTarget, ComputeInstance, AmlCompute
from azureml.core.compute_target import ComputeTargetException

class MLComputeCluster:
    def __init__(self, ws):
        self.ws = ws

    def initiate_compute_cluster(self):
        self.compute_cluster_name = "ci{}".format(self.ws._workspace_id)[:10]
        try:
            self.target_instance = ComputeTarget(workspace=self.ws, name=self.compute_cluster_name)
            print('Found existing instance, use it.')
        except ComputeTargetException:
            self.compute_config = AmlCompute.provisioning_configuration(
                vm_size='STANDARD_D2_V2',
                idle_seconds_before_scaledown=2400,
                min_nodes=0,
                max_nodes=4
            )
            self.target_instance = ComputeTarget.create(self.ws, self.compute_cluster_name, self.compute_config)
            self.target_instance.wait_for_completion(show_output=True)

    def compute_instance_get_status(self):
        self.target_instance.get_status()

    # def compute_instance_stop(self):
    #     self.target_instance.stop(wait_for_completion=True, show_output=True)

    # def compute_instance_start(self):
    #     self.target_instance.start(wait_for_completion=True, show_output=True)

    # def compute_instance_restart(self):
    #     self.target_instance.restart(wait_for_completion=True, show_output=True)

    def compute_instance_delete(self):
        self.target_instance.delete(wait_for_completion=True, show_output=True)

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice, AksWebservice, AksEndpoint
from azureml.core.compute import AksCompute, ComputeTarget
import pandas as pd

class MLExperiment:
    def __init__(self, ws):
        self.ws = ws
    
    def create_experiment(self, experiment_name):
        self.experiment = Experiment(workspace = self.ws, name = experiment_name)

    def get_datastore_experiment(self, target_path):
        self.datastore = self.ws.get_default_datastore()
        self.dataset = Dataset.File.from_files(path=(self.datastore, target_path))

    def config_experiment(self, script_name, compute_cluster_name):
        self.config = ScriptRunConfig(source_directory='./src', script=script_name, compute_target=compute_cluster_name, arguments=['--data_path', self.dataset.as_named_input('input').as_mount()])

    def config_automl_experiment(self, task, X, y, iteration = 30, iteration_timeout_min = 5, metric = "AUC_weighted", n_cross_validation = 5):
        self.config = AutoMLConfig(task=task,
                             X=X,
                             y=y,
                             iterations=iteration,
                             iteration_timeout_minutes=iteration_timeout_min,
                             primary_metric=metric,
                             n_cross_validations=n_cross_validation
                            )

    def get_best_model(self):
        self.best_model = self.run.get_output()

    def predict(self, X_test):
        self.y_predict = self.best_model.predict(X_test)

    def set_env_experiment(self, environment_name, env_file_path):
        self.env = Environment.from_conda_specification(name=environment_name, file_path=env_file_path)
        self.config.run_config.environment = self.env

    def run_experiment(self):
        self.run = self.experiment.submit(self.config)
        self.run.wait_for_completion(show_output=True)
        self.aml_url = self.run.get_portal_url()
        print(self.aml_url)

    def register_model(self, model_name):
        self.model_name = model_name
        self.model = self.run.register_model(model_name=self.model_name, model_path='./outputs/model.pkl')
        print(self.model.name, self.model.id, self.model.version, sep='\t')

    def set_inference_config(self):
        self.inference_config = InferenceConfig(entry_script="./src/score.py",
                                   environment=self.env)

    def set_deploy_config(self, mode='aci', cpu_core=1, memory_gb=1):
        self.deploy_mode = mode
        if(self.deploy_mode=='aci'):
            self.deployment_config = AciWebservice.deploy_configuration(cpu_cores = cpu_core, memory_gb = memory_gb)
        elif(self.deploy_mode=='aks'):
            self.deploy_compute = ComputeTarget(self.ws, 'aks')
            self.deployment_config = AksEndpoint.deploy_configuration(cpu_cores = cpu_core, memory_gb = memory_gb,
                                                              enable_app_insights = True,
                                                              version_name=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                                                              traffic_percentile = 20)

    def deploy_model(self):
        model = Model(self.ws, name=self.model_name)
        if(self.deploy_mode=='aci'):
            try:
                self.service = Webservice(name=self.model_name, workspace=self.ws)
                print('Update existing service')
                self.service.update(models=[model], inference_config=self.inference_config)
            except Exception as e:
                print(e.message)
                print('Create new service')
                self.service = Model.deploy(self.ws, self.model_name, [model], self.inference_config, self.deployment_config, overwrite=True)
                
        elif(self.deploy_mode=='aks'):
            self.service = Model.deploy(self.ws, self.model_name, [model], self.inference_config, self.deployment_config, self.deploy_compute)
        self.service.wait_for_deployment(True)
        print(self.service.state)
        print("scoring URI: " + self.service.scoring_uri)

class MLData:
    def __init__(self, ws):
        self.ws = ws
        self.datastore = self.ws.get_default_datastore()

    def data_upload(self, target_path):
        self.datastore.upload(src_dir='./data',
                 target_path=target_path,
                 overwrite=True)
    



    
    
    