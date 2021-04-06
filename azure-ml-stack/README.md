# Azure ML Stack
Azure ML Stack is a template for rapid build of Azure ML project. The template should minimize the time to produce data science solution on Microsoft Azure platform.
This template includes the following components:
- Generate the computing instance
- Upload dataset to datastore
- Create Azure ML Experiment and Setup the environment from YML file
- Run training script
- Save and deploy model as service
The example scripts provided in the template is based on the iris classification model based on scikit-learn.

## Setup
1. Install Python
2. Install miniconda https://docs.conda.io/en/latest/miniconda.html
3. Run ```conda env create -f .azureml/azure_env.yml```
4. Run ```conda activate azure_env```
5. Edit the file ```.azureml/config.json``` to your Azure environment setup
6. (Optional) Install Postman or similar application for service testing

## How to use the template
After you setup your environment locally, you can run ```python setup.py``` to run the template which will populate the Azure ML environment, train a model on the uploaded dataset, and deploy the trained model as a service for you. The ```setup.py``` script can be modified to suit your requirement. If the script is run successfully, you can check your experiment and find the API of the deployed model as a service on Azure ML.

### Dataset
All dataset in ```data``` directory will be uploaded the Azure ML datastore and can be used inside the training script. The recommended type of data are CSV and JSON file. The example file is ```iris.csv```.

### Training Script
The example training script can be found in ```src/train.py```. You can make a separate script to generate machine learning model and use it in the training script. Please check the data loading section in the training script on different type of dataset (CSV/JSON). The output of the training script is the trained model which can be found in output section inside Azure ML experiment.

### Score Script
The ```src/score.py``` script will be triggered when the deployed model is called as an API. The script includes the model importing, the data transformation, and the prediction steps.

### Testing
You can use Postman or similar software to test the deployed service. The HTTP request should be a POST with the body in the format of:
```
[
  <datapoint#1>,
  <datapoint#2>,
  <datapoint#X>
]
```
From the example training script and dataset provided in the template, the body for the request can be:
```
[
  [5.1,3.5,1.4,0.2],
  [4.9,3.0,1.4,0.2],
  [5.6,2.9,3.6,1.3],
  [7.7,2.8,6.7,2.0]
]
```
The service will return the prediction as a respond.
```
[
  Iris-setosa,
  Iris-setosa,
  Iris-versicolor,
  Iris-virginica
]
```
