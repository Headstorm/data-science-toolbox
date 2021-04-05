import os
import argparse
import json
import joblib
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    # Load dataset
    d = args.data_path
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            ### Read JSON files
            # f = open(full_path)
            # data = json.load(f)
            # print(data[:5])

            ### Read CSV files
            names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
            dataset = read_csv(full_path, names=names)

    print(dataset.describe())
    print(dataset.groupby('class').size())
    
    # Split Train-Test Dataset
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Train
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Score
    result = model.score(X_validation, Y_validation)
    print("Accuracy : " + str(result))

    #Save Model
    joblib.dump(model, './outputs/model.pkl')
    
