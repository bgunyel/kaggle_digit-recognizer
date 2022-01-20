import time
import multiprocessing

import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

import utils





def main(params):
    print(params['name'])

    dataFile = params['data']
    outFolder = params['out']
    hog_params = params['hog']

    df = pd.read_csv(dataFile)

    data = df
    Y = data['label'].to_numpy()

    images = data.drop(['label'], axis="columns").to_numpy(dtype='float32')  # drop the labels, get the image data

    del data  # Delete the initial data frame
    del df
    t1 = time.time()
    X = utils.extract_features_for_image_set(images=images, params=hog_params)
    t2 = time.time()
    print(f'Extracting Features took {t2 - t1} seconds')

    # Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, train_size=0.9, random_state=42, stratify=Y)

    xgb_model = xgb.XGBClassifier(objective="multi:softprob",
                                  eval_metric='mlogloss',
                                  tree_method='gpu_hist',
                                  random_state=42)

    xgb_params_distributions = {"colsample_bytree": uniform(0.7, 0.3),
                                "gamma": uniform(0, 0.5),
                                "learning_rate": uniform(0.03, 0.3),  # default 0.1
                                "max_depth": randint(2, 12),  # default 3
                                "n_estimators": randint(100, 150),  # default 100
                                "subsample": uniform(0.6, 0.4)}

    t1 = time.time()
    search = RandomizedSearchCV(xgb_model, param_distributions=xgb_params_distributions, random_state=42, n_iter=10,
                                cv=5, verbose=2, n_jobs=1, return_train_score=False, refit=True)
    t2 = time.time()
    print(f'Randomized Search CV - Configuration took {t2 - t1} seconds')

    t1 = time.time()
    search.fit(X, Y)
    t2 = time.time()
    print(f'Randomized Search CV - Training took {t2 - t1} seconds')


if __name__ == '__main__':
    parameters = {'name': 'MNIST Training',
                  'data': './data/train.csv',
                  'out': './out/',
                  'number_of_cores': multiprocessing.cpu_count() - 1,
                  'hog': {'number_of_orientations': 8, 'pixels_per_cell': (8, 8), 'cells_per_block': (1, 1)},
                  'plotting_enabled': False}

    t1 = time.time()
    main(parameters)
    t2 = time.time()
    print(f'Execution took {t2 - t1} seconds...')
