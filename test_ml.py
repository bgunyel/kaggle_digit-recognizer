import os
import time
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb

import utils


def main(params):
    print(params['name'])

    model = pickle.load(open(params['model'], "rb"))

    dataFile = params['data']
    outFolder = params['out']
    hog_params = params['hog']

    df = pd.read_csv(dataFile)
    data = df
    number_of_images = data.shape[0]

    X = data.to_numpy(dtype='float32')

    predictions = -1 * np.ones(number_of_images)

    for idx in range(number_of_images):
        print(f'idx: {idx} / {number_of_images}')
        image = np.reshape(X[idx, :], (28, 28))
        predictions[idx] = utils.predict(model=model, params=hog_params, image=image)

    submission = pd.DataFrame(
        data={'ImageId': range(1, len(predictions) + 1), "Label": predictions.astype(int).ravel()})
    submission.to_csv(os.path.join(outFolder, 'submission.csv'), index=False)


    dummy = -32





if __name__ == '__main__':
    parameters = {'name': 'MNIST Test',
                  'data': './data/test.csv',
                  'model': './out/mnist_xgb.pkl',
                  'hog': {'number_of_orientations': 8, 'pixels_per_cell': (8, 8), 'cells_per_block': (1, 1)},
                  'out': './out/'}

    t1 = time.time()
    main(parameters)
    t2 = time.time()
    print(f'Execution took {t2 - t1} seconds...')