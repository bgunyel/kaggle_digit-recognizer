import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import neural_nets


def main(params):
    print(params['name'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'Device: {device}')

    model = neural_nets.NeuralNet().to(device)
    model.load_state_dict(torch.load(params['model']))
    model.eval()


    dataFile = params['data']
    outFolder = params['out']
    batchSize = params['batch size']

    df = pd.read_csv(dataFile)
    data = df

    numberOfImages = data.shape[0]

    pixelMean = 33.408935546875
    pixelStdDev = 78.6775894165039

    X = data.to_numpy(dtype='float32')
    X = (X - pixelMean) / pixelStdDev
    del df
    del data

    X = torch.tensor(X)
    X = torch.reshape(X, (-1, 1, 28, 28))

    # DataLoader
    test_ds = TensorDataset(X)
    test_dl = DataLoader(test_ds, batch_size=batchSize, shuffle=False)
    predictions = -1 * np.ones(numberOfImages)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            batch = batch[0].to(device)
            out = model(batch)
            prediction = out.argmax(dim=1, keepdim=True).cpu().numpy().astype(int).ravel()
            predictions[batch_idx * batchSize: (batch_idx + 1) * batchSize] = prediction

    submission = pd.DataFrame(
        data={'ImageId': range(1, len(predictions) + 1), "Label": predictions.astype(int).ravel()})
    submission.to_csv(os.path.join(outFolder, 'submission.csv'), index=False)

    dummy = -32


if __name__ == '__main__':
    parameters = {'name': 'MNIST Test',
                  'data': './data/test.csv',
                  'model': './out/mnist_cnn.pt',
                  'out': './out/',
                  'batch size': 1000}
    main(parameters)
