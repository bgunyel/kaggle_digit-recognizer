import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

import neural_nets


def update(model, device, trainLoader, optimizer):
    model.train()

    numberOfBatches = len(trainLoader)
    batchSamplingDistance = min(numberOfBatches, 50)
    numberOfSamplingPoints = numberOfBatches // batchSamplingDistance + 1

    averageLoss = 0
    averageAccuracy = 0

    for batch_idx, (data, target) in enumerate(trainLoader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()

        if batch_idx % batchSamplingDistance == 0:
            averageLoss += loss.item()
            prediction = out.argmax(dim=1, keepdim=True)
            correct = prediction.eq(target.view_as(prediction))
            averageAccuracy += correct.sum().item() / len(correct)

    averageLoss /= numberOfSamplingPoints
    averageAccuracy /= numberOfSamplingPoints

    return averageLoss, averageAccuracy


def validate(model, device, validationDataLoader):
    model.eval()

    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in validationDataLoader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)

            loss += F.nll_loss(out, target, reduction='sum').item()  # sum up batch loss
            prediction = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(validationDataLoader.dataset)
    accuracy = correct / len(validationDataLoader.dataset)

    return loss, accuracy


def train(model, device, trainDataLoader, validationDataLoader, optimizer, numberOfEpochs):
    trainingLossVector = np.zeros(numberOfEpochs)
    trainingAccuracyVector = np.zeros(numberOfEpochs)
    validationLossVector = np.zeros(numberOfEpochs)
    validationAccuracyVector = np.zeros(numberOfEpochs)

    for epoch in range(numberOfEpochs):
        trLoss, trAccuracy = update(model=model, device=device, trainLoader=trainDataLoader, optimizer=optimizer)
        vlLoss, vlAccuracy = validate(model=model, device=device, validationDataLoader=validationDataLoader)

        trainingLossVector[epoch] = trLoss
        trainingAccuracyVector[epoch] = trAccuracy
        validationLossVector[epoch] = vlLoss
        validationAccuracyVector[epoch] = vlAccuracy

    return trainingLossVector, trainingAccuracyVector, validationLossVector, validationAccuracyVector


def main(params):
    print(params['name'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    dataFile = params['data']
    outFolder = params['out']

    plotting_enabled = params['plotting_enabled']

    df = pd.read_csv(dataFile)

    # data = df[0:1280]
    data = df
    Y = data['label'].to_numpy()

    X = data.drop(['label'], axis="columns").to_numpy(dtype='float32')  # drop the labels, get the image data
    del data  # Delete the initial data frame
    del df

    avg = np.mean(X)
    stdDev = np.std(X)

    X = (X - avg) / stdDev
    print(f'Mean Pixel Value: {avg}')
    print(f'Standard Deviation Pixel Value: {stdDev}')

    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, train_size=0.9, random_state=42, stratify=Y)

    Xtrain, Ytrain, Xval, Yval = map(torch.tensor, (Xtrain, Ytrain, Xval, Yval))
    Xtrain = torch.reshape(Xtrain, (-1, 1, 28, 28))
    Xval = torch.reshape(Xval, (-1, 1, 28, 28))

    # DataLoader
    train_ds = TensorDataset(Xtrain, Ytrain)
    train_dl = DataLoader(train_ds, batch_size=params['batch size'])

    validation_ds = TensorDataset(Xval, Yval)
    validation_dl = DataLoader(validation_ds, batch_size=params['validation batch size'])

    model = neural_nets.NeuralNet().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['learning rate'])

    t1 = time.time()

    trLossVector, trAccuracyVector, vlLossVector, vlAccuracyVector = train(model=model,
                                                                           device=device,
                                                                           trainDataLoader=train_dl,
                                                                           validationDataLoader=validation_dl,
                                                                           optimizer=optimizer,
                                                                           numberOfEpochs=
                                                                           params['epochs'])
    t2 = time.time()
    print(f'Training took {t2 - t1} seconds')

    torch.save(model.state_dict(), "./out/mnist_cnn.pt")

    # pyplot.imshow(X[0].reshape((28, 28)), cmap="gray")

    if plotting_enabled:
        plt.plot(range(params['epochs']), trLossVector, label='Training Loss')
        plt.plot(range(params['epochs']), vlLossVector, label='Validation Loss')
        plt.legend(title='Loss')
        plt.xlabel('Epoch number')

        plt.figure()
        plt.plot(range(params['epochs']), trAccuracyVector, label='Training Accuracy')
        plt.plot(range(params['epochs']), vlAccuracyVector, label='Validation Accuracy')
        plt.legend(title='Accuracy')
        plt.xlabel('Epoch number')
        plt.show()

    dummy = -32


if __name__ == '__main__':
    parameters = {'name': 'MNIST Training',
                  'data': './data/train.csv',
                  'out': './out/',
                  'batch size': 128,
                  'epochs': 100,
                  'validation batch size': 128,
                  'learning rate': 0.001,
                  'plotting_enabled': False}

    t1 = time.time()
    main(parameters)
    t2 = time.time()
    print(f'Execution took {t2-t1} seconds...')
