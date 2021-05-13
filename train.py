import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot

import neural_nets


def update(model, device, trainLoader, optimizer):
    model.train()

    numberOfBatches = len(trainLoader)
    batchSamplingDistance = min(numberOfBatches, 50)
    numberOfSamplingPoints = numberOfBatches // batchSamplingDistance

    averageLoss = 0

    for batch_idx, (data, target) in enumerate(trainLoader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, target)
        loss.backward()
        optimizer.step()

        if batch_idx % batchSamplingDistance == 0:
            averageLoss += loss.item()

    averageLoss /= numberOfSamplingPoints

    return averageLoss



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
    validationLossVector = np.zeros(numberOfEpochs)
    validationAccuracyVector = np.zeros(numberOfEpochs)

    for epoch in range(numberOfEpochs):
        trLoss = update(model=model, device=device, trainLoader=trainDataLoader, optimizer=optimizer)
        vlLoss, vlAccuracy = validate(model=model, device=device, validationDataLoader=validationDataLoader)

        trainingLossVector[epoch] = trLoss
        validationLossVector[epoch] = vlLoss
        validationAccuracyVector[epoch] = vlAccuracy


def main(params):

    print(params['name'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    dataFile = params['data']
    outFolder = params['out']

    df = pd.read_csv(dataFile)

    data = df[0:1280]
    Y = data['label'].to_numpy()

    X = data.drop(['label'], axis="columns").to_numpy()  # drop the labels, get the image data
    del data  # Delete the initial data frame
    del df

    avg = np.mean(X)
    stdDev = np.std(X)

    X = (X - avg) / stdDev

    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)

    Xtrain, Ytrain, Xval, Yval = map(torch.tensor, (Xtrain, Ytrain, Xval, Yval))
    Xtrain = torch.reshape(Xtrain, (-1, 1, 28, 28))
    Xval = torch.reshape(Xval, (-1, 1, 28, 28))

    # DataLoader
    train_ds = TensorDataset(Xtrain, Ytrain)
    train_dl = DataLoader(train_ds, batch_size=params['batch size'])  # TODO - transforms

    validation_ds = TensorDataset(Xval, Yval)
    validation_dl = DataLoader(validation_ds, batch_size=params['validation batch size'])  # TODO - transforms

    model = neural_nets.NeuralNet().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['learning rate'])


    train(model=model, device=device, trainDataLoader=train_dl, validationDataLoader=validation_dl, optimizer=optimizer,
          numberOfEpochs=params['epochs'])



    # pyplot.imshow(X[0].reshape((28, 28)), cmap="gray")

    dummy = -32


if __name__ == '__main__':
    params = {'name': 'MNIST Training',
              'data': './data/train.csv',
              'out': './out/',
              'batch size': 128,
              'epochs': 10,
              'validation batch size': 128,
              'learning rate': 0.001}
    main(params)
