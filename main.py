import numpy as np
from netdataset import NetDataset
from utils import accuracy
from net import Net


def epoch(_net, _dataset):
    # un ordre aléatoire pour former les minibatches
    idx = np.arange(_dataset.ntrain)
    np.random.shuffle(idx)
    # ici on suppose que n train est bien divisble par m, sinon on perd des
    # données
    for k in range(0, _dataset.ntrain, _net.m):
        # entrée du réseau: les m images du minibatch mises en colonne
        a0 = np.column_stack([_dataset.images_train[idx[i]] for i in range(k, k + _net.m)])

        a3, _, a2, z2, a1, z1 = _net.forward(a0)

        # sortie attendue
        # pour chaque image un vecteur avec un 1 à l'indice correspondant à
        # la bonne réponse, et 0 sinon.
        # on empile en colonne les vecteurs pour chaque image du minibatch
        y = np.column_stack([_dataset.ys_train[idx[i]] for i in range(k, k + _net.m)])

        _net.backward(a3, a2, z2, a1, z1, a0, y) 

def train(_net, _dataset, nepochs):
    print(f'Initially: accuracy {accuracy(_net, _dataset)}')
    for e in range(nepochs):
        epoch(_net, _dataset)
        print(f'Epoch {e}: accuracy {accuracy(_net, _dataset)}')

def main():
    net = Net()
    dataset = NetDataset(net)
    train(net, dataset, nepochs = 30)

if __name__ == '__main__':
    main()
