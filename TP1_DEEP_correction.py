# TP1 DEEP correction
# Didier Lime 2023

import torch
import torchvision
from math import sqrt

# torch.t -> matrix version of transpose(m, d0, d1)
# torch.mm -> matrix version of matmul(m1, m2)

def sigmoid(x):
    '''Fonction d'activation; fonction logistique'''
    return 1/(1 + torch.exp(-x))

def sigmoid_prime(x):
    '''Dérivée de la fonction d'activation; ici fonction logistique'''
    return sigmoid(x) * (1 - sigmoid(x))

class Net:
    def __init__(self):
        # nombre de neurones de la couche d'entrée
        self.ninputs = 784

        # nombre de neurones des couches cachées
        self.nhidden1 = 30
        self.nhidden2 = 20

        # nombre de neurones de la couche de sortie
        self.noutputs = 10

        # pas d'apprentissage
        self.alpha = 0.05

        # nombre d'époques d'apprentissage
        self.nepochs = 30

        # taille des mini-batches
        self.m = 10

        # coefficient pour l'inertie
        self.gamma = 0.4

        # paramètres du réseau
        # couche cachée 1
        self.w1 = torch.normal(0.0, 1.0/sqrt(self.ninputs), (self.ninputs, self.nhidden1))
        # self.w1 = torch.normal(0.0, 1.0, (self.nhidden1, self.ninputs))
        self.b1 = torch.zeros(self.nhidden1, 1)

        self.uw1 = torch.zeros(self.ninputs, self.nhidden1)
        self.ub1 = torch.zeros(self.nhidden1, 1)
        
        # couche cachée 2
        self.w2 = torch.normal(0.0, 1.0/sqrt(self.nhidden1), (self.nhidden1, self.nhidden2))
        # self.w2 = torch.normal(0.0, 1.0, (self.nhidden2, self.nhidden1))
        self.b2 = torch.zeros(self.nhidden2, 1)

        self.uw2 = torch.zeros(self.nhidden1, self.nhidden2)
        self.ub2 = torch.zeros(self.nhidden2, 1)

        # couche de sortie
        self.w3 = torch.normal(0.0, 1.0/sqrt(self.nhidden2), (self.nhidden2, self.noutputs))
        # self.w3 = torch.normal(0.0, 1.0, (self.noutputs, self.nhidden2))
        self.b3 = torch.zeros(self.noutputs, 1)

        self.uw3 = torch.zeros(self.nhidden2, self.noutputs)
        self.ub3 = torch.zeros(self.noutputs, 1)

    def forward(self, a0):
        'calcul de la prédiction du réseau'
        z1 = torch.mm(self.w1.t(), a0) + self.b1
        a1 = sigmoid(z1)
    
        z2 = torch.mm(self.w2.t(), a1) + self.b2
        a2 = sigmoid(z2)
    
        z3 = torch.mm(self.w3.t(), a2) + self.b3
        a3 = sigmoid(z3)
    
        return a3, z3, a2, z2, a1, z1


    def backward(self, a3, a2, z2, a1, z1, a0, y):
        # erreur modifiée
        d3 = a3 - y
        d2 = torch.mm(self.w3, d3) * sigmoid_prime(z2)
        d1 = torch.mm(self.w2, d2) * sigmoid_prime(z1)
    
        ones = torch.ones(self.m, 1)

        # nouveaux paramètres
        # self.w1 = self.w1 - self.alpha / self.m * torch.mm(d1, torch.t(a0))
        # self.w2 = self.w2 - self.alpha / self.m * torch.mm(d2, torch.t(a1))
        # self.w3 = self.w3 - self.alpha / self.m * torch.mm(d3, torch.t(a2))
    
        # self.b1 = self.b1 - self.alpha / self.m * torch.mm(d1, ones)
        # self.b2 = self.b2 - self.alpha / self.m * torch.mm(d2, ones)
        # self.b3 = self.b3 - self.alpha / self.m * torch.mm(d3, ones)
    
        # Inertie
        self.uw1 = self.gamma * self.uw1 + self.alpha / self.m * torch.mm(a0, d1.t())
        self.uw2 = self.gamma * self.uw2 + self.alpha / self.m * torch.mm(a1, d2.t())
        self.uw3 = self.gamma * self.uw3 + self.alpha / self.m * torch.mm(a2, d3.t())
    
        self.ub1 = self.gamma * self.ub1 + self.alpha / self.m * torch.mm(d1, ones)
        self.ub2 = self.gamma * self.ub2 + self.alpha / self.m * torch.mm(d2, ones)
        self.ub3 = self.gamma * self.ub3 + self.alpha / self.m * torch.mm(d3, ones)
    
        self.w1 = self.w1 - self.uw1
        self.w2 = self.w2 - self.uw2
        self.w3 = self.w3 - self.uw3
        self.b1 = self.b1 - self.ub1
        self.b2 = self.b2 - self.ub2
        self.b3 = self.b3 - self.ub3
    
def accuracy(_net, images_test, ntest):
    # test de la précision
    #a0_test = torch.column_stack(images_test)
    #a0_test = torch.column_stack([images_test[i] for i in range(ntest)])
    a0_test = torch.t(images_test)

    a3_test, _, _, _, _, _ = _net.forward(a0_test)

    # trouve l'indice du maximum sur chaque colonne. Le résultat est un array d'indices.
    predictions = torch.argmax(a3_test, axis = 0)

    # nombre de bonnes prédictions
    # on crée un array de booléens qui valent 1 ssi le label est bon
    ngood = torch.sum(predictions == labels_test)

    return ngood / ntest


def epoch(_net):
    # un ordre aléatoire pour former les minibatches
    idx = torch.randperm(ntrain)

    # ici on suppose que n train est bien divisble par m, sinon on perd des
    # données
    for k in range(0, ntrain, _net.m):
        # entrée du réseau: les m images du minibatch mises en colonne
        a0 = torch.column_stack([images_train[idx[i]] for i in range(k, k + _net.m)])

        a3, _, a2, z2, a1, z1 = _net.forward(a0)

        # sortie attendue
        # pour chaque image un vecteur avec un 1 à l'indice correspondant à
        # la bonne réponse, et 0 sinon.
        # on empile en colonne les vecteurs pour chaque image du minibatch
        y = torch.column_stack([ys_train[idx[i]] for i in range(k, k + _net.m)])

        _net.backward(a3, a2, z2, a1, z1, a0, y) 

def train(_net, nepochs):
    print(f'Initially: accuracy {accuracy(_net, images_test, ntest)}')
    for e in range(nepochs):
        epoch(_net)
        print(f'Epoch {e}: accuracy {accuracy(_net, images_test, ntest)}')

# charge les images et étiquettes pour l'entraînement
train_set = torchvision.datasets.MNIST(root = '.', train = True, download = True)
images_train = train_set.data.view(-1, 28 * 28) / 255
labels_train = train_set.targets

# pareil pour le test
test_set = torchvision.datasets.MNIST(root = '.', train = False, download = True)
images_test = test_set.data.view(-1, 28 * 28) / 255
labels_test = test_set.targets

# nombres d'images dans l'ensemble d'apprentissage et celui de test
ntrain = 60000
ntest = 10000

net = Net()
ys_train = [torch.nn.functional.one_hot(labels_train[i], net.noutputs) for i in range(ntrain)]

train(net, 30)
