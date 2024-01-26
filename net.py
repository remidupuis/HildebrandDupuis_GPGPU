import numpy as np
from math import sqrt
from utils import sigmoid, sigmoid_prime

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

        # taille des mini-batches
        self.m = 10

        # coefficient pour l'inertie
        self.gamma = 0.4

        # paramètres du réseau
        # couche cachée 1
        self.w1 = np.random.normal(0.0, 1.0/sqrt(self.ninputs), (self.ninputs, self.nhidden1))
        # self.w1 = torch.normal(0.0, 1.0, (self.nhidden1, self.ninputs))
        self.b1 = np.zeros(shape = (self.nhidden1, 1))

        self.uw1 = np.zeros(shape = (self.ninputs, self.nhidden1))
        self.ub1 = np.zeros(shape = (self.nhidden1, 1))
        
        # couche cachée 2
        self.w2 = np.random.normal(0.0, 1.0/sqrt(self.nhidden1), (self.nhidden1, self.nhidden2))
        self.b2 = np.zeros(shape = (self.nhidden2, 1))

        self.uw2 = np.zeros(shape = (self.nhidden1, self.nhidden2))
        self.ub2 = np.zeros(shape = (self.nhidden2, 1))

        # couche de sortie
        self.w3 = np.random.normal(0.0, 1.0/sqrt(self.nhidden2), (self.nhidden2, self.noutputs))
        self.b3 = np.zeros(shape = (self.noutputs, 1))

        self.uw3 = np.zeros(shape = (self.nhidden2, self.noutputs))
        self.ub3 = np.zeros(shape = (self.noutputs, 1))

    def forward(self, a0):
        'calcul de la prédiction du réseau'
        z1 = np.matmul(self.w1.T, a0) + self.b1
        a1 = sigmoid(z1)
    
        z2 = np.matmul(self.w2.T, a1) + self.b2
        a2 = sigmoid(z2)
    
        z3 = np.matmul(self.w3.T, a2) + self.b3
        a3 = sigmoid(z3)
    
        return a3, z3, a2, z2, a1, z1


    def backward(self, a3, a2, z2, a1, z1, a0, y):
        # erreur modifiée
        d3 = a3 - y
        d2 = np.matmul(self.w3, d3) * sigmoid_prime(z2)
        d1 = np.matmul(self.w2, d2) * sigmoid_prime(z1)
    
        ones = np.ones(shape = (self.m, 1))
    
        # Inertie
        self.uw1 = self.gamma * self.uw1 + self.alpha / self.m * np.matmul(a0, d1.T)
        self.uw2 = self.gamma * self.uw2 + self.alpha / self.m * np.matmul(a1, d2.T)
        self.uw3 = self.gamma * self.uw3 + self.alpha / self.m * np.matmul(a2, d3.T)
    
        self.ub1 = self.gamma * self.ub1 + self.alpha / self.m * np.matmul(d1, ones)
        self.ub2 = self.gamma * self.ub2 + self.alpha / self.m * np.matmul(d2, ones)
        self.ub3 = self.gamma * self.ub3 + self.alpha / self.m * np.matmul(d3, ones)
    
        self.w1 = self.w1 - self.uw1
        self.w2 = self.w2 - self.uw2
        self.w3 = self.w3 - self.uw3
        self.b1 = self.b1 - self.ub1
        self.b2 = self.b2 - self.ub2
        self.b3 = self.b3 - self.ub3
    
