import numpy as np

def sigmoid(x):
    '''Fonction d'activation; fonction logistique'''
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    '''Dérivée de la fonction d'activation; ici fonction logistique'''
    return sigmoid(x) * (1 - sigmoid(x))

def accuracy(_net, _dataset):
    # test de la précision
    a0_test = _dataset.images_test.T
    a3_test, _, _, _, _, _ = _net.forward(a0_test)

    # trouve l'indice du maximum sur chaque colonne. Le résultat est un array d'indices.
    predictions = np.argmax(a3_test, axis = 0)

    # nombre de bonnes prédictions
    # on crée un array de booléens qui valent 1 ssi le label est bon
    ngood = np.sum(predictions == _dataset.labels_test)

    return ngood / _dataset.ntest
