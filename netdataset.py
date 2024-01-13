import numpy as np
import torchvision

class NetDataset:
    def __init__(self, _net):
        # charge les images et étiquettes pour l'entraînement
        train_set, test_set = self.get_data() 

        self.images_train = (train_set.data.view(-1, 28 * 28) / 255).numpy()
        self.labels_train = (train_set.targets).numpy()
        self.ys_train = [np.eye(_net.noutputs)[label].astype(np.float32) for label in self.labels_train]

        # pareil pour le test
        self.images_test = (test_set.data.view(-1, 28 * 28) / 255).numpy()
        self.labels_test = (test_set.targets).numpy()
    
    def get_data(self, dataset_name = 'mnist'):
        if dataset_name == 'mnist':
            return (torchvision.datasets.MNIST(root = '.', train = train, download = True) for train in [True, False])
        else:
            raise NotImplementedError('Not implemented for other datasets yet')
        
    # nombres d'images dans l'ensemble d'apprentissage et celui de test
    @property
    def ntrain(self):
        return 60_000
    
    @property
    def ntest(self):
        return 10_000
