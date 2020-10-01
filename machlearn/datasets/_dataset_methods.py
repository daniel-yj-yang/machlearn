# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np

class Fashion_MNIST_methods(object):
    """
    Some reference: https://github.com/zalandoresearch/fashion-mnist
    """
    
    def __init__(self):
        self.description = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

    def get_label_desc(self, label_int):
        """
        Return the description of an integer label, range [0,9]
        """
        return self.description[label_int]

    def reshape_image(self, ndarray):
        return np.reshape(ndarray, (28, 28))

    def demo(self):
        from . import public_dataset
        X_train, y_train, X_test, y_test = public_dataset('Fashion_MNIST')
        import matplotlib.pyplot as plt
        for i in range(5000, 5005):
            sample = self.reshape_image(X_test[i])
            plt.figure(figsize = (1,1))
            title = f"{self.get_label_desc(y_test[i])}"
            print(title)
            plt.title(title)
            plt.imshow(255 - sample, cmap='gray', vmin=0, vmax=255)
        plt.show()
