import numpy as np
import matplotlib.pyplot as plt

def imshow_multi(imgs, labels):
    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax, label in zip(imgs, axs, labels):
        ax.imshow(img)
        ax.set_title(label)
    plt.show()