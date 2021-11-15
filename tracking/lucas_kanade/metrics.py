import numpy as np


def ssd(template, img_patch):
    return -np.sum((img_patch - template) ** 2)


def ncc(template, img_patch):
    img_patch = (img_patch - img_patch.mean()) / np.sqrt(np.sum(img_patch ** 2)))
    template = (template - template.mean()) / np.sqrt(np.sum(template ** 2)))
    return (img_patch * template).sum()


def sad(template, img_patch):
    return -np.sum(np.abs(img_patch - template))