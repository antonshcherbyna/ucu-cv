import numpy as np
import cv2


def template_matching(img, template, metric):

    assert metric is not None, 'Please, provide fucntion to measure template proximity!'

    img = img.astype(float)
    template = template.astype(float)

    img_height, img_width = img.shape
    template_height, template_width = template.shape

    scores = np.zeros((img_height - template_height + 1, img_width - template_width + 1))
    scores_height, scores_width = scores.shape

    # TODO: change it to im2col like in NNs for better performance
    for y in range(scores_height):
        for x in range(scores_width):
            img_patch = img[y : y + template_height, x: x + template_width]
            scores[y, x] = metric(template, img_patch)

    _, _, _, best_match = cv2.minMaxLoc(scores)

    return best_match