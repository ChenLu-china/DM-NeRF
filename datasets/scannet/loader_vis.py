import numpy as np
import matplotlib.pyplot as plt
import cv2


def ins2img(gt_label, colors):
    unique_labels = np.unique(gt_label)
    rgb = [0, 0, 0]
    h, w = gt_label.shape
    ra_in_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        if label == -1:
            ra_in_im_t[gt_label == label] = rgb
        else:
            ra_in_im_t[gt_label == label] = colors[label]
    return ra_in_im_t.astype(np.uint8)


def ins2img_full(gt_label, colors):
    unique_labels = np.unique(gt_label)
    h, w = gt_label.shape
    ra_in_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        ra_in_im_t[gt_label == label] = colors[label]
    return ra_in_im_t.astype(np.uint8)


def vis_segmentation(img, seg_img):
    img = (img * 255).astype(np.uint8)
    plt.figure()
    plt.imshow(img)
    plt.imshow(seg_img, alpha=0.7)
    plt.axis('off')
    plt.show()


def vis_crop(img):
    img = (img * 255).astype(np.uint8)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def vis_load(images, depths):
    N, H, W, C = images.shape
    random_index = np.random.randint(0, len(images), 1)
    fig = plt.figure(num=1, figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    image = images[random_index].reshape([H, W, C])
    H, W = 480, 640
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
    ax1.imshow(image)
    ax1.axis('off')
    depths = depths / 1000.
    depth = depths[random_index]
    depths_image = np.zeros_like(image)
    depths_image[..., -1] = (depth / np.max(depth)) * 255
    ax2 = fig.add_subplot(122)
    ax2.imshow(depths_image)
    ax2.axis('off')
    plt.show()
