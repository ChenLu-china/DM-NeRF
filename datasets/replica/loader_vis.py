import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def labeltoimg(instance_label, colors):
    # instance_label = np.argmax(instance_label, axis=-1)
    unique_labels = np.unique(instance_label).astype(np.int64)
    ra_se_im_t = np.zeros(shape=(instance_label.shape[0], instance_label.shape[1], 3))
    for i in unique_labels:
        if i == unique_labels[-1]:
            ra_se_im_t[instance_label == i] = [0, 0, 0]
        else:
            ra_se_im_t[instance_label == i] = colors[i]
    return ra_se_im_t.astype(np.uint8)


# visualize input isntance map and rgb image into one image
def vis_segmentation(img, seg_img):
    plt.figure()
    plt.imshow(img)
    plt.imshow(seg_img, alpha=0.7)
    plt.axis('off')
    plt.show()


def vis_selected_pixels(semantic_instances, selected_obj_coords, instance_colors):
    bts, H, W = semantic_instances.shape
    index_x, index_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    coords = np.stack([index_y, index_x], axis=-1).reshape((-1, 2))  # H*Wx2
    vis_img1 = semantic_instances[0]
    selected_obj_coord = selected_obj_coords[0].astype(np.int64)
    selected_obj_coord = selected_obj_coord[selected_obj_coord >= 0]
    selected_obj_coord = coords[selected_obj_coord].astype(np.int64)
    append_channel = np.ones_like(vis_img1) * 100
    append_channel[selected_obj_coord[:, 0], selected_obj_coord[:, 1]] = 255
    append_channel = append_channel[..., np.newaxis].astype(np.uint8)
    vis_img1 = labeltoimg(vis_img1, instance_colors)
    vis_img1 = np.concatenate((vis_img1, append_channel), axis=-1)
    vis_img1 = Image.fromarray(vis_img1, mode='RGBA')
    vis_img1.save('./clicked_pixels_vis.png')
