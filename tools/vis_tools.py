import torch
import numpy as np
import matplotlib.pyplot as plt



# some function to help changing labels to rgb images. Including Semantic and Instance both.
def ins2img(predicted_onehot, rgbs):
    predicted_labels = torch.argmax(predicted_onehot, dim=-1)
    predicted_labels = predicted_labels.cpu()
    unique_labels = torch.unique(predicted_labels)
    rgb = [0, 0, 0]
    h, w = predicted_labels.shape
    ra_in_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        if label == 0:
            ra_in_im_t[predicted_labels == label] = rgb
        else:
            ra_in_im_t[predicted_labels == label] = rgbs[label]
    return ra_in_im_t.astype(np.uint8)


# vis instance map after shift
def editor_label2img(predicted_labels, rgbs):
    predicted_labels = predicted_labels.cpu()
    unique_labels = torch.unique(predicted_labels)
    rgb = [0, 0, 0]
    sh = predicted_labels.shape
    ra_in_im_t = np.zeros(shape=(sh[0], sh[1], 3))
    for index, label in enumerate(unique_labels):
        if label == 32:
            ra_in_im_t[predicted_labels == label] = rgb
        else:
            ra_in_im_t[predicted_labels == label] = rgbs[label]
    return ra_in_im_t.astype(np.uint8)


# vis instance map after matching
def matching_labeltoimg(predicted_labels, rgbs):
    unique_labels = torch.unique(predicted_labels).long()
    predicted_labels = predicted_labels.cpu()
    unique_labels = unique_labels.cpu()
    rgb = [0, 0, 0]
    unmatched_rgb = [255, 255, 255]
    h, w = predicted_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        if label == -1:
            ra_se_im_t[predicted_labels == label] = rgb
        elif label == -2:
            ra_se_im_t[predicted_labels == label] = unmatched_rgb
        else:
            ra_se_im_t[predicted_labels == label] = rgbs[label]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t

def render_gt_label2img(gt_labels, rgbs, color_dict):
    unique_labels =  torch.unique(gt_labels)
    gt_labels = gt_labels.cpu()
    unique_labels = unique_labels.cpu()
    h, w = gt_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        label_cpu = str(int(label.cpu()))
        gt_keys = color_dict.keys()
        if label_cpu in gt_keys:
            ra_se_im_t[gt_labels == label] = rgbs[color_dict[str(label_cpu)]]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t

# vis instance at testing phrase
def render_label2img(predicted_labels, rgbs, color_dict, ins_map):
    unique_labels = torch.unique(predicted_labels)
    predicted_labels = predicted_labels.cpu()
    unique_labels = unique_labels.cpu()
    h, w = predicted_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        label_cpu = str(int(label.cpu()))
        gt_keys = ins_map.keys()
        if label_cpu in gt_keys:
            gt_label_cpu = ins_map[label_cpu]
            ra_se_im_t[predicted_labels == label] = rgbs[color_dict[str(gt_label_cpu)]]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t


# visualize input isntance map and rgb image into one image
def vis_segmentation(img, seg_img):
    plt.figure()
    plt.imshow(img)
    plt.imshow(seg_img, alpha=0.7)
    plt.axis('off')
    plt.show()


def vis_img(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return


# get all instance rgbs and corresponding labels
def show_instance_rgb(ins_rgbs, save_rgbs_file):
    show_boxes = np.zeros(shape=[len(ins_rgbs), 8, 8, 3])
    y_ax = 4
    x_ax = int((len(ins_rgbs)) // y_ax)
    fig, ax = plt.subplots(x_ax, y_ax, figsize=(8, 8))
    fontdict = {'fontsize': 6}
    for i in range(len(ins_rgbs)):
        rgb = ins_rgbs[i]
        show_boxes[i, ..., 0:3] = rgb
        x_index, y_index = i // y_ax, i % y_ax
        ax[x_index][y_index].imshow(show_boxes[i].astype(np.uint8))
        ax[x_index][y_index].set_title(f'Label:{i}: [{str(rgb[0])},{str(rgb[1])},{str(rgb[2])}]',
                                       fontdict=fontdict)
        ax[x_index][y_index].grid(False)
        ax[x_index][y_index].axis('off')
    plt.savefig(save_rgbs_file)
    # plt.show()
    return
