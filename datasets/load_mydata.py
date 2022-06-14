import torch
import numpy as np
from datasets.mydata import data_loader
from datasets.mydata.loader_vis import vis_segmentation, to8b, labeltoimg, vis_selected_pixels

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_data(args):
    # load color image RGB
    imgs, poses, i_split, camera_angle_x = data_loader.rgb_processor(args.datadir, testskip=args.testskip).load_rgb()
    # add another load class which assigns to semantic labels
    imgs = imgs[..., :3]
    # load instance labels
    ins_info = data_loader.ins_processor(args.datadir, None, None, testskip=args.testskip)
    gt_labels = ins_info.gt_labels
    ins_rgbs = ins_info.ins_rgbs

    H, W = imgs[0].shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([[focal, 0, W * 0.5], [0, -focal, H * 0.5], [0, 0, -1]])
    hwk = [int(H), int(W), K]
    # ins_img = labeltoimg(torch.LongTensor(gt_labels[80]), ins_rgbs)
    # rgb8 = to8b(imgs[80])
    # vis_segmentation(rgb8, ins_img)
    # vis_selected_pixels(gt_labels, ins_indices, ins_rgbs)

    return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_info.ins_num
