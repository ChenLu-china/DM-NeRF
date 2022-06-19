import torch
import numpy as np
from datasets.replica.data_loader import *

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
    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = list(range(1, total_num, step))
    # test_ids = [x + step // 2 for x in train_ids]
    if args.editor:
        objs, view_id,ins_map, poses, ins_rgbs = processor(args.datadir, train_ids, test_ids, testskip=args.testskip).load_rgb()
        
        if view_id is not None:
            view_poses = np.repeat(poses[view_id][np.newaxis, ...], args.views, axis=0)
        else:
            view_poses = torch.stack(
                [pose_spherical(angle, -65.0, 7.0) for angle in np.linspace(-180, 180, args.views)], 0)
        
        ins_num = len(ins_rgbs)
        H, W = int(480), int(640)
        focal = W / 2.0
        K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])
        hwk = [int(H), int(W), K]

        # ins_img = labeltoimg(torch.LongTensor(gt_labels[40]), ins_rgbs)
        # rgb8 = to8b(imgs[40])
        # vis_segmentation(rgb8, ins_img)
        return objs, view_poses, ins_map, poses, hwk, ins_rgbs, ins_num
    else:
        imgs, poses, i_split = rgb_processor(args.datadir, train_ids, test_ids, testskip=args.testskip).load_rgb()
        # add another load class which assigns to semantic labels

        # load instance labels
        ins_info = ins_processor(args.datadir, train_ids, test_ids, None, None, testskip=args.testskip)
        gt_labels = ins_info.gt_labels
        ins_rgbs = ins_info.ins_rgbs

        # ins_indices = ins_info.ins_indices
        H, W = imgs[0].shape[:2]
        # camera_angle_x = np.pi / 3.0
        # focal = .5 * W / np.tan(.5 * camera_angle_x)
        focal = W / 2.0
        K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])
        hwk = [int(H), int(W), K]
        # ins_img = labeltoimg(torch.LongTensor(instances[80]), instances_colors)
        # rgb8 = to8b(imgs[80])
        # vis_segmentation(rgb8, ins_img)

        return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_info.ins_num
