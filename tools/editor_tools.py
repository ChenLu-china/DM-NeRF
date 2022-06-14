import torch
import os
import imageio
import numpy as np
import time
import torch.nn.functional as F
from tools.helpers_tools import get_rays, sample_pdf
from networks.ins_eval import to8b
from tools.vis_tools import editor_label2img


def exchanger(ori_raw, tar_raw, move_label):
    """
    :param q_pre_label:
    :param e_pre_label:
    :param t_label:
    :return operation_mask: -1 means not exchange, 0 means eliminate, 1 means exchange
    """
    ori_pred_ins = ori_raw[..., 4:]
    ori_pred_ins = torch.sigmoid(ori_pred_ins)
    ori_pred_label = torch.argmax(ori_pred_ins, dim=-1)  # 0-32

    tar_pred_ins = tar_raw[..., 4:]
    tar_pred_ins = torch.sigmoid(tar_pred_ins)
    tar_pred_label = torch.argmax(tar_pred_ins, dim=-1)  # 0-32

    operation_mask = torch.zeros_like(ori_pred_label)
    ori_move_mask, tar_move_mask = torch.zeros_like(ori_pred_label), torch.zeros_like(tar_pred_label)
    ori_move_mask[ori_pred_label == move_label] = -2
    tar_move_mask[tar_pred_label == move_label] = 1

    reduced_mask = tar_move_mask - ori_move_mask

    operation_mask[reduced_mask == 0] = -1
    operation_mask[reduced_mask == 1] = 1
    operation_mask[reduced_mask == 2] = 0
    operation_mask[reduced_mask == 3] = 1

    '''-1 means not exchange, 0 means eliminate, 1 means exchange'''
    ori_raw[operation_mask == 1] = tar_raw[operation_mask == 1]
    ori_raw[operation_mask == 0] = ori_raw[operation_mask == 0] * 0

    return ori_raw, tar_raw, ori_pred_label, tar_pred_label


def editor_render(raw, z_vals, rays_d):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    ins_labels = raw[..., 4:]
    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    ins_map = torch.sum(weights[..., None] * ins_labels, -2)  # [N_rays, 16]
    ins_map = torch.sigmoid(ins_map)
    # semantic_map = torch.softmax(semantic_map, dim=1)
    depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # acc_map = torch.sum(weights, -1)

    return rgb_map, weights, depth_map, ins_map


def editor_nerf(rays, position_embedder, view_embedder, model, N_samples=None, near=None, far=None, z_vals=None):
    rays_o, rays_d = rays
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # flatten

    N_rays, c = rays_d.shape

    if z_vals is None:
        near_, far_ = near * torch.ones(size=(N_rays, 1)), far * torch.ones(size=(N_rays, 1))  # N_rays,
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near_ * (1. - t_vals) + far_ * t_vals
        z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    embedded_pos = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded_pos, embedded_dirs], -1)

    raw = model(embedded)
    raw = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])  # B,N_sample,rgb+density+instance

    return raw, z_vals


def editor_ins(position_embedder, view_embedder, model_fine, ori_rays, tar_rays, args):
    """
        change stratagy, the integral and move color are high accuracy, so we just eliminate the color's value of points
        which has the final label and the position betweeen the camera and the plane.
    """
    # extract parameter
    N_samples, N_importance, near, far = args.N_samples, args.N_importance, args.near, args.far

    """first step"""
    # sample 64 points
    ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, N_samples, near, far)
    tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, N_samples, near, far)

    # exchange
    ori_raw, tar_raw, ori_pred_label, tar_pred_label = exchanger(ori_raw, tar_raw, args.target_label)

    # calculate weights
    ori_rgb, ori_weights, ori_depth, ori_ins = editor_render(ori_raw, ori_z_vals, ori_rays[1])
    tar_rgb, tar_weights, tar_depth, tar_ins = editor_render(tar_raw, tar_z_vals, tar_rays[1])

    """step2"""
    # reset weights
    tar_z_vals_mid = .5 * (tar_z_vals[..., 1:] + tar_z_vals[..., :-1])
    tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)
    tar_z_vals_unreset, _ = torch.sort(torch.cat([tar_z_vals, tar_z_samples], dim=-1), dim=-1)
    tar_raw_unreset, tar_z_vals_unreset = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine,
                                                      z_vals=tar_z_vals_unreset)
    tar_rgb, tar_weights_unreset, tar_depth, tar_ins = editor_render(tar_raw_unreset, tar_z_vals_unreset, tar_rays[1])
    weights_mask = (tar_pred_label == args.target_label).type(torch.float32)
    tar_weights = weights_mask * tar_weights

    # resample 128 points respectively
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points
    tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)

    # cat all the points 64+128+128
    ori_z_vals, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples, tar_z_samples], dim=-1), dim=-1)
    tar_z_vals, _ = torch.sort(torch.cat([tar_z_vals, ori_z_samples, tar_z_samples], dim=-1), dim=-1)

    ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, z_vals=ori_z_vals)
    tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, z_vals=tar_z_vals)

    ori_raw, tar_raw, ori_pred_label, tar_pred_label = exchanger(ori_raw, tar_raw, args.target_label)

    # final render a rgb and ins map
    final_rgb, final_weights, final_depth, final_ins = editor_render(ori_raw, ori_z_vals, ori_rays[1])

    return final_rgb, final_ins, tar_rgb


def editor_test(position_embedder, view_embedder, model_fine, ori_pose, hwf, trans_dicts, save_dir, ins_rgbs, args):
    """
            the first step to find the
    """
    """move_object must between 1 to args.class_number"""
    if ori_pose.shape != [4, 4]:
        ori_pose = torch.cat((ori_pose, torch.tensor([[0, 0, 0, 1]])), dim=0)
    H, W, focal = hwf
    ori_rgbs = []
    ins_imgs = []
    # original
    ori_rays_o, ori_rays_d = get_rays(H, W, focal, torch.Tensor(ori_pose))
    ori_rays_o = torch.reshape(ori_rays_o, [-1, 3]).float()
    ori_rays_d = torch.reshape(ori_rays_d, [-1, 3]).float()
    for index, trans_dict in enumerate(trans_dicts):
        time_0 = time.time()
        trans = torch.Tensor(trans_dict['transformation'])
        tar_pose = trans @ ori_pose
        tar_rays_o, tar_rays_d = get_rays(H, W, focal, torch.Tensor(tar_pose))
        tar_rays_o = torch.reshape(tar_rays_o, [-1, 3]).float()
        tar_rays_d = torch.reshape(tar_rays_d, [-1, 3]).float()
        full_rgb, full_ins, full_tar_rgb = None, None, None
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
            # original view rays render
            ori_rays_io = ori_rays_o[step:step + N_test]  # (chuck, 3)
            ori_rays_id = ori_rays_d[step:step + N_test]  # (chuck, 3)
            ori_batch_rays = torch.stack([ori_rays_io, ori_rays_id], dim=0)
            # target view rays render
            tar_rays_io = tar_rays_o[step:step + N_test]  # (chuck, 3)
            tar_rays_id = tar_rays_d[step:step + N_test]  # (chuck, 3)
            tar_batch_rays = torch.stack([tar_rays_io, tar_rays_id], dim=0)
            # edit render
            ori_rgb, ins, tar_rgb = editor_ins(position_embedder, view_embedder, model_fine, ori_batch_rays, tar_batch_rays,
                                           args)
            # all_info = ins_nerf(tar_batch_rays, position_embedder, view_embedder, model_fine, model_coarse, args)
            if full_rgb is None and full_ins is None:
                full_rgb, full_ins, full_tar_rgb = ori_rgb, ins, tar_rgb
            else:
                full_rgb = torch.cat((full_rgb, ori_rgb), dim=0)
                full_ins = torch.cat((full_ins, ins), dim=0)
                full_tar_rgb = torch.cat((full_tar_rgb, tar_rgb), dim=0)
        ori_rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])
        ori_rgb = ori_rgb.cpu().numpy()
        ori_rgb = ori_rgb.reshape([H, W, 3])
        ori_rgb = to8b(ori_rgb)
        tar_rgb = full_tar_rgb.reshape([H, W, full_tar_rgb.shape[-1]])
        tar_rgb = tar_rgb.cpu().numpy()
        tar_rgb = tar_rgb.reshape([H, W, 3])
        tar_rgb = to8b(tar_rgb)
        ins = full_ins.reshape([H, W, full_ins.shape[-1]])
        label = torch.argmax(ins, dim=-1)
        label = label.reshape([H, W])
        ins_img = editor_label2img(label, ins_rgbs)

        # generate a video
        ori_rgbs.append(ori_rgb)
        ins_imgs.append(ins_img)
        img_file = os.path.join(save_dir, f'img_{trans_dict["mode"]}_{str(index).zfill(3)}.png')
        imageio.imwrite(img_file, ori_rgb)
        ins_file = os.path.join(save_dir, f'ins_{trans_dict["mode"]}_{str(index).zfill(3)}.png')
        imageio.imwrite(ins_file, ins_img)
        tar_img_file = os.path.join(save_dir, f'tar_img_{trans_dict["mode"]}_{str(index).zfill(3)}.png')
        imageio.imwrite(tar_img_file, tar_rgb)
        time_1 = time.time()
        print(f'IMAGE[{index}] TIME: {np.round(time_1 - time_0, 6)} second')
    # generation video
    imageio.mimwrite(os.path.join(save_dir, f'editor_video.mp4'), ori_rgbs, fps=len(ori_rgbs) // 6, quality=8)
