import torch
import os
import imageio
import numpy as np
import time
import json
import torch.nn.functional as F
from networks.helpers import get_rays_k, sample_pdf
from networks.ins_eval import to8b
from tools.visualisation import render_label2img
import cv2


def exchanger(ori_raw, tar_raws, ori_raw_pre, tar_raw_pres, move_labels):
    """
    :param q_pre_label:
    :param e_pre_label:
    :param t_label:
    :return operation_mask: -1 means not exchange, 0 means eliminate, 1 means exchange
    """
    ori_pred_ins = ori_raw[..., 4:]
    ori_pred_ins = torch.sigmoid(ori_pred_ins)
    ori_pred_label = torch.argmax(ori_pred_ins, dim=-1)  # 0-32

    ori_accu_ins = ori_raw_pre[..., :-1]
    ori_accu_ins = torch.sigmoid(ori_accu_ins)
    ori_accu_label = torch.argmax(ori_accu_ins, dim=-1)  # 0-32
    ori_accu_label = ori_accu_label[:, None].repeat(1, ori_pred_label.shape[-1])

    for idx, move_label in enumerate(move_labels):
        tar_raw = tar_raws[idx]
        tar_raw_pre = tar_raw_pres[idx]
        ######################################################

        ori_is_move = ori_pred_label == move_label
        ori_acc_not_move = ori_accu_label != move_label
        ori_occludes = ori_acc_not_move * ori_is_move
        ori_pred_label[ori_occludes == True] = ori_accu_label[ori_occludes == True]

        ######################################################

        ori_not_move = ori_pred_label != move_label
        ori_accu_move = ori_accu_label == move_label
        fillings = ori_accu_move * ori_not_move
        # ori_pred_label[ccc == True] = move_label

        tar_pred_ins = tar_raw[..., 4:]
        tar_pred_ins = torch.sigmoid(tar_pred_ins)
        tar_pred_label = torch.argmax(tar_pred_ins, dim=-1)  # 0-32
        tar_pred_label_temp = tar_pred_label

        tar_accu_ins = tar_raw_pre[..., :-1]
        tar_accu_ins = torch.sigmoid(tar_accu_ins)
        tar_accu_label = torch.argmax(tar_accu_ins, dim=-1)  # 0-32
        tar_accu_label = tar_accu_label[:, None].repeat(1, tar_pred_label.shape[-1])

        ######################################################

        tar_is_move = tar_pred_label == move_label
        tar_accu_not_move = tar_accu_label != move_label
        tar_occludes = tar_accu_not_move * tar_is_move
        tar_pred_label[tar_occludes == True] = tar_accu_label[tar_occludes == True]

        ######################################################

        operation_mask = torch.zeros_like(ori_pred_label)
        ori_move_mask, tar_move_mask = torch.zeros_like(ori_pred_label), torch.zeros_like(tar_pred_label)
        ori_move_mask[ori_pred_label == move_label] = -2
        tar_move_mask[tar_pred_label == move_label] = 1

        reduced_mask = tar_move_mask - ori_move_mask

        operation_mask[reduced_mask == 0] = -1
        operation_mask[reduced_mask == 1] = 1
        operation_mask[reduced_mask == 2] = 0
        operation_mask[reduced_mask == 3] = 1
        # print(torch.sum(reduced_mask == 2))
        '''-1 means not exchange, 0 means eliminate, 1 means exchange'''
        ######################################################
        ori_raw[fillings] = tar_raw[fillings]
        ######################################################

        ori_raw[operation_mask == 1] = tar_raw[operation_mask == 1]
        ori_raw[operation_mask == 0] = ori_raw[operation_mask == 0] * 0

    return ori_raw, tar_raws, ori_pred_label, tar_pred_label_temp


def exchanger_fine(ori_raw, tar_raw, ori_raw_fine, tar_raw_fine, move_label):
    """
    :param q_pre_label:
    :param e_pre_label:
    :param t_label:
    :return operation_mask: -1 means not exchange, 0 means eliminate, 1 means exchange
    """
    ori_pred_ins = ori_raw_fine[..., 4:]
    ori_pred_ins = torch.sigmoid(ori_pred_ins)
    ori_pred_label = torch.argmax(ori_pred_ins, dim=-1)  # 0-32

    tar_pred_ins = tar_raw_fine[..., 4:]
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


def editor_ins_b(position_embedder, view_embedder, model_coarse, model_fine, ori_rays, tar_rays, args):
    """
            change stratagy, the integral and move color are high accuracy, so we just eliminate the color's value of points
            which has the final label and the position betweeen the camera and the plane.
        """
    # extract parameter
    N_samples, N_importance, near, far = args.N_samples, args.N_importance, args.near, args.far

    """first step"""
    # sample 64 points
    ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_coarse, N_samples, near, far)
    tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_coarse, N_samples, near, far)

    # ori_raw_fine, ori_z_vals_fine = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, N_samples, near,
    #                                             far)
    # tar_raw_fine, tar_z_vals_fine = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, N_samples, near,
    #                                             far)
    # ori
    _, ori_weights, _, _ = editor_render(ori_raw, ori_z_vals, ori_rays[1])
    # sample 128
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points
    ori_z_vals_prec, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples], dim=-1), dim=-1)
    ori_raw_prec, _ = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, N_samples, near, far,
                                  z_vals=ori_z_vals_prec)
    _, _, _, ori_ins_accu = editor_render(ori_raw_prec, ori_z_vals_prec, ori_rays[1])

    # tar
    tar_rgb, tar_weights, tar_depth, tar_ins = editor_render(tar_raw, tar_z_vals, tar_rays[1])
    tar_z_vals_mid = .5 * (tar_z_vals[..., 1:] + tar_z_vals[..., :-1])
    tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)
    tar_z_vals_prec, _ = torch.sort(torch.cat([tar_z_vals, tar_z_samples], dim=-1), dim=-1)
    tar_raw_prec, _ = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine,
                                  z_vals=tar_z_vals_prec)
    _, _, _, tar_ins_accu = editor_render(tar_raw_prec, tar_z_vals_prec, tar_rays[1])
    # exchange
    ori_raw, tar_raw, _, tar_pred_label = exchanger(ori_raw, tar_raw, ori_ins_accu, tar_ins_accu, args.target_label)

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
    tar_weights = tar_weights * 1

    # resample 128 points respectively
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points
    tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)

    # cat all the points 64+128+128
    ori_z_vals, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples, tar_z_samples], dim=-1), dim=-1)
    tar_z_vals, _ = torch.sort(torch.cat([tar_z_vals, ori_z_samples, tar_z_samples], dim=-1), dim=-1)

    ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, z_vals=ori_z_vals)
    tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, z_vals=tar_z_vals)

    ori_raw, tar_raw, _, _ = exchanger(ori_raw, tar_raw, ori_ins_accu, tar_ins_accu, args.target_label)

    # final render a rgb and ins map
    final_rgb, final_weights, final_depth, final_ins = editor_render(ori_raw, ori_z_vals, ori_rays[1])

    return final_rgb, final_ins, tar_rgb, tar_ins_accu


def editor_ins(position_embedder, view_embedder, model_coarse, model_fine, ori_rays, f_tar_rays, args):
    """
        change stratagy, the integral and move color are high accuracy, so we just eliminate the color's value of points
        which has the final label and the position betweeen the camera and the plane.
    """
    # extract parameter
    N_samples, N_importance, near, far = args.N_samples, args.N_importance, args.near, args.far
    ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_coarse, N_samples, near, far)

    # ori
    _, ori_weights, _, _ = editor_render(ori_raw, ori_z_vals, ori_rays[1])

    # sample 128
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points
    ori_z_vals_prec, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples], dim=-1), dim=-1)
    ori_raw_prec, _ = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, N_samples, near, far,
                                  z_vals=ori_z_vals_prec)
    _, _, _, ori_ins_accu = editor_render(ori_raw_prec, ori_z_vals_prec, ori_rays[1])

    tar_raws, tar_rgbs, f_tar_weights, tar_instances, f_tar_z_vals, f_tar_z_samples, tar_ins_accus, = [], [], [], [], [], [], []
    for idx, tar_rays in enumerate(f_tar_rays):
        # sample 64
        tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_coarse, N_samples, near,
                                          far)
        tar_raws.append(tar_raw)
        f_tar_z_vals.append(tar_z_vals)
        # ori_raw_fine, ori_z_vals_fine = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, N_samples, near,
        #                                             far)
        # tar_raw_fine, tar_z_vals_fine = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, N_samples, near,
        #                                             far)
        # tar
        tar_rgb, tar_weights, tar_depth, tar_ins = editor_render(tar_raw, tar_z_vals, tar_rays[1])
        tar_rgbs.append(tar_rgb)
        f_tar_weights.append(tar_weights)
        tar_instances.append(tar_ins)

        # sample 128
        tar_z_vals_mid = .5 * (tar_z_vals[..., 1:] + tar_z_vals[..., :-1])
        tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)
        tar_z_vals_prec, _ = torch.sort(torch.cat([tar_z_vals, tar_z_samples], dim=-1), dim=-1)
        tar_raw_prec, _ = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine,
                                      z_vals=tar_z_vals_prec)
        _, _, _, tar_ins_accu = editor_render(tar_raw_prec, tar_z_vals_prec, tar_rays[1])
        f_tar_z_samples.append(tar_z_samples)
        tar_ins_accus.append(tar_ins_accu)

    # exchange
    ori_raw, tar_raw, _, tar_pred_label = exchanger(ori_raw, tar_raws, ori_ins_accu, tar_ins_accus, args.target_label)

    """step2"""
    # calculate weights
    ori_rgb, ori_weights, ori_depth, ori_ins = editor_render(ori_raw, ori_z_vals, ori_rays[1])

    # resample 128 points respectively
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points

    # cat all the points 64+128+(128)
    f_tar_z_samples = torch.cat(f_tar_z_samples, dim=-1)
    ori_z_vals, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples, f_tar_z_samples], dim=-1), dim=-1)
    for idx, tar_rays in enumerate(f_tar_rays):
        tar_z_vals = f_tar_z_vals[idx]
        ori_raw, ori_z_vals = editor_nerf(ori_rays, position_embedder, view_embedder, model_fine, z_vals=ori_z_vals)
        tar_z_vals, _ = torch.sort(torch.cat([tar_z_vals, ori_z_samples, f_tar_z_samples], dim=-1), dim=-1)
        tar_raw, tar_z_vals = editor_nerf(tar_rays, position_embedder, view_embedder, model_fine, z_vals=tar_z_vals)
        tar_raws[idx] = tar_raw

    ori_raw, tar_raws, _, _ = exchanger(ori_raw, tar_raws, ori_ins_accu, tar_ins_accus, args.target_label)
    # final render a rgb and ins map
    final_rgb, final_weights, final_depth, final_ins = editor_render(ori_raw, ori_z_vals, ori_rays[1])

    return final_rgb, final_ins, tar_rgb, tar_ins_accu


def editor_test(position_embedder, view_embedder, model_coarse, model_fine, ori_poses, hwk, save_dir, ins_rgbs, args,
                objs, objs_trans, view_poses, ins_map):
    """
            the first step to find the
    """

    """move_object must between 1 to args.class_number"""
    _, _, dataset_name, scene_name = args.datadir.split('/')
    H, W, K = hwk

    gt_color_dict_path = './data/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    color_dict = gt_color_dict[dataset_name][scene_name]

    save_dir = os.path.join(save_dir, "editor_output")
    os.makedirs(save_dir, exist_ok=True)

    # original
    # oper_num = len(objs_trans[objs[0]['obj_name']])
    print(view_poses.shape)
    deform_v = np.concatenate(
        (np.linspace(0, 0.18, 2), np.linspace(0.18, 0, 2), np.linspace(0, -0.18, 2), np.linspace(-0.18, 0, 2)))
    for i, ori_pose in enumerate(view_poses):
        # operate objects at same time
        time_0 = time.time()

        ori_rays_o, ori_rays_d = get_rays_k(H, W, K, torch.Tensor(ori_pose))
        ori_rays_o = torch.reshape(ori_rays_o, [-1, 3]).float()
        ori_rays_d = torch.reshape(ori_rays_d, [-1, 3]).float()

        tar_rays_os, tar_rays_ds, target_labels = [], [], []
        for obj in objs:
            obj_name = obj['obj_name']
            target_labels.append(obj['tar_id'])
            editor_mode = obj['editor_mode']
            if editor_mode == 'deform':
                print("deforming......")
                v_1 = np.linspace(1, H, H)
                editor_func = obj['deform_func']
                if editor_func == 'sin':
                    """deform sin"""
                    v_1 = ((8 * np.pi) / 400) * v_1
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = np.sin(v_1) * deform_v[i]
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif editor_func == 'ex':
                    """deform e^x"""
                    v_1 = np.exp(-1 * v_1 / 50)
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif editor_func == 'linear':
                    """"deform linear"""
                    v_1 = (v_1 - 200) / 215
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif editor_func == 'abs_linear':
                    """"deform linear"""
                    v_1 = np.abs(v_1 - 200) / 200
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif editor_func == 'ln':
                    """deform ln"""
                    v_1 = v_1 / 200
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = np.log(v_1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                tar_rays_o = ori_rays_o.clone()
                tar_rays_o[:, 0] = tar_rays_o[:, 0] + v_1
                tar_rays_d = ori_rays_d.clone()
            else:
                trans = torch.Tensor(objs_trans[obj_name][i]['transformation'])
                tar_pose = trans @ ori_pose
                tar_rays_o, tar_rays_d = get_rays_k(H, W, K, torch.Tensor(tar_pose))
            tar_rays_os.append(tar_rays_o)
            tar_rays_ds.append(tar_rays_d)

        args.target_label = target_labels
        tar_rays_os = torch.stack(tar_rays_os)
        tar_rays_ds = torch.stack(tar_rays_ds)
        tar_rays_os = torch.reshape(tar_rays_os, [len(objs), -1, 3]).float()
        tar_rays_ds = torch.reshape(tar_rays_ds, [len(objs), -1, 3]).float()
        full_rgb, full_ins, full_tar_rgb = None, None, None

        """doing editor"""
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
            # original view rays render
            ori_rays_io = ori_rays_o[step:step + N_test]  # (chuck, 3)
            ori_rays_id = ori_rays_d[step:step + N_test]  # (chuck, 3)
            ori_batch_rays = torch.stack([ori_rays_io, ori_rays_id], dim=0)
            # target view rays render
            tar_rays_ios = tar_rays_os[:, step:step + N_test]  # (chuck, 3)
            tar_rays_ids = tar_rays_ds[:, step:step + N_test]  # (chuck, 3)
            tar_batch_rays = torch.stack([tar_rays_ios, tar_rays_ids], dim=1)
            # edit render
            ori_rgb, ins, tar_rgb, tar_ins = editor_ins(position_embedder, view_embedder, model_coarse, model_fine,
                                                        ori_batch_rays,
                                                        tar_batch_rays, args)
            if full_rgb is None and full_ins is None:
                full_rgb, full_ins, full_tar_rgb, full_tar_ins = ori_rgb, ins, tar_rgb, tar_ins
            else:
                full_rgb = torch.cat((full_rgb, ori_rgb), dim=0)
                full_ins = torch.cat((full_ins, ins), dim=0)
                full_tar_rgb = torch.cat((full_tar_rgb, tar_rgb), dim=0)
                full_tar_ins = torch.cat((full_tar_ins, tar_ins), dim=0)
        ori_rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])
        ins = full_ins.reshape([H, W, full_ins.shape[-1]])
        # ori_rgb_s = ori_rgb.cpu().numpy()
        # ori_rgb_s = ori_rgb_s.reshape([H, W, 3])
        # ori_rgb_s = to8b(ori_rgb_s)
        # img_file = os.path.join(save_dir, f'{0}_rgb.png')
        # imageio.imwrite(img_file, ori_rgb_s)

        """editor finish"""
        # get predicted rgb
        ori_rgb_s = ori_rgb.cpu().numpy()
        ori_rgb_s = ori_rgb_s.reshape([H, W, 3])
        ori_rgb_s = to8b(ori_rgb_s)

        # get predicted ins color
        label = torch.argmax(ins, dim=-1)
        label = label.reshape([H, W])
        ins_img = render_label2img(label, ins_rgbs, color_dict, ins_map)

        # save images
        img_file = os.path.join(save_dir, f'{i}_rgb.png')
        imageio.imwrite(img_file, ori_rgb_s)
        ins_file = os.path.join(save_dir, f'{i}_ins.png')
        cv2.imwrite(ins_file, ins_img)

        gt_ins_file = os.path.join(save_dir, f'{i}_ins_pred_mask.png')
        imageio.imwrite(gt_ins_file, np.array(label.cpu().numpy(), dtype=np.uint8))
        time_1 = time.time()
        print(f"Image{i}: {time_1 - time_0}")
    print("finished!!!!!!!")
    return

