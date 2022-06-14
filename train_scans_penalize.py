import torch
import os
import numpy as np
from config import initial, create_nerf
from datasets.loader_scannet import load_data
from networks.ins_eval import ins_criterion, img2mse, mse2psnr
from networks.ins_penalizer import ins_penalizer
from tools.test_tools import render_test
from tools.helpers_tools import gt_select_crop, z_val_sample
from tools.render_tools import ins_nerf

np.random.seed(0)
torch.cuda.manual_seed(4)


def train():
    model_fine.train()
    model_coarse.train()
    N_iters = 500000 + 1

    z_val_coarse = z_val_sample(args.N_train, args.near, args.far, args.N_samples)
    for i in range(1, N_iters):
        img_i = np.random.choice(i_train)
        gt_rgb = images[img_i].to(args.device)
        pose = poses[img_i, :3, :4].to(args.device)
        gt_label = gt_labels[img_i].to(args.device)
        ins_index = ins_indices[img_i]

        target_c, target_i, batch_rays, args.N_ins = gt_select_crop(gt_rgb, pose, K, gt_label, ins_index, crop_mask,
                                                                    args.N_train)

        all_info = ins_nerf(batch_rays, position_embedder, view_embedder, model_coarse, model_fine, z_val_coarse, args)

        # coarse losses
        rgb_loss_coarse = img2mse(all_info['rgb_coarse'], target_c)
        psnr_coarse = mse2psnr(rgb_loss_coarse)

        ins_loss_coarse, valid_ce_coarse, invalid_ce_coarse, valid_siou_coarse = \
            ins_criterion(all_info['ins_coarse'], target_i, args.ins_num)
        penalize_coarse = ins_penalizer(all_info['raw_coarse'], all_info['z_vals_coarse'],
                                        all_info['depth_coarse'], batch_rays[1], args)

        # fine losses
        rgb_loss_fine = img2mse(all_info['rgb_fine'], target_c)
        psnr_fine = mse2psnr(rgb_loss_fine)
        ins_loss_fine, valid_ce_fine, invalid_ce_fine, valid_siou_fine = \
            ins_criterion(all_info['ins_fine'], target_i, args.ins_num)
        penalize_fine = ins_penalizer(all_info['raw_fine'], all_info['z_vals_fine'],
                                      all_info['depth_fine'], batch_rays[1], args)
        # trans = extras['raw'][..., -1]

        # total losses
        ins_loss = ins_loss_fine + ins_loss_coarse
        rgb_loss = rgb_loss_fine + rgb_loss_coarse
        penalize_loss = penalize_fine + penalize_coarse
        total_loss = rgb_loss + ins_loss + penalize_loss
        # optimizing
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # losses decay
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** ((i) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ###################################

        if i % args.i_print == 0:
            print(
                f"[TRAIN] Iter: {i} F_PSNR: {psnr_fine.item()} C_PSNR: {psnr_coarse.item()} Total_Loss: {total_loss.item()} \n"
                f"RGB_Loss: {rgb_loss.item()} Ins_Loss: {ins_loss.item()} Ins_SIoU_Loss: {valid_siou_fine.item()} \n"
                f"Ins_CE_Loss: {valid_ce_fine.item()} Ins_in_CE_Loss: {invalid_ce_fine.item()}  Reg_Loss: {penalize_loss.item()}")
        if i % args.i_save == 0:
            path = os.path.join(args.basedir, args.expname, args.log_time, '{:06d}.tar'.format(i))
            save_model = {
                'iteration': i,
                'network_coarse_state_dict': model_coarse.state_dict(),
                'network_fine_state_dict': model_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_model, path)

        if i % args.i_test == 0:
            model_coarse.eval()
            model_fine.eval()
            args.is_train = False
            selected_indices = np.random.choice(len(i_test), size=[10], replace=False)
            selected_i_test = i_test[selected_indices]
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'testset_{:06d}'.format(i))
            matched_file = os.path.join(testsavedir, 'matching_log.txt')
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[selected_i_test].shape)
            with torch.no_grad():
                test_poses = torch.Tensor(poses[selected_i_test].to(args.device))
                test_imgs = images[selected_i_test]
                test_gt_labels = gt_labels[selected_i_test].to(args.device)
                render_test(position_embedder, view_embedder, model_coarse, model_fine, test_poses, hwk, args,
                            gt_imgs=test_imgs, gt_labels=test_gt_labels, ins_rgbs=ins_rgbs, savedir=testsavedir,
                            matched_file=matched_file, crop_mask=crop_mask)
            print('Saved test set')
            args.is_train = True
            model_coarse.train()
            model_fine.train()


if __name__ == '__main__':
    args = initial()

    # load data
    images, poses, hwk, i_split, gt_labels, ins_rgbs, args.ins_num, ins_indices, crop_mask = load_data(args)
    print('Loaded blender', images.shape, hwk, args.datadir)

    i_train, i_test = i_split
    H, W, K = hwk

    # Create nerf model
    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    # Create optimizer
    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # move data to gpu
    images = torch.Tensor(images).cpu()
    gt_labels = torch.Tensor(gt_labels).type(torch.int16).cpu()
    poses = torch.Tensor(poses).cpu()

    train()
