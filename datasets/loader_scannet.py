import numpy as np
from datasets.scannet import data_loader
from datasets.scannet.loader_vis import vis_segmentation, ins2img, ins2img_full, vis_crop


def crop_data(H, W, crop_size):
    # center_w, center_h = H // 2, W // 2
    crop_mask = np.zeros(shape=(H, W))
    new_w, new_h = crop_size
    margin_h = (H - new_h) // 2
    margin_w = (W - new_w) // 2
    crop_mask[margin_h: (H - margin_h), margin_w: (W - margin_w)] = 1
    return crop_mask.astype(np.int8)


def load_data(args):
    imgs, poses, i_split, intrinsic = data_loader.img_processor(args.datadir,
                                                                args.testskip,
                                                                resize=args.resize).load_rgb()

    ins_processor = data_loader.ins_processor(args.datadir,
                                              testskip=args.testskip,
                                              resize=args.resize,
                                              weakly_value=args.weakly_value)
    gt_labels, ins_rgbs, ins_num =ins_processor.load_semantic_instance()
    crop_size = [args.crop_width, args.crop_height]

    H, W = imgs[0].shape[:2]
    hwk = [int(H), int(W), intrinsic]
    crop_mask = crop_data(H, W, crop_size)
    ins_indices = ins_processor.selected_pixels(gt_labels, ins_num, crop_mask)
    # gt_img = ins2img(gt_labels[0], ins_rgbs)
    # gt_img = ins2img_full(gt_labels[0], ins_rgbs)
    # vis_segmentation(imgs[0], gt_img)
    # img = imgs[0]
    # img[crop_mask == 0] = [1, 1, 1]
    # vis_crop(img)

    return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_num, ins_indices, crop_mask
