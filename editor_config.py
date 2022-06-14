import os
import time
import configargparse
import torch
from networks.ins_nerf import get_embedder, Instance_NeRF


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, type=str,
                        default='configs/replica/0050/editor_rotation/room_0.txt',
                        help='configs file path')
    parser.add_argument("--expname", type=str, default='bathroom',
                        help='experiment name')
    parser.add_argument("--log_time", default=None,
                        help="save as time level")
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/hypersims/bathroom',
                        help='input data directory')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    # 32*32*4
    parser.add_argument("--N_train", type=int, default=4096,
                        help='batch size (number of random rays per gradient step)')

    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 250
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--N_test", type=int, default=1024 * 2,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--is_train", type=bool, default=True,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # semantic instance network options

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # 0
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--render", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # datasets options
    parser.add_argument("--testskip", type=int, default=10,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--resize", action='store_true',
                        help='will resize image and instance map shape of ScanNet dataset')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--near", type=float,
                        help='set the nearest depth')
    parser.add_argument("--far", type=float,
                        help='set the farest depth')

    parser.add_argument("--crop_width", type=int,
                        help='set the width of crop')
    parser.add_argument("--crop_height", type=int,
                        help='set the height of crop')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_save", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # semantic instance options
    parser.add_argument("--weakly_mode", type=str, default='weakly_ins',
                        help="select how to weakly instance labels can be set as weakly_ins, weakly_img, weakly_click")
    parser.add_argument("--weakly_value", type=float, default=1.0,
                        help="select how to weakly instance labels, 0-1")
    parser.add_argument("--over_penalize", action='store_true',
                        help="aim to penalize unlabeled rays to air")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="")
    parser.add_argument("--deta_w", type=float, default=None,
                        help="")

    # visualizer hyper-parameter
    parser.add_argument("--target_label", type=int, default=None,
                        help='sign the instance you want to move')

    parser.add_argument("--center_index", type=int, default=None,
                        help='sign the instance center')

    parser.add_argument("--ori_pose", type=int, default=None,
                        help='sign the instance center')
    # editor parameter
    parser.add_argument("--editor", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--editor_mode", type=str, default='rotation',
                        help='select operation mode includes translation, rotation, scale, multi')
    return parser


def create_nerf(args):
    """
        Instantiate NeRF's MLP model.
    """
    position_embedder, input_ch_pos = get_embedder(args.multires, args.i_embed)
    view_embedder, input_ch_view = get_embedder(args.multires_views, args.i_embed)
    model_coarse = \
        Instance_NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4], args.ins_num).to(args.device)
    print(model_coarse)

    model_fine = \
        Instance_NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4], args.ins_num).to(args.device)
    print(model_fine)
    return position_embedder, view_embedder, model_coarse, model_fine, args


def initial():
    parser = config_parser()
    args = parser.parse_args()

    # get log time
    if args.log_time is None:
        args.log_time = time.strftime("%Y%m%d%H%M", time.localtime())

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    return args
