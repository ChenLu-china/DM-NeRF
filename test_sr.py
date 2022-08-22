import os
from config import create_nerf, initial
from datasets.sr.data_loader import load_editor_poses
from datasets.sr.load_sr import *
from tools.editor_tools import editor_test
from tools.test_tools import render_test
from data.sr.poses_generator import generate_poses


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():
        if args.render:
            print('RENDER ONLY')
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', iteration))
            os.makedirs(testsavedir, exist_ok=True)
            mathed_file = os.path.join(testsavedir, 'matching_log.txt')
            render_test(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk, args,
                        gt_imgs=images, gt_labels=instances, ins_rgbs=ins_colors, savedir=testsavedir,
                        matched_file=mathed_file)
            print('Done rendering', testsavedir)
        elif args.editor:
            print('EDIT RENDER ONLY')
            """this operations list can re-design"""

            # ori_pose, tar_poses = load_editor_poses(args)
            obj_trans = load_editor_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'editor_testset_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            editor_test(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk,
                        save_dir=testsavedir, ins_rgbs=ins_colors, args=args, objs=objs,
                        objs_trans=obj_trans, view_poses=view_poses, ins_map=ins_map)
    return


if __name__ == '__main__':

    args = initial()

    # load data
    if args.render:
        images, poses, hwk, i_split, instances, ins_colors, args.ins_num = load_data(args)
        print('Loaded blender', images.shape, hwk, args.datadir)
        i_train, i_test = i_split
        images = torch.Tensor(images[i_test])
        instances = torch.Tensor(instances[i_test]).type(torch.int16)
        poses = torch.Tensor(poses[i_test])

    elif args.editor:
        args.views = 720
        objs, view_poses, ins_map, poses, hwk, ins_colors, args.ins_num = load_data(args)
        print('Loaded blender', hwk, args.datadir)

        view_poses = torch.Tensor(view_poses)

        generate_poses(objs, args)

    # load transformation matrix
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    args.perturb = False
    H, W, K = hwk

    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    iteration = 0
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, args.log_time, "300000.tar")]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        if torch.cuda.is_available():
            ckpt = torch.load(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        iteration = ckpt['iteration']
        # Load model
        model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    test()

