import os
import torch
from config import create_nerf, initial
from tools import pose_generator
from datasets.dmsr import dmsr_loader_val
from datasets.dmsr import dmsr_loader
from networks import manipulator
from networks import manipulator
from networks.tester import render_test
from tools.mesh_generator import mesh_main
import trimesh


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():
        if args.render:
            print("###########################################")
            print()
            print('RENDER ONLY')
            print()
            print("###########################################")
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', iteration))
            os.makedirs(testsavedir, exist_ok=True)
            mathed_file = os.path.join(testsavedir, 'matching_log.txt')
            i_train, i_test = i_split
            in_images = torch.Tensor(images[i_test])
            in_instances = torch.Tensor(instances[i_test]).type(torch.int16)
            in_poses = torch.Tensor(poses[i_test])
            render_test(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk, args,
                        gt_imgs=in_images, gt_labels=in_instances, ins_rgbs=ins_colors, savedir=testsavedir,
                        matched_file=mathed_file)
            print('Done rendering', testsavedir)

        elif args.editor_eval:
            print('EDIT EVALUATION ONLY')
            """this operations list can re-design"""

            # ori_pose, tar_poses = load_editor_poses(args)
            in_images = torch.Tensor(images)
            in_instances = torch.Tensor(instances).type(torch.int8)
            in_poses = torch.Tensor(poses)
            pose_generator.generate_poses_eval(args)
            trans_dicts = dmsr_loader_val.load_editor_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'editor_testset_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.editor_test_eval(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk,
                                         trans_dicts=trans_dicts, save_dir=testsavedir, ins_rgbs=ins_colors, args=args, gt_rgbs=in_images
                                         , gt_labels=in_instances)
            
            pass

        elif args.editor_demo:
            print("###########################################")
            print()
            print('EDIT DEMO ONLY')
            print()
            print("###########################################")

            """this operations list can re-design"""

            # ori_pose, tar_poses = load_editor_poses(args)
            print('Loaded blender', hwk, args.datadir)
            int_view_poses = torch.Tensor(view_poses)
            pose_generator.generate_poses_demo(objs, args)
            obj_trans = dmsr_loader.load_editor_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'editor_testset_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.editor_test_demo(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk,
                                    save_dir=testsavedir, ins_rgbs=ins_colors, args=args, objs=objs,
                                    objs_trans=obj_trans, view_poses=int_view_poses, ins_map=ins_map)

        elif args.mesh:
            print("###########################################")
            print()
            print("MESH ONLY")
            print()
            print("###########################################")
            mesh_file = os.path.join(args.datadir, "mesh.ply")
            assert os.path.exists(mesh_file)
            trimesh_scene = trimesh.load(mesh_file, process=False)
            meshsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mesh_testset_{:06d}'.format(iteration))
            os.makedirs(meshsavedir, exist_ok=True)
            mesh_main(position_embedder, view_embedder, model_coarse, model_fine, args, trimesh_scene, ins_colors,
                      meshsavedir, ins_map)
    return


if __name__ == '__main__':

    args = initial()

    # load data
    if args.editor_val:
        images, poses, hwk, instances, ins_colors, args.ins_num = dmsr_loader_val.load_data(args)
    else:
        images, poses, hwk, i_split, instances, ins_colors, args.ins_num, objs, view_poses, ins_map = dmsr_loader.load_data(
            args)
    print('Loaded blender', images.shape, hwk, args.datadir)

    # load transformation matrix
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    args.perturb = False
    H, W, K = hwk

    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    iteration = 0
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, args.log_time, "200000.tar")]
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
