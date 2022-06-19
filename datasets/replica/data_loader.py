import os
import h5py
import numpy as np
import json
import imageio

np.random.seed(0)


def png_i(f):
    return imageio.imread(f)

class processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(rgb_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_rgb(self):
        # testskip operation

        objs_info_fname = os.path.join(self.basedir, 'objs_info.json')
        with open(objs_info_fname, 'r') as f_obj_info:
            objs_info = json.load(f_obj_info)
        f_obj_info.close()
        objs = objs_info["objects"]
        view_id = objs_info["view_id"]
        ins_map = objs_info["ins_map"]
        
        if view_id is not None:
            _, _, dataset_name, scene_name = self.basedir.split('/')
            skip_idx = np.arange(0, len(self.test_ids), self.testskip)
            selected_test_ids = np.array(self.test_ids)[skip_idx]
            gt_id = self.train_ids[view_id]
            print(gt_id)
            # gt_id = selected_test_ids[view_id[scene_name]]
            
            # load poses
            traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
            Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
            poses = Ts_full[self.train_ids]
            ori_pose = poses[view_id[scene_name]]
            
            # load rgbs
            rgb_basedir = os.path.join(self.basedir, f'rgb_{gt_id}.png')
            gt_img = imageio.imread(rgb_basedir)

            # load ins gt
            ins_basedir = os.path.join(self.basedir, f'semantic_instance_{gt_id}.png')
            gt_label = imageio.imread(ins_basedir)
            gt_unique_labels = np.unique(gt_label)
            
            # for label in gt_unique_labels:
            #     vis_mask = gt_label == label
            #     vis_img(vis_mask)

        else:
            ori_pose = None
            gt_img = None
            gt_label = None

        color_f = os.path.join(self.basedir, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        ins_num = len(ins_rgbs)
        return gt_img, gt_label, ori_pose, ins_rgbs, ins_num

class rgb_processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(rgb_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_rgb(self):
        # testskip operation
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)

        # load poses
        traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
        Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        train_poses = Ts_full[self.train_ids]
        test_poses = Ts_full[self.test_ids]
        test_poses = test_poses[skip_idx]
        poses = np.concatenate([train_poses, test_poses], axis=0)

        # load rgbs
        rgb_basedir = os.path.join(self.basedir, 'rgb')
        train_rgbs = [png_i(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.train_ids]
        test_rgbs = [png_i(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.test_ids]
        train_rgbs = np.array(train_rgbs)
        test_rgbs = np.array(test_rgbs)[skip_idx]
        rgbs = np.concatenate([train_rgbs, test_rgbs], axis=0)
        rgbs = (rgbs / 255.).astype(np.float32)

        i_train = np.arange(0, len(self.train_ids), 1)
        i_test = np.arange(len(self.train_ids), len(self.train_ids) + len(skip_idx), 1)
        i_split = [i_train, i_test]

        return rgbs, poses, i_split


class ins_processor:
    def __init__(self, base_path, train_ids, test_ids, weakly_mode, weakly_value, testskip=1):
        super(ins_processor, self).__init__()
        self.weakly_mode = weakly_mode
        self.weakly_value = weakly_value
        self.base_path = base_path
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.testskip = testskip

        # load operation
        self.gt_labels, self.ins_rgbs = self.load_semantic_instance()
        self.ins_num = len(self.ins_rgbs)
        # self.unique_labels = np.unique(self.gt_labels)

    def load_semantic_instance(self):
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)
        ins_base_path = os.path.join(self.base_path, 'semantic_instance')
        train_sem_ins = [png_i(os.path.join(ins_base_path, f'semantic_instance_{idx}.png')) for idx in self.train_ids]
        train_sem_ins = np.array(train_sem_ins).astype(np.float32)
        test_sem_ins = [png_i(os.path.join(ins_base_path, f'semantic_instance_{idx}.png')) for idx in self.test_ids]
        test_sem_ins = np.array(test_sem_ins)[skip_idx].astype(np.float32)

        gt_labels = np.concatenate([train_sem_ins, test_sem_ins], 0)
        color_f = os.path.join(self.base_path, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        return gt_labels, ins_rgbs

    # def load_semantic_instance(self):
    #     splits = ['train', 'test']
    #     all_ins = []
    #     for s in splits:
    #         if s == 'train' or self.testskip == 0:
    #             skip = 1
    #         else:
    #             skip = self.testskip
    #         ins_path = os.path.join(self.base_path, f'{s}_ins')
    #         ins_files = [os.path.join(ins_path, f'semantic_instance_{i}.png') for i in range(len(os.listdir(ins_path)))]
    #         gt_labels = np.array([png_i(f) for f in ins_files]).astype(np.float32)
    #
    #         index = np.arange(0, len(gt_labels), skip)
    #         gt_labels = gt_labels[index]
    #         all_ins.append(gt_labels)
    #
    #     gt_labels = np.concatenate(all_ins, 0)
    #     f = os.path.join(self.base_path, 'ins_rgb.hdf5')
    #     with h5py.File(f, 'r') as f:
    #         ins_rgbs = f['dataset'][:]
    #     f.close()
    #     return gt_labels, ins_rgbs

def load_editor_poses(args):
    load_path = os.path.join(args.datadir, 'transformation_matrix.json')
    with open(load_path, 'r') as rf:
        obj_trans = json.load(rf)
    rf.close()
    return obj_trans
