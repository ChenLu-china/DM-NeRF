import os
import h5py
import numpy as np
import json
import imageio

np.random.seed(0)


def png_i(f):
    return imageio.imread(f)


# Editor testing data loader
class processor:
    def __init__(self, basedir, testskip=1):
        super(processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_gts(self):
        poses = []

        posefile = os.path.join(self.basedir, f'train_transforms.json')
        with open(posefile, 'r') as read_pose:
            meta = json.load(read_pose)

        angle_x = meta['camera_angle_x']
        for frame in meta['frames'][::self.testskip]:
            poses.append(frame['transform_matrix'])
        poses = np.array(poses).astype(np.float32)

        f = os.path.join(self.basedir, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()

        objs_info_fname = os.path.join(self.basedir, 'objs_info.json')
        with open(objs_info_fname, 'r') as f_obj_info:
            objs_info = json.load(f_obj_info)
        f_obj_info.close()
        objs = objs_info["objects"]
        view_id = objs_info["view_id"]
        ins_map = objs_info["ins_map"]
        return objs, view_id, ins_map, poses, ins_rgbs, angle_x


# Training and Testing data loader class
class rgb_processor:
    def __init__(self, basedir, testskip=1):
        super(rgb_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_rgb(self):
        splits = ['train', 'test']
        all_rgb = []
        all_pose = []
        counts = [0]
        angle_x = None
        for s in splits:
            poses = []
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            fname = os.path.join(self.basedir, s, 'rgbs')

            imagefile = [os.path.join(fname, f) for f in sorted(os.listdir(fname))]

            rgbs = [png_i(f) for f in imagefile]

            posefile = os.path.join(self.basedir, s, f'transforms.json')
            with open(posefile, 'r') as read_pose:
                meta = json.load(read_pose)

            angle_x = meta['camera_angle_x']
            for frame in meta['frames'][::skip]:
                poses.append(frame['transform_matrix'])
            poses = np.array(poses).astype(np.float32)

            index = np.arange(0, len(rgbs), skip)
            rgbs = np.array(rgbs)[index]
            rgbs = (rgbs / 255.).astype(np.float32)  # keep all 3 channels (RGB)
            counts.append(counts[-1] + rgbs.shape[0])
            all_rgb.append(rgbs)
            all_pose.append(poses)

        all_rgb = np.concatenate(all_rgb, 0)
        all_pose = np.concatenate(all_pose, 0)

        if all_pose.shape[-1] == 16:
            all_pose = all_pose.reshape((all_pose.shape[0], 4, 4))

        # all_pose_ = np.linalg.inv(all_pose)
        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(2)]

        return all_rgb, all_pose, i_split, angle_x


class ins_processor:
    def __init__(self, base_path, weakly_mode, weakly_value, testskip=1):
        super(ins_processor, self).__init__()
        self.weakly_mode = weakly_mode
        self.weakly_value = weakly_value
        self.base_path = base_path
        self.testskip = testskip

        # load operation
        self.gt_labels, self.ins_rgbs = self.load_semantic_instance()
        self.ins_num = len(self.ins_rgbs)
        self.unique_labels = np.unique(self.gt_labels)
        print()

    def load_semantic_instance(self):
        splits = ['train', 'test']
        all_ins = []
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            ins_path = os.path.join(self.base_path, s, 'semantic_instance')
            ins_files = [os.path.join(ins_path, f) for f in sorted(os.listdir(ins_path))]
            gt_labels = np.array([png_i(f) for f in ins_files])

            index = np.arange(0, len(gt_labels), skip)
            gt_labels = gt_labels[index]
            all_ins.append(gt_labels)

        gt_labels = np.concatenate(all_ins, 0)
        f = os.path.join(self.base_path, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        return gt_labels, ins_rgbs

    def selected_pixels(self, full_ins):
        """
        Explanation:
        This function aims to process the instance image, the main idea is using fixed 0.1(this value can private change)
        pixels, this part comes from each object, if the pixels of one object less than average threshold, we select all
        the pixels, others are selected follow weight ratio. All in all, we wish every image contain fixed number of
        semantic instance pixel easy for us to train and calculate cost matrix for one iteration.
        """

        def weakly_ins():
            """select ins label regard object as unit"""
            ins_hws = None
            amounts = label_amounts.astype(np.int32)
            for index, label in enumerate(unique_labels):
                if label != self.ins_num:
                    ins_indices = np.where(ins == label)[0]
                    select_indices = np.random.choice(label_amounts[index], size=[amounts[index]], replace=False)
                    ins_hw = ins_indices[select_indices]
                    if ins_hws is None:
                        ins_hws = ins_hw
                    else:
                        ins_hws = np.concatenate((ins_hws, ins_hw), 0)
            return ins_hws

        # begin weakly
        N, H, W = full_ins.shape
        full_ins = np.reshape(full_ins, [N, -1])  # (N, H*W)
        all_ins_hws = []

        for i in range(N):
            ins = full_ins[i]
            unique_labels, label_amounts = np.unique(ins, return_counts=True)
            # need a parameter
            hws = weakly_ins()
            all_ins_hws.append(hws)

        return all_ins_hws


def load_editor_poses(args):
    load_path = os.path.join(args.datadir, 'transformation_matrix.json')
    with open(load_path, 'r') as rf:
        obj_trans = json.load(rf)
    rf.close()
    return obj_trans