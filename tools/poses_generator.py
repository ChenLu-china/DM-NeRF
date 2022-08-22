import os
import numpy as np
from networks.helpers import r_x, r_y, r_z
# from data.hypersims.processing.ins_centers import ins_center_loader
import json

# ins_centers = {'bathroom': [0.779178, 1.05247, 0.380208], 'bedroom': [-1.29552, 1.72703, 0.2946],
#                'dinningroom': [-0.633653, 0.295162, 0.279743], 'kitchen': [-2.52579, -0.103821, 1.47165],
#                'livingroom_reception': [0.579352, -0.099242, 0.092597],
#                'livingroom_rest': [-0.001277, -2.85079, 0.588084],
#                'office_reception': [-0.717374, 0.929292, 0.904515],
#                'office_study': [-0.519422, -2.16509, 1.07392]}


def generate_poses(objs, args, defined_transformations=None):
    """pose also can generated by yourself or selected from the data set"""
    # import you self designed transformation matrix
    save_path = os.path.join(args.datadir, 'transformation_matrix.json')
    out_puts = {}
    views = args.views
    for obj in objs:
        obj_name = obj["obj_name"]
        ins_center = obj["obj_center"]
        editor_mode = obj["editor_mode"]

        obj_transformations = {}
        poses_list = []
        # ins_centers = ins_center_loader(args)
        ins_center = np.array(ins_center)
        # ins_center = ins_centers[args.center_index]
        translation = np.eye(4, 4, dtype=np.float32)
        translation[:3, -1] = -1 * ins_center
        translation_inverse = np.eye(4, 4, dtype=np.float32)
        translation_inverse[:3, -1] = -1 * translation[:3, -1]
        if editor_mode == 'translation':
            oper_dists = obj['distance']
            num = len(oper_dists)
            for oper_dist in oper_dists:
                move_step = 2 * num
                oper_dist_step = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, oper_dist / (views // move_step)],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
                t = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
                for i in range(0, views // move_step):
                    t = t @ oper_dist_step
                    tar_pose = translation_inverse @ t @ translation
                    pose_dict = dict({'transformation': tar_pose.tolist(), 'mode': 'translation'})
                    poses_list.append(pose_dict)

                t = np.array([[1, 0, 0, 0],
                              [0, 1, 0, oper_dist],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

                oper_dist_step[1, -1] = oper_dist_step[1, -1] * -1
                for i in range(0, views // move_step):
                    t = t @ oper_dist_step
                    tar_pose = translation_inverse @ t @ translation
                    pose_dict = dict({'transformation': tar_pose.tolist(), 'mode': 'translation'})
                    poses_list.append(pose_dict)
        if editor_mode == 'rotation':
            oper_degree = obj["rotation"]
            'self rotation'
            roll = np.zeros(1)
            pitch = np.zeros(1)
            rotation_step = oper_degree / views
            for i in range(0, views):
                oper_degree = oper_degree + rotation_step
                yaw = np.array([oper_degree * np.pi / 180])
                r = r_z(yaw[0]) @ r_y(pitch[0]) @ r_x(roll[0])
                tar_pose = translation_inverse @ r @ translation
                pose_dict = dict({'transformation': tar_pose.tolist(), 'mode': 'rotation'})
                poses_list.append(pose_dict)

        if editor_mode == 'scale':
            scales = np.array([1.2])
            for i in range(len(scales)):
                s = np.array([[scales[i], 0, 0, 0],
                              [0, scales[i], 0, 0],
                              [0, 0, scales[i], 0],
                              [0, 0, 0, 1]])
                tar_pose = translation_inverse @ s @ translation
                pose_dict = dict({'transformation': tar_pose.tolist(), 'mode': 'scale'})
                poses_list.append(pose_dict)

        if editor_mode == 'multi':
            scales = np.array([1.2])
            s = np.array([[scales[0], 0, 0, 0],
                          [0, scales[0], 0, 0],
                          [0, 0, scales[0], 0],
                          [0, 0, 0, 1]])

            roll = np.zeros(1)
            pitch = np.zeros(1)
            yaw = np.array([90 * np.pi / 180])
            r = r_z(yaw[0]) @ r_y(pitch[0]) @ r_x(roll[0])
            multi_transforamtion = s @ r

            distance = - 0.25
            t = np.array([[1, 0, 0, 0],
                          [0, 1, 0, distance],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            multi_transforamtion = multi_transforamtion @ t
            tar_pose = translation_inverse @ multi_transforamtion @ translation
            pose_dict = dict({'transformation': tar_pose.tolist(), 'mode': 'multi'})
            poses_list.append(pose_dict)
        obj_transformations.update({obj_name: poses_list})
        out_puts.update(obj_transformations)
    with open(save_path, 'w') as fp:
        json.dump(out_puts, fp, ensure_ascii=False, indent=2)
    fp.close()

    return
