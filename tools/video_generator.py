import imageio
import os

# scene_names = ['dinningroom', 'study', 'living_rest']

replica_scene_names = ['office_study', 'dinningroom']
for scene_name in replica_scene_names:
    log_dir = f"./editor_logs/synthetic_rooms/0050/{scene_name}/editor_testset_200000/editor_output"

    rgb_fnames = [os.path.join(log_dir, f'{i}_rgb.png') for i in range(0, 360)]
    rgbs = [imageio.imread(f) for f in rgb_fnames]

    ins_fnames = [os.path.join(log_dir, f'{i}_ins.png') for i in range(0, 360)]
    ins = [imageio.imread(f) for f in ins_fnames]

    save_dir = os.path.join(log_dir, 'videos')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fps = 20
    imageio.mimwrite(os.path.join(save_dir, f'rgb_pred_video_{fps}.mp4'), rgbs, fps=fps, quality=8)
    imageio.mimwrite(os.path.join(save_dir, f'instance_pred_video_{fps}.mp4'), ins, fps=fps, quality=8)
