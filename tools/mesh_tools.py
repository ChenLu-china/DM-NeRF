import os
import json
import torch
import open3d as o3d
import skimage.measure as ski_measure
import numpy as np
import trimesh
import torch.nn.functional as F
from tools.vis_tools import *
from networks.render import ins_nerf
from networks.helpers import z_val_sample


def operate_hemisphere(N=256, radius=7.5):
    x = np.linspace(-radius, radius, N)
    y = np.linspace(-radius, radius, N)
    z = np.linspace(-radius, radius, N)

    query_pts = np.stack(np.meshgrid(x, y, z), -1).astype(np.float32)
    x0, y0, z0 = (0, 0, 0)
    r = np.sqrt((query_pts[..., 0] - x0) ** 2 + (query_pts[..., 1] - y0) ** 2 + (query_pts[..., 2] - z0) ** 2)
    query_pts = query_pts[r < radius]
    hemisphere_pts = query_pts[query_pts[:, -1] >= 0]
    return hemisphere_pts

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=7.5, resolution=20)
    # # pcd = sphere.sample_points_uniformly(number_of_points=10000)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(sphere, voxel_size=0.1)
    # point_cloud_np = np.asarray(
    #     [voxel_grid.origin + pt.grid_index * voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
    # point_cloud_np = point_cloud_np[point_cloud_np[:, -1] > 0]
    point_cloud_color = np.zeros_like(hemisphere_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hemisphere_pts)
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_color)
    # print(point_cloud_np)
    o3d.visualization.draw_geometries([pcd])


def operate_cube(x, y, z, N=256):
    t_x = np.linspace(x[0], x[1], N)
    t_y = np.linspace(y[0], y[1], N)
    t_z = np.linspace(z[0], z[1], N)

    query_pts = np.stack(np.meshgrid(t_x, t_y, t_z), -1).astype(np.float32)
    return query_pts


def mesh_main(position_embedder, view_embedder, model_coarse, model_fine, args, trimesh_scene, ins_rgbs, save_dir,
              ins_map=None):
    _, _, dataset_name, scene_name = args.datadir.split('/')

    if dataset_name == 'dmsr':

        gt_color_dict_path = './data/color_dict.json'
        gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
        color_dict = gt_color_dict[dataset_name][scene_name]
        print(color_dict)
        print(ins_map)
        print(ins_rgbs)
        level = 0.45  # level = 0
        threshold = 0.2
        grid_dim = 256

        to_origin_transform, extents = trimesh.bounds.oriented_bounds(trimesh_scene)
        # print(to_origin_transform, extents)
        T_extent_to_scene = np.linalg.inv(to_origin_transform)
        scene_transform = T_extent_to_scene
        scene_extents = np.array([1.9, 7.0, 7.])
        grid_query_pts, scene_scale = grid_within_bound([-1.0, 1.0], scene_extents, scene_transform, grid_dim=grid_dim)
        grid_query_pts = grid_query_pts[:, :, [0, 2, 1]]
        grid_query_pts[:, :, 1] = grid_query_pts[:, :, 1] * -1
        query = grid_query_pts.numpy()[:, 0, :]
        print(extents)
        print(np.max(query, axis=0), np.min(query, axis=0))
        grid_query_pts = grid_query_pts.cuda().reshape(-1, 3)  # Num_rays, 1, 3-xyz

        N = grid_query_pts.shape[0]
        raw = None

        for step in range(0, N, args.N_test):
            N_test = args.N_test
            if step + N_test > N:
                N_test = N - step
            in_pcd = grid_query_pts[step:step + N_test]
            embedded = position_embedder.embed(in_pcd)
            viewdirs = torch.zeros_like(in_pcd)
            embedded_dirs = view_embedder.embed(viewdirs)
            embedded = torch.cat([embedded, embedded_dirs], -1)
            raw_fine = model_fine(embedded)
            if raw is None:
                raw = raw_fine
            else:
                raw = torch.cat((raw, raw_fine), dim=0)

        alpha = raw[..., 3]
        raw = raw.cpu().numpy()

        def occupancy_activation(alpha, distances):
            occ = 1.0 - torch.exp(-F.relu(alpha) * distances)
            # notice we apply RELU to raw sigma before computing alpha
            return occ

        voxel_size = (args.far - args.near) / args.N_importance  # or self.N_importance
        occ = occupancy_activation(alpha, voxel_size)
        print("Compute Occupancy Grids")
        occ = occ.reshape(grid_dim, grid_dim, grid_dim)
        occupancy_grid = occ.detach().cpu().numpy()

        print('fraction occupied:', (occupancy_grid > threshold).mean())
        print('Max Occ: {}, Min Occ: {}, Mean Occ: {}'.format(occupancy_grid.max(), occupancy_grid.min(),
                                                              occupancy_grid.mean()))
        vertices, faces, vertex_normals, _ = ski_measure.marching_cubes(occupancy_grid, level=level,
                                                                        gradient_direction='ascent')

        dim = occupancy_grid.shape[0]
        vertices = vertices / (dim - 1)
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)

        # Transform to [-1, 1] range
        mesh_canonical = mesh.copy()

        vertices_ = np.array(mesh_canonical.vertices).reshape([-1, 3]).astype(np.float32)
        print(np.max(vertices_, axis=0), np.min(vertices_, axis=0))

        mesh_canonical.apply_translation([-0.5, -0.5, -0.5])
        mesh_canonical.apply_scale(2)

        scene_scale = scene_extents / 2.0
        # Transform to scene coordinates
        mesh_canonical.apply_scale(scene_scale)
        mesh_canonical.apply_transform(scene_transform)
        # mesh.show()

        exported = trimesh.exchange.export.export_mesh(mesh_canonical,
                                                       os.path.join(save_dir, 'mesh_canonical.ply'))
        print("Saving Marching Cubes mesh to mesh_canonical.ply !")
        exported = trimesh.exchange.export.export_mesh(mesh_canonical, os.path.join(save_dir, 'mesh.ply'))
        print("Saving Marching Cubes mesh to mesh.ply !")

        o3d_mesh = trimesh_to_open3d(mesh)
        o3d_mesh_canonical = trimesh_to_open3d(mesh_canonical)

        print('Removing noise ...')
        print(
            f'Original Mesh has {len(o3d_mesh_canonical.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical.triangles) / 1e6:.2f} M faces.')
        o3d_mesh_canonical_clean = clean_mesh(o3d_mesh_canonical, keep_single_cluster=False,
                                              min_num_cluster=400)

        vertices_ = np.array(o3d_mesh_canonical_clean.vertices).reshape([-1, 3]).astype(np.float32)
        triangles = np.asarray(o3d_mesh_canonical_clean.triangles)  # (n, 3) int
        N_vertices = vertices_.shape[0]
        print(
            f'Denoised Mesh has {len(o3d_mesh_canonical_clean.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical_clean.triangles) / 1e6:.2f} M faces.')

        print("###########################################")
        print()
        print("Using Normals for colour predictions!")
        print()
        print("###########################################")

        ## use normal vector method as suggested by the author, see https://github.com/bmild/nerf/issues/44
        mesh_recon_save_dir = os.path.join(save_dir, "use_vertex_normal")
        os.makedirs(mesh_recon_save_dir, exist_ok=True)

        selected_mesh = o3d_mesh_canonical_clean
        rays_d = - torch.FloatTensor(
            np.asarray(selected_mesh.vertex_normals))  # use negative normal directions as ray marching directions
        rays_d = rays_d[:, [0, 2, 1]]
        rays_d[:, 1] = rays_d[:, 1] * -1

        vertices_ = vertices_[:, [0, 2, 1]]
        vertices_[:, 1] = vertices_[:, 1] * -1
        rays_o = torch.FloatTensor(vertices_) - rays_d * 0.03 * args.near

        print(np.max(vertices_, axis=0), np.min(vertices_, axis=0))

        full_ins = None
        N = rays_o.shape[0]
        print(N)
        print(rays_o.shape)
        z_val_coarse = z_val_sample(args.N_test, 0.01, 15, args.N_samples)
        with torch.no_grad():
            for step in range(0, N, args.N_test):
                N_test = args.N_test
                if step + N_test > N:
                    N_test = N - step
                    z_val_coarse = z_val_sample(N_test, 0.01, 15, args.N_samples)
                rays_io = rays_o[step:step + N_test]  # (chuck, 3)
                rays_id = rays_d[step:step + N_test]  # (chuck, 3)
                batch_rays = torch.stack([rays_io, rays_id], dim=0)
                batch_rays = batch_rays.to(args.device)
                all_info = ins_nerf(batch_rays, position_embedder, view_embedder,
                                    model_coarse, model_fine, z_val_coarse, args)
                if full_ins is None:
                    full_ins = all_info['ins_fine']
                else:
                    full_ins = torch.cat((full_ins, all_info['ins_fine']), dim=0)
        print(full_ins.shape)
        pred_label = torch.argmax(full_ins, dim=-1)
        ins_color = render_label2world(pred_label, ins_rgbs, color_dict, ins_map)

        o3d_mesh_canonical_clean.vertex_colors = o3d.utility.Vector3dVector(ins_color[:, [2, 1, 0]] / 255.0)
        o3d.io.write_triangle_mesh(
            os.path.join(mesh_recon_save_dir, 'ins_mesh_canonical_dim{}neart_{}.ply'.format(grid_dim, args.near)),
            o3d_mesh_canonical_clean)
        print("Saving Marching Cubes mesh to instance_mesh_canonical_dim{}neart_{}.ply".format(grid_dim, args.near))

        print("--------end----------")

    elif dataset_name == 'replica':

        level = 0.05  # level = 0
        threshold = 0.2
        grid_dim = 256

        to_origin_transform, extents = trimesh.bounds.oriented_bounds(trimesh_scene)
        T_extent_to_scene = np.linalg.inv(to_origin_transform)
        scene_transform = T_extent_to_scene
        scene_extents = extents
        grid_query_pts, scene_scale = grid_within_bound([-1.0, 1.0], scene_extents, scene_transform, grid_dim=grid_dim)
        grid_query_pts = grid_query_pts.cuda().reshape(-1, 3)  # Num_rays, 1, 3-xyz
        print(grid_query_pts.shape)
        N = grid_query_pts.shape[0]
        raw = None

        for step in range(0, N, args.N_test):
            N_test = args.N_test
            if step + N_test > N:
                N_test = N - step
            in_pcd = grid_query_pts[step:step + N_test]
            embedded = position_embedder.embed(in_pcd)
            viewdirs = torch.zeros_like(in_pcd)
            embedded_dirs = view_embedder.embed(viewdirs)
            embedded = torch.cat([embedded, embedded_dirs], -1)
            raw_fine = model_fine(embedded)
            if raw is None:
                raw = raw_fine
            else:
                raw = torch.cat((raw, raw_fine), dim=0)

        alpha = raw[..., 3]
        raw = raw.cpu().numpy()

        def occupancy_activation(alpha, distances):
            occ = 1.0 - torch.exp(-F.relu(alpha) * distances)
            # notice we apply RELU to raw sigma before computing alpha
            return occ

        voxel_size = (args.far - args.near) / args.N_importance  # or self.N_importance
        occ = occupancy_activation(alpha, voxel_size)
        print("Compute Occupancy Grids")
        occ = occ.reshape(grid_dim, grid_dim, grid_dim)
        occupancy_grid = occ.detach().cpu().numpy()

        print('fraction occupied:', (occupancy_grid > threshold).mean())
        print('Max Occ: {}, Min Occ: {}, Mean Occ: {}'.format(occupancy_grid.max(), occupancy_grid.min(),
                                                              occupancy_grid.mean()))
        vertices, faces, vertex_normals, _ = ski_measure.marching_cubes(occupancy_grid, level=level,
                                                                        gradient_direction='ascent')
        print()

        dim = occupancy_grid.shape[0]
        vertices = vertices / (dim - 1)
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)

        # Transform to [-1, 1] range
        mesh_canonical = mesh.copy()
        mesh_canonical.apply_translation([-0.5, -0.5, -0.5])
        mesh_canonical.apply_scale(2)

        scene_scale = scene_extents / 2.0
        # Transform to scene coordinates
        mesh_canonical.apply_scale(scene_scale)
        # mesh_canonical.apply_transform(scene_transform)
        # mesh.show()
        exported = trimesh.exchange.export.export_mesh(mesh_canonical,
                                                       os.path.join(save_dir, 'mesh_canonical.ply'))
        print("Saving Marching Cubes mesh to mesh_canonical.ply !")
        exported = trimesh.exchange.export.export_mesh(mesh_canonical, os.path.join(save_dir, 'mesh.ply'))
        print("Saving Marching Cubes mesh to mesh.ply !")

        o3d_mesh = trimesh_to_open3d(mesh)
        o3d_mesh_canonical = trimesh_to_open3d(mesh_canonical)

        print('Removing noise ...')
        print(
            f'Original Mesh has {len(o3d_mesh_canonical.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical.triangles) / 1e6:.2f} M faces.')
        o3d_mesh_canonical_clean = clean_mesh(o3d_mesh_canonical, keep_single_cluster=False,
                                              min_num_cluster=400)

        vertices_ = np.array(o3d_mesh_canonical_clean.vertices).reshape([-1, 3]).astype(np.float32)
        triangles = np.asarray(o3d_mesh_canonical_clean.triangles)  # (n, 3) int
        N_vertices = vertices_.shape[0]
        print(
            f'Denoised Mesh has {len(o3d_mesh_canonical_clean.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical_clean.triangles) / 1e6:.2f} M faces.')

        print("###########################################")
        print()
        print("Using Normals for colour predictions!")
        print()
        print("###########################################")

        ## use normal vector method as suggested by the author, see https://github.com/bmild/nerf/issues/44
        mesh_recon_save_dir = os.path.join(save_dir, "use_vertex_normal")
        os.makedirs(mesh_recon_save_dir, exist_ok=True)

        selected_mesh = o3d_mesh_canonical_clean
        rays_d = - torch.FloatTensor(
            np.asarray(selected_mesh.vertex_normals))  # use negative normal directions as ray marching directions
        near = 0.1 * torch.ones_like(rays_d[:, :1])
        far = 10.0 * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near

        full_ins = None
        N = rays_o.shape[0]
        print(N)
        print(rays_o.shape)
        z_val_coarse = z_val_sample(args.N_test, args.near, args.far, args.N_samples)
        with torch.no_grad():
            for step in range(0, N, args.N_test):
                N_test = args.N_test
                if step + N_test > N:
                    N_test = N - step
                    z_val_coarse = z_val_sample(N_test, args.near, args.far, args.N_samples)
                rays_io = rays_o[step:step + N_test]  # (chuck, 3)
                rays_id = rays_d[step:step + N_test]  # (chuck, 3)
                batch_rays = torch.stack([rays_io, rays_id], dim=0)
                batch_rays = batch_rays.to(args.device)
                all_info = ins_nerf(batch_rays, position_embedder, view_embedder,
                                    model_coarse, model_fine, z_val_coarse, args)
                if full_ins is None:
                    full_ins = all_info['ins_fine']
                else:
                    full_ins = torch.cat((full_ins, all_info['ins_fine']), dim=0)
        ins = full_ins.cpu().numpy()
        print(ins.shape)
        pred_label = np.argmax(ins, axis=-1)

        ins_color = render_label2rgb(pred_label, ins_rgbs)
        print(ins_color)
        o3d_mesh_canonical_clean.vertex_colors = o3d.utility.Vector3dVector(ins_color / 255.0)
        o3d.io.write_triangle_mesh(
            os.path.join(mesh_recon_save_dir, 'ins_mesh_canonical_dim{}neart_{}.ply'.format(grid_dim, args.near)),
            o3d_mesh_canonical_clean)
        print("Saving Marching Cubes mesh to instance_mesh_canonical_dim{}neart_{}.ply".format(grid_dim, args.near))

        print("--------end----------")
