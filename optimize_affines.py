import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Union, Iterable
import numpy as np
import torch
import tqdm
import cv2
import itertools
import pickle
import os
import matplotlib.pyplot as plt

from data_classes import SubjectData
from plot_results import plot_param_diff_before_after, plot_param_diff

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

from geo_utils import get_image_plane_from_array, plane_intersection, closest_point_on_line, batch_normalize_vector
from utils import normalize_image, normalize_image_with_mean_lv_value, fast_trilinear_interpolation, mat_to_params, \
    params_to_mat, to_radians
from data_utils import find_subjects
from metrics import L2, L1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BGR colors
c_red = (0, 0, 255)
c_green = (0, 255, 0)
c_mint = (175, 255, 127)
c_orange = (0, 127, 255)
c_blue = (255, 0, 0)

METRIC = L2()


def visualize_intersection_differences(images, affines, affines_new, pair_indices, shapes, names=None):
    metric = METRIC
    images1, images2 = images[pair_indices[:, 0]], images[pair_indices[:, 1]]
    shapes1, shapes2 = shapes[pair_indices[:, 0]], shapes[pair_indices[:, 1]]

    sample_coords_scanner_space = compute_intersection_sampling_line(affines, pair_indices, shapes)
    affines1, affines2 = affines[pair_indices[:, 0]], affines[pair_indices[:, 1]]
    sampled_im1, sample_mask1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1, shapes1)
    sampled_im2, sample_mask2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2, shapes2)
    loss = metric(sampled_im1, sampled_im2, mask1=sample_mask1, mask2=sample_mask2)

    sample_coords_scanner_space = compute_intersection_sampling_line(affines_new, pair_indices, shapes)
    affines1_new, affines2_new = affines_new[pair_indices[:, 0]], affines_new[pair_indices[:, 1]]
    sampled_im1_new, sample_mask1_new, line_im1_new = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1_new, shapes1)
    sampled_im2_new, sample_mask2_new, line_im2_new = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2_new, shapes2)
    loss_new = metric(sampled_im1_new, sampled_im2_new, mask1=sample_mask1_new, mask2=sample_mask2_new)

    images = images.cpu().detach().numpy()
    pair_indices = pair_indices.cpu().detach().numpy()
    sampled_im1 = sampled_im1.cpu().detach().numpy()
    sampled_im2 = sampled_im2.cpu().detach().numpy()
    sampled_im1_new = sampled_im1_new.cpu().detach().numpy()
    sampled_im2_new = sampled_im2_new.cpu().detach().numpy()
    sample_mask1 = sample_mask1.cpu().detach().numpy()
    sample_mask2 = sample_mask2.cpu().detach().numpy()
    sample_mask1_new = sample_mask1_new.cpu().detach().numpy()
    sample_mask2_new = sample_mask2_new.cpu().detach().numpy()
    line_im1 = line_im1.cpu().detach().numpy()
    line_im2 = line_im2.cpu().detach().numpy()
    line_im1_new = line_im1_new.cpu().detach().numpy()
    line_im2_new = line_im2_new.cpu().detach().numpy()
    loss = loss.cpu().detach().numpy()
    loss_new = loss_new.cpu().detach().numpy()
    # Visualize
    for i in range(pair_indices.shape[0]):
        idx1, idx2 = pair_indices[i, 0], pair_indices[i, 1]
        name1, name2 = "im1", "im2"
        if names is not None:
            name1, name2 = names[idx1], names[idx2]
        scaling = 10
        # Original
        im1_vis = normalize_image_with_mean_lv_value(sampled_im1[i, ..., 0]) * 255
        im1_vis_rgb = np.stack([im1_vis]*3, axis=-1)
        im2_vis = normalize_image_with_mean_lv_value(sampled_im2[i, ..., 0]) * 255
        im2_vis_rgb = np.stack([im2_vis]*3, axis=-1)
        mask = np.logical_and(sample_mask1[i, ..., 0], sample_mask2[i, ..., 0]).astype(np.float32)
        mask_rgb = np.stack([mask]*3, -1)
        mask_rgb = np.where(mask_rgb, np.array([c_red]*mask.shape[0]), np.array([c_red]*mask.shape[0]))
        # New
        im1_vis_new = normalize_image_with_mean_lv_value(sampled_im1_new[i, ..., 0]) * 255
        im1_vis_new_rgb = np.stack([im1_vis_new] * 3, axis=-1)
        im2_vis_new = normalize_image_with_mean_lv_value(sampled_im2_new[i, ..., 0]) * 255
        im2_vis_new_rgb = np.stack([im2_vis_new] * 3, axis=-1)
        mask_new = np.logical_and(sample_mask1_new[i, ..., 0], sample_mask2_new[i, ..., 0]).astype(np.float32)
        mask_new_rgb = np.stack([mask_new]*3, -1)
        mask_new_rgb = np.where(mask_new_rgb, np.array([c_green]*mask.shape[0]), np.array([c_green]*mask.shape[0]))

        cat_line = np.stack((*[im1_vis_rgb]*2,
                             np.zeros_like(mask_rgb), mask_rgb, np.zeros_like(mask_rgb),
                             *[im2_vis_rgb]*2,
                             *[np.zeros_like(mask_rgb)]*5,
                             *[im1_vis_new_rgb]*2,
                             np.zeros_like(mask_new_rgb), mask_new_rgb, np.zeros_like(mask_new_rgb),
                             *[im2_vis_new_rgb]*2,), axis=0)
        cat_line_ = cv2.UMat(cat_line.astype(np.uint8))
        cat_line_ = cv2.resize(cat_line_, (im1_vis.shape[0] * scaling, cat_line.shape[0] * scaling))
        cv2.imshow(f"{name1}-{name2} sampling line.   "
                   f"Original loss (top):  {float('%.5f' % loss[i])},   "
                   f"New loss (bottom):  {float('%.5f' % loss_new[i])}    "
                   f"Loss diff:  {float('%.5f' % (loss_new[i] - loss[i]))}", cat_line_)

        # Plot images and draw sampling lines along images
        scaling = 3
        im1_vis = images[idx1, ..., 0]
        im2_vis = images[idx2, ..., 0]
        pad = np.zeros((im1_vis.shape[0], im1_vis.shape[1]//4))
        im_vis = np.concatenate((im1_vis, pad, im2_vis), axis=1)
        im_vis = normalize_image(im_vis) * 255
        # im_vis = np.concatenate((im_vis, im_vis), axis=0)
        im_vis_ = cv2.UMat(np.stack([im_vis.astype(np.uint8)]*3, axis=-1))
        im_vis_ = cv2.resize(im_vis_, (im2_vis.shape[1] * scaling * 2 + pad.shape[1] * scaling, im2_vis.shape[0] * scaling))

        # Plot lines of old affines
        for j in range(line_im1.shape[1]-1):
            p1 = (line_im1[i, j, :2] * scaling).round().astype(int)
            p2 = (line_im1[i, j+1, :2] * scaling).round().astype(int)
            c = c_red if sample_mask1[i, j, 0] else c_blue
            cv2.line(im_vis_,
                     (p1[1], p1[0],),
                     (p2[1], p2[0],),
                     c,
                     thickness=3)
        for j in range(line_im2.shape[1]-1):
            p1 = (line_im2[i, j, :2] * scaling).round().astype(int)
            p2 = (line_im2[i, j+1, :2] * scaling).round().astype(int)
            c = c_red if sample_mask1[i, j, 0] else c_blue
            cv2.line(im_vis_,
                     (p1[1] + images.shape[2] * scaling + pad.shape[1] * scaling, p1[0],),
                     (p2[1] + images.shape[2] * scaling + pad.shape[1] * scaling, p2[0],),
                     c,
                     thickness=3)
        # Plot lines of new affines
        for j in range(line_im1_new.shape[1]-1):
            p1 = (line_im1_new[i, j, :2] * scaling).round().astype(int)
            p2 = (line_im1_new[i, j+1, :2] * scaling).round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_blue
            cv2.line(im_vis_,
                     (p1[1], p1[0],),
                     (p2[1], p2[0],),
                     c,
                     thickness=3)
        for j in range(line_im2_new.shape[1]-1):
            p1 = (line_im2_new[i, j, :2] * scaling).round().astype(int)
            p2 = (line_im2_new[i, j+1, :2] * scaling).round().astype(int)
            c = c_green if sample_mask1[i, j, 0] else c_blue
            cv2.line(im_vis_,
                     (p1[1] + images.shape[2] * scaling + pad.shape[1] * scaling, p1[0],),
                     (p2[1] + images.shape[2] * scaling + pad.shape[1] * scaling, p2[0],),
                     c,
                     thickness=3)
        im_vis_ = cv2.resize(im_vis_, (im2_vis.shape[0] * scaling * 2, im2_vis.shape[0] * scaling))
        cv2.imshow(f"{name1}-{name2} intersection.  "
                   f"Original loss (top):  {float('%.5f' % loss[i])},   "
                   f"New loss (bottom):  {float('%.5f' % loss_new[i])}", im_vis_)

        cv2.waitKey()


def image_center_to_scanner_space(affines: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
    centers = torch.zeros((shapes.shape[0], 4), dtype=affines.dtype, device=affines.device)
    centers[:, -1] = 1.0
    centers[:, :2] = shapes[:, :2] / 2

    centers_scanner_space = torch.einsum("ijk,ik->ij", [affines, centers])
    return centers_scanner_space[..., :3]


def find_sampling_center(image_centers1, image_centers2, intersec_lines) -> torch.Tensor:
    centers1_on_line = closest_point_on_line(intersec_lines, image_centers1)
    centers2_on_line = closest_point_on_line(intersec_lines, image_centers2)
    sampling_centers = (centers1_on_line + centers2_on_line) / 2
    return sampling_centers


def compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_mm=5.0, num_samples=100) -> torch.Tensor:
    """ Compute a sampling line of num_samples points along the intersection lane between all plane pairs.
     Each line is centered on the average image centers. Each point is spaced sampling_step_mm appart. """
    planes = get_image_plane_from_array(affines)
    planes1, planes2 = planes[pair_indices[:, 0]], planes[pair_indices[:, 1]]
    # We assume the plane pairs always intersect
    intersec_scanner_space = plane_intersection(planes1, planes2)
    # The sampling line will be centered on the average image centers projected along the intersection line
    img_centers = image_center_to_scanner_space(affines, shapes)
    img_centers1, img_centers2 = img_centers[pair_indices[:, 0]], img_centers[pair_indices[:, 1]]
    sampling_centers = find_sampling_center(img_centers1, img_centers2, intersec_scanner_space)

    # We sample a line along the intersection N mm at a time for num_samples/2 in each direction
    intersec_dirs = intersec_scanner_space[:, 1] - intersec_scanner_space[:, 0]
    step_vectors = batch_normalize_vector(intersec_dirs)
    step_vectors_ = step_vectors[:, None].tile((1, num_samples, 1))
    dists_from_centers = (torch.arange(0, num_samples, dtype=affines.dtype, device=affines.device) - num_samples / 2)
    dists_from_centers *= sampling_step_mm
    dists_from_centers_ = dists_from_centers[None, :, None].tile((sampling_centers.shape[0], 1, 3))
    sampling_centers_ = sampling_centers[:, None].tile((1, num_samples, 1))
    sample_lines = sampling_centers_ + step_vectors_ * dists_from_centers_

    assert sample_lines.shape[0] == pair_indices.shape[0]
    assert sample_lines.shape[1] == num_samples
    assert sample_lines.shape[2] == 3
    return sample_lines


def sample_image_along_scanner_line(coords_scanner_space: torch.Tensor, images: torch.Tensor,
                                    affines: torch.Tensor, shapes: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = coords_scanner_space.dtype
    device = coords_scanner_space.device
    # Add 4th coord dimension in order to multiply by affine
    if coords_scanner_space.shape[-1] == 3:
        last_dim = torch.ones((*coords_scanner_space.shape[:-1], 1), dtype=dtype, device=device)
        coords_scanner_space = torch.cat((coords_scanner_space, last_dim), dim=-1)
    inverse_affines = torch.linalg.inv(affines)  # For (scanner space -> voxel) we need inverse affine
    # Tile affine into (batch, num_point, 4, 4)
    inverse_affines_ = inverse_affines[:, None].tile((1, coords_scanner_space.shape[1], 1, 1))
    # Batch-wise dot product
    points_voxel_space = torch.einsum("ijkl,ijl->ijk", [inverse_affines_, coords_scanner_space])
    # Disregard z dimension, we assume it's a 2D image. Ideally z should always be ~0.0.
    coords_voxel_space = points_voxel_space[:, :, :2]
    # We want to sample these point along all time points in the 2D+time scan. We need to generate the time indices.
    _, _, t = torch.meshgrid(torch.arange(0, coords_scanner_space.shape[0], dtype=dtype, device=device),
                             torch.arange(0, coords_scanner_space.shape[1], dtype=dtype, device=device),
                             torch.arange(0, images.shape[-1], dtype=dtype, device=device))
    # Concatenate time indices to xy coordinates
    coords_voxel_space_t = coords_voxel_space[:, :, None].tile((1, 1, images.shape[-1], 1))
    coords_voxel_space_t = torch.cat((coords_voxel_space_t, t[..., None]), dim=-1)
    # Treat 2D+time images are volumes and use trilinear interpolation to extract values at each time point
    sampled_points = fast_trilinear_interpolation(images,
                                                  coords_voxel_space_t[..., 0].reshape((images.shape[0], -1)),
                                                  coords_voxel_space_t[..., 1].reshape((images.shape[0], -1)),
                                                  coords_voxel_space_t[..., 2].reshape((images.shape[0], -1)))
    sampled_points = sampled_points.reshape((images.shape[0], coords_scanner_space.shape[1], images.shape[-1]))
    # Create mask to deliniate which sampled points were inside/outside image.
    shapes_ = shapes[:, None, :2].tile((1, coords_voxel_space.shape[1], 1))
    out_mask = torch.logical_or((coords_voxel_space < 0.0).any(dim=-1),
                                (coords_voxel_space > (shapes_ - 1)).any(dim=-1))
    in_mask = ~out_mask
    in_mask = in_mask[..., None].tile((1, 1, images.shape[-1]))
    return sampled_points, in_mask, coords_voxel_space


def compute_pairwise_loss(images, affines, pair_indices, shapes) -> torch.Tensor:
    metric = METRIC
    sample_coords_scanner_space = compute_intersection_sampling_line(affines, pair_indices, shapes)
    images1, images2 = images[pair_indices[:, 0]], images[pair_indices[:, 1]]
    affines1, affines2 = affines[pair_indices[:, 0]], affines[pair_indices[:, 1]]
    shapes1, shapes2 = shapes[pair_indices[:, 0]], shapes[pair_indices[:, 1]]
    sampled_im1, sample_mask1, line_im1 = sample_image_along_scanner_line(sample_coords_scanner_space, images1, affines1, shapes1)
    sampled_im2, sample_mask2, line_im2 = sample_image_along_scanner_line(sample_coords_scanner_space, images2, affines2, shapes2)
    loss = metric(sampled_im1, sampled_im2, mask1=sample_mask1, mask2=sample_mask2)
    return loss


def optimize_affines(images, affines, pair_indices, shapes, spacings,
                     max_epochs=5000, early_stop_epochs=1000, display_loss_curve=False):
    best_affines = affines.clone()
    best_loss = compute_pairwise_loss(images, best_affines, pair_indices, shapes).mean().item()
    best_epoch = 0
    needs_flip = torch.det(affines[:, :3, :3]) <= 0

    params_og = mat_to_params(affines, spacings, needs_flip)
    max_delta = torch.tensor([[torch.pi/8, torch.pi/8, torch.pi/8, 20, 20, 20]],
                             dtype=params_og.dtype, device=params_og.device)
    clamp_low = params_og - max_delta.tile((params_og.shape[0], 1))
    clamp_high = params_og + max_delta.tile((params_og.shape[0], 1))

    params = torch.nn.Parameter(params_og, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=1e-3)
    losses = []
    for i in tqdm.tqdm(range(max_epochs)):
        optimizer.zero_grad()
        epoch_affines = params_to_mat(params, spacings, needs_flip)
        loss = compute_pairwise_loss(images, epoch_affines, pair_indices, shapes)
        # TODO: Log image-wise and pair-wise losses?
        loss = loss.mean()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_affines = epoch_affines.clone().detach()
            best_epoch = i
        if i > early_stop_epochs and np.min(losses[-early_stop_epochs:]) > best_loss:
            break
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        params.data.clamp(clamp_low, clamp_high)

    param_diff = mat_to_params(best_affines.clone(), spacings, needs_flip) - mat_to_params(affines.clone(), spacings, needs_flip)
    if display_loss_curve and losses:
        print("Best loss:", best_loss, "Best epoch:", best_epoch)
        print("Rot diff:", param_diff[:, :3])
        print("Tra diff:", param_diff[:, 3:])
        plt.plot(losses)
        plt.xlabel("Optimization steps")
        plt.ylabel("Total Inter-slice Intensity Error (L2)")
        plt.title("Total intensity error across slices over optimization process")
        plt.show()
    new_params = mat_to_params(affines.clone(), spacings, needs_flip)
    return best_affines, new_params, param_diff


def parse_command_line():
    main_parser = argparse.ArgumentParser(description="Implicit Segmentation",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_subparsers = main_parser.add_subparsers(dest='pipeline')

    # Align existing image
    parser_align = main_subparsers.add_parser("align")
    parser_align.add_argument("-s", "--subject_dir",
                              help="Directory holding subject data", required=True,
                              )
    parser_align.add_argument("-u", "--unregistered_dir",
                              help="Directory in which to hold starting unregistered data", required=True,
                              )
    parser_align.add_argument("-r", "--registered_dir",
                              help="Directory in which to store final registered data", required=True,
                              )
    parser_align.add_argument("-e", "--max_epochs",
                              help="Maximum optimization epochs", required=False,
                              default=10000,
                              )
    parser_align.add_argument("-ese", "--early_stop_epochs",
                              help="Number of epochs of no impovement before optimization stops", required=False,
                              default=1000,
                              )
    parser_align.add_argument("-n", "--num_subjects",
                              help="Number of subjects to use.", required=False,
                              default=-1,
                              )
    parser_align.add_argument("-v", "--visualize",
                              help="Visualize loss curve and final aligned side-by-side image intersections", required=False,
                              default=False,
                              )

    # Randomly misalign slices and correct motion
    parser_random = main_subparsers.add_parser("random_align")
    parser_random.add_argument("-r", "--registered_dir",
                               help="Directory in which to store final registered data", required=True,
                               )
    parser_random.add_argument("-e", "--max_epochs",
                               help="Maximum optimization epochs", required=False,
                               default=1000,
                               )
    parser_random.add_argument("-ese", "--early_stop_epochs",
                               help="Number of epochs of no impovement before optimization stops", required=False,
                               default=100,
                               )
    parser_random.add_argument("-n", "--num_subjects",
                               help="Number of subjects to use.", required=False,
                               default=10,
                               )
    parser_random.add_argument("-nr", "--num_randomizations",
                               help="Number randomizations per each condition in each subject.", required=False,
                               default=10,
                               )
    return main_parser.parse_args()


def align_images(subject_dir: str, unregistered_dir: str, registered_dir: str,
                 max_epochs: int, early_stop_epochs: int, num_subjects: int, visualize=False):
    subject_list = find_subjects(subject_dir, unregistered_dir, num_subj=num_subjects)
    num_subjects = len(subject_list) if num_subjects <= 0 else num_subjects
    for idx, sub in enumerate(subject_list[:num_subjects]):
        print(f"Subject: {sub.name}     Idx: {idx}")
        subject_data = SubjectData(sub)

        images_pad = subject_data.images_pad.clone()
        affines = subject_data.affines.clone()
        idx_pairs = subject_data.idx_pairs.clone()
        shapes = subject_data.shapes.clone()
        spacings = subject_data.spacings.clone()
        plane_names = [p.name for p in subject_data.planes]

        start_loss = compute_pairwise_loss(images_pad, affines, idx_pairs, shapes)
        print(f"Starting loss: {start_loss.mean()}")
        print(f"Starting pair-wise loss:")
        for i in range(start_loss.shape[0]):
            print(f"  {plane_names[idx_pairs[i, 0]]} - {plane_names[idx_pairs[i, 1]]} loss:  {start_loss[i]}")

        new_affines, new_params, param_diff = optimize_affines(images_pad, affines, idx_pairs, shapes, spacings,
                                                               max_epochs=max_epochs,
                                                               early_stop_epochs=early_stop_epochs,
                                                               display_loss_curve=visualize)

        print("-------------------------------------------")
        final_loss = compute_pairwise_loss(images_pad, new_affines, idx_pairs, shapes)
        print(f"Final loss: {final_loss.mean()}")
        print(f"Final loss change: {final_loss.mean() - start_loss.mean()}")
        print(f"Final pair-wise loss:")
        for i in range(final_loss.shape[0]):
            print(
                f"  {plane_names[idx_pairs[i, 0]]} - {plane_names[idx_pairs[i, 1]]} loss change:  {final_loss[i] - start_loss[i]}")

        save_path = Path(registered_dir)
        save_path.mkdir(exist_ok=True)
        print(str(save_path / sub.name))
        subject_data.save_niftis(str(save_path / sub.name), new_affines)
        if visualize:
            visualize_intersection_differences(images_pad, affines, new_affines, idx_pairs, shapes, names=plane_names)

    print("----------------------------------------------------------------------------------------------")


def random_align_sweep(registered_dir, max_epochs, early_stop_epochs, num_subjects, num_randomizations):
    print("Deforming....", registered_dir)
    subject_list = find_subjects(registered_dir, registered_dir)
    num_subjects = len(subject_list) if num_subjects <= 0 else num_subjects
    rot_deform = [0, 5, 15, 45]  # In degrees
    tra_deform = [0, 5, 15, 45]  # In mm
    pickle_name = "results.pkl"
    try:
        with open(pickle_name, 'rb') as handle:
            results = pickle.load(handle)
            start = min([len(i) for i in results.values()])
            results = {k: v[:start] for k, v in results.items()}
    except (FileNotFoundError, FileExistsError) as e:
        results = {}

    for idx, sub in enumerate(subject_list[:num_subjects]):
        print(f"Subject: {sub.name}     Idx: {idx}")
        subject_data = SubjectData(sub)
        images_pad = subject_data.images_pad.clone()
        affines = subject_data.affines.clone()
        idx_pairs = subject_data.idx_pairs.clone()
        shapes = subject_data.shapes.clone()
        spacings = subject_data.spacings.clone()

        aff_needs_flip = torch.det(affines[:, :3, :3]) <= 0
        params = mat_to_params(affines, spacings, aff_needs_flip)
        for i, (r, t) in enumerate(itertools.product(rot_deform, tra_deform)):
            print(r, t)
            r_rad = to_radians(r)  # To radians
            result = []
            deform = torch.tensor([r_rad, r_rad, r_rad, t, t, t], device=params.device).reshape((1, params.shape[1]))
            deform = deform.tile((params.shape[0], 1))
            for j in range(num_randomizations):
                print(j)
                def_params = params.clone() + torch.rand_like(params) * deform - deform / 2
                def_affines = params_to_mat(def_params, spacings, aff_needs_flip)
                _, _, param_diff = optimize_affines(images_pad, def_affines, idx_pairs, shapes, spacings,
                                                    max_epochs=max_epochs, early_stop_epochs=early_stop_epochs)
                result.append(param_diff.detach().cpu())
            print(np.mean([i.abs().mean().detach().cpu().numpy() for i in result]),
                  np.std([i.abs().mean().detach().cpu().numpy() for i in result]))
            try:
                results[(r, t)].append(result)
            except KeyError:
                results[(r, t)] = [result]
            with open(pickle_name, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # #######################  Create plots  ###########################################
    plot_param_diff_before_after(results)
    plot_param_diff(results)


if __name__ == '__main__':
    args = parse_command_line()
    if args.pipeline is None or args.pipeline == "align":
        align_images(args.subject_dir, args.unregistered_dir, args.registered_dir,
                     int(args.max_epochs), int(args.early_stop_epochs), int(args.num_subjects), bool(args.visualize))
    elif args.pipeline == "random_align":
        random_align_sweep(args.registered_dir, int(args.max_epochs),
                           int(args.early_stop_epochs), int(args.num_subjects), int(args.num_randomizations))
    else:
        raise ValueError("Unknown pipeline selected.")
