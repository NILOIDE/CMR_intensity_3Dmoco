import math
from typing import Union, Optional, Tuple, List, Dict

import numpy as np
import torch

from data_classes import SubjectData
from data_utils import find_subjects

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0


def normalize_image(im: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, 1] """
    min_, max_ = 0.0, im.max()
    im_ = (im - min_) / (max_ - min_)
    return im_


def normalize_image_with_mean_lv_value(im: Union[np.ndarray, torch.Tensor], mean_value=MEAN_SAX_LV_VALUE, target_value=0.5) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize such that LV pool has value of 0.5. Assumes min value is 0.0. """
    return im / (mean_value / target_value)


def fast_trilinear_interpolation(input_array: torch.Tensor,
                                 y_indices: torch.Tensor,
                                 x_indices: torch.Tensor,
                                 z_indices: torch.Tensor) -> torch.Tensor:
    """ Trilinear interpolation of a batch of 3D volumes.
     :param input_array: Images used as source for the sampling.                Shape: (batch, height, width, depth)
     :param y_indices: Indices of the 1st spatial dimension of a given image.   Shape: (batch, num_points)
     :param x_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     :param z_indices: Input image of shape (batch, height, width, depth)       Shape: (batch, num_points)
     """
    x0 = torch.floor(y_indices.detach()).to(torch.long)
    y0 = torch.floor(x_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = y_indices - x0
    y = x_indices - y0
    z = z_indices - z0

    b, _ = torch.meshgrid(torch.arange(0, x.shape[0], device=x.device),
                          torch.arange(0, x.shape[1], device=x.device))
    b_ = b.reshape(-1)
    x0_ = x0.reshape(-1)
    x1_ = x1.reshape(-1)
    y0_ = y0.reshape(-1)
    y1_ = y1.reshape(-1)
    z0_ = z0.reshape(-1)
    z1_ = z1.reshape(-1)
    x_ = x.reshape(-1)
    y_ = y.reshape(-1)
    z_ = z.reshape(-1)
    output_ = (
        input_array[b_, x0_, y0_, z0_] * (1 - x_) * (1 - y_) * (1 - z_) +
        input_array[b_, x1_, y0_, z0_] * x_ * (1 - y_) * (1 - z_) +
        input_array[b_, x0_, y1_, z0_] * (1 - x_) * y_ * (1 - z_) +
        input_array[b_, x0_, y0_, z1_] * (1 - x_) * (1 - y_) * z_ +
        input_array[b_, x1_, y0_, z1_] * x_ * (1 - y_) * z_ +
        input_array[b_, x0_, y1_, z1_] * (1 - x_) * y_ * z_ +
        input_array[b_, x1_, y1_, z0_] * x_ * y_ * (1 - z_) +
        input_array[b_, x1_, y1_, z1_] * x_ * y_ * z_
    )
    output = output_.reshape(x0.shape)
    return output


def flip_affine(affines, needs_flip):
    # If the original affine had a determinant is <= 0, it is an umproper affine matrix and it needs to be flipped
    needs_flip = needs_flip[:, None, None].repeat((1, 4, 4))
    flip = torch.eye(3, dtype=affines.dtype, device=affines.device)
    flip[0, 0] = -1
    flip = flip.repeat((affines.shape[0], 1, 1))
    affines_flipped = affines.clone()
    affines_flipped[:, :3, :3] = torch.bmm(flip, affines[:, :3, :3])
    affines_flipped[:, :3, 3:] = torch.bmm(flip, affines[:, :3, 3:])
    affines = torch.where(needs_flip, affines_flipped, affines)
    return affines


def mat_to_params(affines, spacings, needs_flip, cy_thresh=1e-3):
    affines = flip_affine(affines, needs_flip)
    affines[:, :3, :3] = torch.bmm(affines[:, :3, :3], torch.diag_embed(1 / spacings))
    translation = affines[:, :3, 3]

    # The rotation euler params are extracted following nibabel's mat2euler
    cy = torch.sqrt(affines[:, 2, 2] * affines[:, 2, 2] + affines[:, 1, 2] * affines[:, 1, 2])  # math.sqrt(r33 * r33 + r23 * r23)

    z = torch.atan2(-affines[:, 0, 1], affines[:, 0, 0])
    y = torch.atan2(affines[:, 0, 2], cy)
    x = torch.atan2(-affines[:, 1, 2], affines[:, 2, 2])

    z_eps = torch.atan2(-affines[:, 1, 0], affines[:, 1, 1])
    x_eps = torch.zeros_like(x)

    z = torch.where(cy > cy_thresh, z, z_eps)
    x = torch.where(cy > cy_thresh, x, x_eps)
    rotation = torch.stack((z, y, x), dim=1)

    params = torch.cat((rotation, translation), 1)
    return params


def params_to_mat(params: torch.Tensor, spacings: torch.Tensor, needs_flip: torch.Tensor):
    assert params.shape[1] == 6
    rotation, translation = params[:, :3], params[:, 3:]
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation_z = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_z[:, 0, 0] = cos[:, 0]
    rotation_z[:, 0, 1] = -sin[:, 0]
    rotation_z[:, 1, 0] = sin[:, 0]
    rotation_z[:, 1, 1] = cos[:, 0]

    rotation_y = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_y[:, 0, 0] = cos[:, 1]
    rotation_y[:, 0, 2] = sin[:, 1]
    rotation_y[:, 2, 0] = -sin[:, 1]
    rotation_y[:, 2, 2] = cos[:, 1]

    rotation_x = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_x[:, 1, 1] = cos[:, 2]
    rotation_x[:, 1, 2] = -sin[:, 2]
    rotation_x[:, 2, 1] = sin[:, 2]
    rotation_x[:, 2, 2] = cos[:, 2]

    rotation = torch.bmm(rotation_x, rotation_y)
    rotation = torch.bmm(rotation, rotation_z)
    rotation = torch.bmm(rotation, torch.diag_embed(spacings))

    affines = torch.eye(4, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    affines[:, :3, :3] = rotation
    affines[:, :3, 3] = translation
    affines = flip_affine(affines, needs_flip)
    return affines


def to_radians(x):
    return x * math.pi / 180


def to_degrees(x):
    return x * 180 / math.pi


def calculate_distances(p1, p2):
    """ Calculate distances from every point in p1 to every point in p2 """
    assert len(p1.shape) == len(p2.shape)
    assert p1.shape[-1] == p2.shape[-1]
    dists = p1.unsqueeze(-2) - p2.unsqueeze(-3)
    dists = dists * dists
    dists = dists.sum(-1)
    return dists


def interpolate_la_with_sa(la_images: List[torch.Tensor], la_affines: List[torch.Tensor],
                           sa_images: List[torch.Tensor], sa_affines: List[torch.Tensor]) \
        -> List[torch.Tensor]:

    la_interps = []
    for la_im, la_aff in zip(la_images, la_affines):
        sa_coords = []
        sa_values = []
        for i, (sax_im, aff) in enumerate(zip(sa_images, sa_affines)):
            sac = torch.meshgrid(torch.arange(0, sax_im.shape[0], dtype=la_im.dtype, device=la_im.device),
                                 torch.arange(0, sax_im.shape[1], dtype=la_im.dtype, device=la_im.device))
            sac = torch.stack(sac, -1).reshape(-1, 2)
            sac = torch.cat((sac, torch.zeros_like(sac[:, :1]), torch.ones_like(sac[:, :1])), -1)
            sacw = (aff @ sac.t()).t()
            sav = sax_im[..., 0].reshape(-1, 1)
            sa_coords.append(sacw)
            sa_values.append(sav)
        sa_coords = torch.cat(sa_coords, 0)[:, :3]
        sa_values = torch.cat(sa_values, 0)

        pad = 0
        lac = torch.meshgrid(torch.arange(0 - pad, la_im.shape[0] + pad, dtype=la_im.dtype, device=la_im.device),
                             torch.arange(0 - pad, la_im.shape[1] + pad, dtype=la_im.dtype, device=la_im.device))
        lac = torch.stack(lac, -1).reshape(-1, 2)
        lac = torch.cat((lac, torch.zeros_like(lac[:, :1]), torch.ones_like(lac[:, :1])), -1)
        lac_world = (la_aff @ lac.t()).t()
        lac_world = lac_world[:, :3]

        # To avoid memory issues, split the distance calculation into pieces
        closest = []
        step_size = 1
        for i in range(0, lac_world.shape[0], step_size):
            c = calculate_distances(lac_world[i:i + step_size], sa_coords).argmin(-1)
            closest.append(c)
        closest = torch.cat(closest, 0)

        la_newv = sa_values[closest]
        la_newv = la_newv.reshape(la_im.shape[0] + pad * 2, la_im.shape[1] + pad * 2)
        la_newv = la_newv.cpu()
        la_interps.append(la_newv)
    return la_interps


if __name__ == '__main__':
    download_dir = r"D:\UKBB_subjects"
    registered_dataset_dir = r"D:\UKBB_subjects_aligned"
    sax_unregistered_dataset_dir = r"D:\UKBB_subjects_unaligned"
    subject_list = find_subjects(download_dir, sax_unregistered_dataset_dir, num_subj=1)
    subject = SubjectData(subject_list[0])
    # Find la images by name
    la_images = [im.cpu() for im, name in zip(subject.images, subject.names) if name[:2] == "la"]
    la_affines = [aff.cpu() for aff, name in zip(subject.affines, subject.names) if name[:2] == "la"]

    sa_images = [im.cpu() for im, name in zip(subject.images, subject.names) if name[:2] == "sa"]
    sa_affines = [aff.cpu() for aff, name in zip(subject.affines, subject.names) if name[:2] == "sa"]
    la_interpolations = interpolate_la_with_sa(la_images, la_affines, sa_images, sa_affines)
    la_interpolations = [normalize_image(i.numpy()) for i in la_interpolations]
    print(1)