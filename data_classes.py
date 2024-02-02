from dataclasses import dataclass
from typing import Optional
from itertools import combinations, product
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from data_utils import SubjectFiles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OptimizableImage:
    image: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    name: Optional[str] = None
    seg: Optional[np.ndarray] = None


class SubjectData:
    def __init__(self, subject: SubjectFiles):
        self.planes = []
        self.seg_planes = []
        index = 0
        if subject.la4ch is not None:
            try:
                nii_4ch = nib.load(subject.la4ch)
                nii_4ch_seg = None
                if subject.la4ch_seg is not None:
                    try:
                        nii_4ch_seg = nib.load(subject.la4ch_seg).dataobj[:]
                    except FileNotFoundError as e:
                        print(f"Subject {subject.name}: 4ch segmentation not found.")
                self.planes.append(OptimizableImage(nii_4ch.dataobj[:].squeeze(2), nii_4ch.affine, nii_4ch.header.get_zooms()[:3], Path(subject.la4ch).name.split(".")[0], seg=nii_4ch_seg))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 4ch image not found.\n", e)
        if subject.la3ch is not None:
            try:
                nii_3ch = nib.load(subject.la3ch)
                nii_3ch_seg = None
                if subject.la3ch_seg is not None:
                    try:
                        nii_3ch_seg = nib.load(subject.la3ch_seg).dataobj[:]
                    except FileNotFoundError as e:
                        print(f"Subject {subject.name}: 3ch segmentation not found.")
                self.planes.append(OptimizableImage(nii_3ch.dataobj[:].squeeze(2), nii_3ch.affine, nii_3ch.header.get_zooms()[:3], Path(subject.la3ch).name.split(".")[0], seg=nii_3ch_seg))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 3ch image not found.\n", e)
        if subject.la2ch is not None:
            try:
                nii_2ch = nib.load(subject.la2ch)
                nii_2ch_seg = None
                if subject.la2ch_seg is not None:
                    try:
                        nii_2ch_seg = nib.load(subject.la2ch_seg).dataobj[:]
                    except FileNotFoundError as e:
                        print(f"Subject {subject.name}: 2ch segmentation not found.")
                self.planes.append(OptimizableImage(nii_2ch.dataobj[:].squeeze(2), nii_2ch.affine, nii_2ch.header.get_zooms()[:3], Path(subject.la2ch).name.split(".")[0], seg=nii_2ch_seg))
                index += 1
            except FileNotFoundError as e:
                print(f"Subject {subject.name}: 2ch image not found.\n", e)
        # Index pairs for all Long-axis to Long-axis image pairs (if 2ch, 3ch, 4ch are present, that will be 3 pairs)
        la_la_product = [*combinations(list(range(len(self.planes))), 2)]
        # Long-axis to Short-axis image pairs combinations (if 3 LA images and N SA images, that will be 3*N pairs)
        sa_la_product = list(product(list(range(len(self.planes))), list(range(len(self.planes), len(self.planes) + len(subject.sax)))))
        self.idx_pairs = torch.tensor(la_la_product + sa_la_product)
        # Load Short-axis planes
        segs = subject.sax_seg if subject.sax_seg is not None else [None]*len(subject.sax)
        for i, (sa_lice, sa_lice_seg) in enumerate(zip(subject.sax, segs)):
            im_nii = nib.load(sa_lice)
            nii_seg = None
            if subject.la4ch_seg is not None:
                nii_seg = nib.load(sa_lice_seg).dataobj[:]
            self.planes.append(OptimizableImage(im_nii.dataobj[:].squeeze(2), im_nii.affine, im_nii.header.get_zooms()[:3], Path(sa_lice).name.split(".")[0], seg=nii_seg))
            index += 1
        assert min([min(i) for i in self.idx_pairs]) == 0
        assert max([max(i) for i in self.idx_pairs]) == len(self.planes) - 1

        self.images = [torch.tensor(i.image, dtype=torch.float32, device=DEVICE) for i in self.planes]
        self.shapes = torch.stack([torch.tensor(i.shape, dtype=torch.int64, device=DEVICE) for i in self.images])
        self.affines = torch.stack([torch.tensor(i.affine, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)
        self.spacings = torch.stack([torch.tensor(i.spacing, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)
        self.names = [i.name for i in self.planes]
        self.max_im_shape = self.shapes.amax(dim=0)
        self.images_pad = torch.zeros((len(self.images), *self.max_im_shape), dtype=torch.float32, device=DEVICE)
        for i, (im, sh) in enumerate(zip(self.images, self.shapes)):
            self.images_pad[i, :sh[0], :sh[1]] = im
            assert im.any(1).any(0).all()

    def update_affines(self, affines: torch.Tensor):
        affines = affines.detach().cpu().numpy()
        assert affines.shape == (len(self.planes), 4, 4)
        for i in range(affines.shape[0]):
            self.planes[i].affine = affines[i]
        self.affines = torch.stack([torch.tensor(i.affine, dtype=torch.float32, device=DEVICE) for i in self.planes], dim=0)

    def save_niftis(self, directory: str, affines: torch.Tensor = None):
        if affines is not None:
            self.update_affines(affines)
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        for plane in self.planes:
            nii = nib.Nifti1Image(plane.image[:, :, None], affine=plane.affine)
            if plane.name[:2] == "sa":
                path = Path(directory) / "sa_slices"
                path.mkdir(exist_ok=True)
                path = path / (plane.name + ".nii.gz")
            else:
                path = Path(directory) / (plane.name + ".nii.gz")
            nib.save(nii, path)

            if plane.seg is not None:
                nii = nib.Nifti1Image(plane.seg[:, :, None], affine=plane.affine)
                if plane.name[:2] == "sa":
                    path = Path(directory) / "sa_slices"
                    path.mkdir(exist_ok=True)
                    path = path / ("seg_" + plane.name + ".nii.gz")
                else:
                    path = Path(directory) / ("seg_" + plane.name + ".nii.gz")
                nib.save(nii, path)
        return
