import torch


class Metric:
    def mask_out_metric(self, metric: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        mask = self.merge_masks(mask1, mask2)
        # Mask out section of line out of bounds
        metric *= mask
        metric = torch.sum(metric, dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
        metric[~mask.any(2).any(1)] = 9999.0
        return metric

    def merge_masks(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        # Line section where both images are in-bounds
        mask = torch.ones_like(mask1)
        mask = torch.logical_and(mask, mask1)
        mask = torch.logical_and(mask, mask2)
        return mask


class L2(Metric):
    def __call__(self, sapmle1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        diff = sapmle1 - sample2
        metric = diff * diff  # L2
        metric = self.mask_out_metric(metric, mask1, mask2)
        return metric


class L1(Metric):
    def __call__(self, sapmle1: torch.Tensor, sample2: torch.Tensor,
                 mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        diff = sapmle1 - sample2
        metric = torch.abs(diff)  # L1
        metric = self.mask_out_metric(metric[..., :1], mask1[..., :1], mask2[..., :1])
        return metric
