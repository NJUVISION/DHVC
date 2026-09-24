import math

import scipy.stats
import torch
import torch.distributions as td
from compressai.entropy_models import GaussianConditional
from compressai.ops import LowerBound


def _check_scale_table(scale_table):
    if not isinstance(scale_table, torch.Tensor):
        raise TypeError("scale_table must be a tensor")
    if scale_table.dim() != 1 or scale_table.numel() == 0:
        raise ValueError("scale_table must be one-dimensional and nonempty")
    if scale_table.min() <= 0 or not torch.equal(scale_table, scale_table.sort()[0]):
        raise ValueError("scale_table must contain sorted positive values")


class DiscretizedGaussian(GaussianConditional):
    def __init__(self, scale_table=None):
        super(GaussianConditional, self).__init__()
        if scale_table is None:
            scale_table = self._get_default_scale_table()
        _check_scale_table(scale_table)
        self.register_buffer("scale_table", scale_table, persistent=False)
        self.tail_mass = 1e-9
        self.lower_bound_scale = LowerBound(scale_table[0])
        self.standard_gaussian = td.Normal(loc=0, scale=1)

    @staticmethod
    def _get_default_scale_table():
        return torch.exp(torch.linspace(math.log(0.11), math.log(20.0), 64))

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def _standardized_cumulative(self, inputs):
        return self.standard_gaussian.cdf(inputs)
