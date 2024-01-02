
import sys
import os
import torch
from glob import glob

root = "/".join(sys.path[0].split('/')[:-2])
sys.path.insert(0, root)

import doorcam.aggregation as agg

def test_l2norm():
    norm = agg.L2Norm()
    tensor = torch.randn(100, 50)
    normed_tensor = norm(tensor)
    row_norms = torch.norm(normed_tensor, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms))


def test_mac_aggregation():
    fn = agg.MAC()
    tensor = torch.randn(10, 128, 14, 14)
    desc = fn(tensor)
    assert desc.ndim == 2
    assert desc.shape[0] == 10
    assert desc.shape[1] == 128


def test_spoc_aggregation():
    fn = agg.SPoC()
    tensor = torch.randn(10, 128, 14, 14)
    desc = fn(tensor)
    assert desc.ndim == 2
    assert desc.shape[0] == 10
    assert desc.shape[1] == 128


def test_gem_aggregation():
    fn = agg.GeM()
    tensor = torch.randn(10, 128, 14, 14)
    desc = fn(tensor)
    assert desc.ndim == 2
    assert desc.shape[0] == 10
    assert desc.shape[1] == 128



    