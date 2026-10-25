"""Ensure the publication oracle cannot excuse unexplained zero scale values."""
import importlib.util
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('publication_oracle',
    Path(__file__).parents[1]/'gpu/validate_publication.py')
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def tensors(values):
    torch=pytest.importorskip('torch')
    pre=torch.tensor(values,dtype=torch.float32)
    return torch,pre,torch.exp(pre)


def test_zero_requires_exact_activation_and_strict_rounding_evidence():
    torch,pre,scales=tensors([-104.,-150.,0.])
    evidence=oracle.validate_scale_underflow(scales,pre)
    assert evidence['zero_count']==2
    assert evidence['exact_upstream_exp_verified']
    assert evidence['zero_exp_float64_max']<2.**-150
    torch.testing.assert_close(scales,torch.exp(pre),rtol=0,atol=0)


def test_unexplained_zero_is_rejected():
    torch,pre,scales=tensors([-100.])
    scales[0]=0
    with pytest.raises(AssertionError,match='not explained'):
        oracle.validate_scale_underflow(scales,pre)


def test_negative_or_nonfinite_scale_evidence_is_rejected():
    torch,pre,scales=tensors([-104.])
    with pytest.raises(AssertionError,match='Negative'):
        oracle.validate_scale_underflow(torch.full_like(scales,-1),pre)
    with pytest.raises(AssertionError,match='Nonfinite'):
        oracle.validate_scale_underflow(scales,torch.full_like(pre,float('-inf')))


def test_positive_activation_mismatch_is_rejected_even_with_valid_zero():
    torch,pre,scales=tensors([-104.,0.])
    scales[1]=2
    with pytest.raises(AssertionError,match='differ from exact'):
        oracle.validate_scale_underflow(scales,pre)
