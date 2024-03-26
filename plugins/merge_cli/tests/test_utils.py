"""Tests for merging utilities."""

import math
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from git_theta_merge_cli.merges import utils


def test_is_seq():
    data = (
        (1.0, False),
        (3, False),
        (None, False),
        ("", False),
        ("test", False),
        ([], True),
        ([23, 45], True),
        (tuple(), True),
        ((4.6, 1.0), True),
        ({1: "a", 2: "b"}.values(), True),
        ({1: "a", 2: "b"}.keys(), True),
        (np.arange(10), True),
        (torch.arange(10), True),
    )
    for in_, gold in data:
        assert utils.is_seq(in_) == gold


def test_get_merge_lambdas_scalar_2():
    merge_lambda = random.random()
    ml, cml = utils.get_merge_lambdas(merge_lambda, size=2)
    assert ml == merge_lambda
    assert math.isclose(math.fsum([ml, cml]), 1)


def test_get_merge_lambdas_scalar_2_and_uniform():
    merge_lambda = random.random()
    ml, cml = utils.get_merge_lambdas(merge_lambda, size=2, uniform=True)
    assert math.isclose(math.fsum([ml, cml]), 1.0)


def test_get_merge_lambdas_None_2():
    ml, cml = utils.get_merge_lambdas(None, size=2)
    assert ml == cml
    assert ml == 1.0


def test_get_merge_lambdas_None_uniform_2():
    ml, cml = utils.get_merge_lambdas(None, size=2, uniform=True)
    assert ml == cml
    assert math.isclose(math.fsum([ml, cml]), 1.0)


def test_get_merge_lambdas_scalar_uniform():
    length = random.randint(5, 10)
    mls = utils.get_merge_lambdas(None, length, uniform=True)
    assert len(mls) == length
    for ml in mls:
        assert ml == 1 / length
    assert math.isclose(math.fsum(mls), 1.0)


def test_get_merge_lambdas_scalar_uniform_wrong_value():
    length = random.randint(5, 10)
    with patch("git_theta_merge_cli.merges.utils.logging.warning") as p:
        mls = utils.get_merge_lambdas(1, length, uniform=True)
    p.assert_called_once()
    assert len(mls) == length
    for ml in mls:
        assert ml == 1 / length
    assert math.isclose(math.fsum(mls), 1.0)


def test_get_merge_lambdas_scalar_uniform_right_value():
    length = random.randint(5, 10)
    with patch("git_theta_merge_cli.merges.utils.logging.warning") as p:
        mls = utils.get_merge_lambdas(1 / length, length, uniform=True)
    p.assert_not_called()
    assert len(mls) == length
    for ml in mls:
        assert ml == 1 / length
    assert math.isclose(math.fsum(mls), 1.0)


def test_get_merge_lambdas_None_uniform():
    length = random.randint(5, 10)
    mls = utils.get_merge_lambdas(None, size=length, uniform=True)
    assert len(mls) == length
    for ml in mls:
        assert ml == 1 / length
    assert math.isclose(math.fsum(mls), 1.0)


def test_get_merge_lambda_None():
    length = random.randint(5, 10)
    mls = utils.get_merge_lambdas(None, length, uniform=False)
    assert len(mls) == length
    for ml in mls:
        assert ml == 1.0


def test_get_merge_lambdas_scalar_replicate():
    merge_lambda = random.random()
    length = random.randint(5, 10)
    mls = utils.get_merge_lambdas(merge_lambda, length, uniform=False)
    assert len(mls) == length
    for ml in mls:
        assert ml == merge_lambda


def test_get_merge_lambdas_list_wrong_size():
    length = random.randint(5, 10)
    offset = random.choice([-2, -1, 1, 2])
    merge_lambda = [random.random() for _ in range(length + offset)]
    with pytest.raises(ValueError):
        utils.get_merge_lambdas(merge_lambda, length)


def test_get_merge_lambdas_list():
    length = random.randint(5, 10)
    merge_lambda = [random.random() for _ in range(length)]
    mls = utils.get_merge_lambdas(merge_lambda, length)
    assert mls == merge_lambda
    assert len(mls) == length
    mls = utils.get_merge_lambdas(np.asarray(merge_lambda), length)
    np.testing.assert_allclose(mls, merge_lambda)
    mls = utils.get_merge_lambdas(torch.Tensor(merge_lambda), length)
    np.testing.assert_allclose(mls.numpy(), merge_lambda)


def test_get_merge_lambdas_list_overrides_uniform():
    length = random.randint(5, 10)
    merge_lambda = [random.random() for _ in range(length)]
    mls = utils.get_merge_lambdas(merge_lambda, length, uniform=True)
    assert mls == merge_lambda
    assert len(mls) == length
    mls = utils.get_merge_lambdas(np.asarray(merge_lambda), length)
    np.testing.assert_allclose(mls, merge_lambda)
    mls = utils.get_merge_lambdas(torch.Tensor(merge_lambda), length)
    np.testing.assert_allclose(mls.numpy(), merge_lambda)


def test_interpolate_sum():
    length = random.randint(5, 10)
    tensors = [torch.rand(100, 200) for _ in range(length)]
    gold = [[0 for _ in range(tensors[0].shape[1])] for _ in range(tensors[0].shape[0])]
    for t in tensors:
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                gold[i][j] += t[i, j]
    np.testing.assert_allclose(np.asarray(gold), utils.interpolate(tensors))


def test_memory_efficient_sum():
    length = random.randint(5, 10)
    tensors = [torch.rand(100, 200) for _ in range(length)]
    assert torch.allclose(
        utils.memory_efficient_interpolate(tensors), utils.interpolate(tensors)
    )
