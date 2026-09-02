import torch
from src.core.device import get_device


def test_get_device_auto():
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_get_device_explicit_cpu():
    device = get_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_get_device_explicit_case_insensitivity():
    device = get_device("CPU")
    assert device.type == "cpu"
