import torch

from dhic_codec.eval import crop, pad


def test_pad_and_crop_round_trip():
    x = torch.rand(1, 3, 65, 70)

    padded = pad(x, p=64)
    restored = crop(padded, (65, 70))

    assert padded.shape[-2:] == (128, 128)
    assert torch.equal(restored, x)
