import os
from pathlib import Path

import pytest


MODEL = Path(__file__).resolve().parents[1] / "weights" / "model.pth"


@pytest.mark.skipif(not MODEL.is_file(), reason="pretrained model is not downloaded")
def test_codec_round_trip(tmp_path):
    from PIL import Image
    from dhic import Codec

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (120, 80, 40)).save(source)

    codec = Codec(str(MODEL), device=os.environ.get("DHIC_TEST_DEVICE", "cuda:0"))
    stream = codec.encode(str(source), qp=31)
    reconstruction = codec.decode(stream)

    assert len(stream) > 0
    assert reconstruction.size == (64, 64)
