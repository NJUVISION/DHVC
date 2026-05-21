import torch

from dhic_codec.models import dhic


def test_legacy_conditioning_checkpoint_keys_are_remapped():
    model = dhic()
    state = model.state_dict()

    state["enc1_lmb_embedding"] = state.pop("conditioning.lambda_embeddings.enc1").clone()
    state["z1_scale_embedding"] = state.pop("conditioning.z_scale_embeddings.z1").clone()

    model.remap_legacy_state_dict(state)
    result = model.load_state_dict(state, strict=False)

    assert "conditioning.lambda_embeddings.enc1" not in result.missing_keys
    assert "conditioning.z_scale_embeddings.z1" not in result.missing_keys
    assert "enc1_lmb_embedding" not in result.unexpected_keys
    assert "z1_scale_embedding" not in result.unexpected_keys


def test_model_forward_smoke():
    model = dhic()
    model.eval()
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        metrics, fdict = model(x, cur_qp=3, return_fdict=True)

    assert fdict["x_hat"].shape == x.shape
    assert "loss" in metrics
    assert "bpp" in metrics
