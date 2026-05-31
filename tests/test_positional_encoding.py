import torch

from src.positional_encoding import SinusoidalPositionalEncoding


def test_output_shape_is_preserved():
    x = torch.zeros(4, 10, 16)
    pos_enc = SinusoidalPositionalEncoding(d_model=16, dropout=0.0)

    y = pos_enc(x)

    assert y.shape == x.shape


def test_position_zero_pattern():
    x = torch.zeros(1, 3, 8)
    pos_enc = SinusoidalPositionalEncoding(d_model=8, dropout=0.0)

    y = pos_enc(x)

    expected = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    assert torch.allclose(y[0, 0], expected)


def test_encoding_is_not_trainable_parameter():
    pos_enc = SinusoidalPositionalEncoding(d_model=8)

    parameter_names = [name for name, _ in pos_enc.named_parameters()]
    buffer_names = [name for name, _ in pos_enc.named_buffers()]

    assert "encoding" not in parameter_names
    assert "encoding" in buffer_names