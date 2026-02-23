import numpy as np

from soccer_vit.metrica.normalize import PitchSpec, norm_to_meters


def test_norm_to_meters_center():
    x_m, y_m = norm_to_meters(0.5, 0.5, pitch=PitchSpec(length_m=105.0, width_m=68.0))
    assert np.isclose(x_m, 0.0)
    assert np.isclose(y_m, 0.0)


def test_norm_to_meters_y_flip():
    _, y_top = norm_to_meters(0.5, 0.0, pitch=PitchSpec(length_m=105.0, width_m=68.0))
    _, y_bottom = norm_to_meters(0.5, 1.0, pitch=PitchSpec(length_m=105.0, width_m=68.0))
    assert y_top > 0
    assert y_bottom < 0
