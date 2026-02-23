import numpy as np

from soccer_vit.labeling.linebreak import LineBreakParams, bypassed_defenders, label_line_break


def test_bypassed_count_matches_expected_simple_case():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([20.0, 0.0])
    defenders = np.array(
        [
            [5.0, 1.0],   # bypassed
            [10.0, -2.0], # bypassed
            [15.0, 9.0],  # outside corridor
            [25.0, 0.0],  # beyond receiver x
            [-5.0, 0.0],  # behind passer
        ]
    )
    mask, _, _ = bypassed_defenders(p0, p1, defenders, corridor_w_m=8.0)
    assert int(mask.sum()) == 2


def test_label_line_break_threshold_k():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([20.0, 0.0])
    defenders = np.array([[6.0, 1.0], [11.0, -1.0]])
    params = LineBreakParams(min_forward_m=5.0, corridor_w_m=8.0, k_bypassed=2)
    res = label_line_break(p0, p1, defenders, params=params)
    assert res.bypassed_count == 2
    assert res.label == 1
