import cv2
import numpy as np

import simulation as sim
from qvio.frontend import KLTTracker, TrackerConfig, load_tracks, save_tracks


def textured(shift_x: int, shift_y: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur((rng.random((600, 900)) * 255).astype(np.uint8), (0, 0), 2.0)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    return big[60 + shift_y:540 + shift_y, 70 + shift_x:822 + shift_x].copy()


def test_tracks_follow_a_pure_shift(tmp_path):
    tracker = KLTTracker(sim.CAMERA, TrackerConfig(clahe=False))
    first = tracker.track(textured(0, 0), 0)
    second = tracker.track(textured(3, -2), 50_000_000)
    common, a, b = np.intersect1d(first.ids, second.ids, return_indices=True)
    assert len(common) > 100
    flow = second.px[b] - first.px[a]
    np.testing.assert_allclose(np.median(flow, axis=0), [-3, 2], atol=0.1)

    save_tracks(tmp_path / "t.npz", [first, second])
    loaded = load_tracks(tmp_path / "t.npz")
    np.testing.assert_array_equal(loaded[1].ids, second.ids)
    np.testing.assert_allclose(loaded[1].xy, second.xy, atol=1e-6)
