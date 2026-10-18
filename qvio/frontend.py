"""KLT feature tracking on cam0.

Ported from the tracker in the original project (grid-limited Shi-Tomasi
corners, pyramidal LK with a forward-backward check) with two additions: a
fundamental-matrix RANSAC between consecutive frames, and CLAHE so the dark
Machine Hall sequences keep enough corners.

Tracks are reported as undistorted, normalised image coordinates so the
filter never needs the distortion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .euroc import Camera


@dataclass(frozen=True)
class TrackerConfig:
    max_features: int = 200
    grid: tuple[int, int] = (8, 5)          # columns, rows
    min_distance_px: float = 15.0
    quality_level: float = 0.01
    lk_window_px: int = 21
    lk_levels: int = 3
    max_fb_error_px: float = 1.0
    ransac_threshold_px: float = 1.0
    border_px: int = 10
    clahe: bool = True


@dataclass
class Frame:
    t: int
    ids: np.ndarray       # int64 track ids
    xy: np.ndarray        # [N, 2] undistorted normalised coordinates
    px: np.ndarray        # [N, 2] raw pixel coordinates (for drawing)


class KLTTracker:
    def __init__(self, camera: Camera, config: TrackerConfig = TrackerConfig()):
        self.camera = camera
        self.config = config
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if config.clahe else None
        self._prev: np.ndarray | None = None
        self._pts = np.empty((0, 2), np.float32)
        self._ids = np.empty(0, np.int64)
        self._next_id = 0

    def reset(self) -> None:
        """Drop every track, e.g. at the start of a camera outage."""
        self._prev = None
        self._pts = np.empty((0, 2), np.float32)
        self._ids = np.empty(0, np.int64)

    def undistort(self, px: np.ndarray) -> np.ndarray:
        if len(px) == 0:
            return np.empty((0, 2))
        out = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), self.camera.K, self.camera.dist)
        return out.reshape(-1, 2)

    def track(self, image: np.ndarray, t: int) -> Frame:
        gray = self._clahe.apply(image) if self._clahe is not None else image
        if self._prev is not None and len(self._pts):
            self._track_existing(gray)
        self._detect(gray)
        self._prev = gray
        return Frame(t, self._ids.copy(), self.undistort(self._pts), self._pts.astype(float).copy())

    def _track_existing(self, gray: np.ndarray) -> None:
        cfg = self.config
        lk = dict(
            winSize=(cfg.lk_window_px, cfg.lk_window_px),
            maxLevel=cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        p0 = self._pts.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, p0, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev, p1, None, **lk)
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(back.reshape(-1, 2) - self._pts, axis=1)
        w, h = self.camera.size
        b = cfg.border_px
        ok = (
            st1.ravel().astype(bool) & st2.ravel().astype(bool)
            & (fb < cfg.max_fb_error_px)
            & (p1[:, 0] >= b) & (p1[:, 0] < w - b) & (p1[:, 1] >= b) & (p1[:, 1] < h - b)
        )
        idx = np.flatnonzero(ok)
        if len(idx) >= 8:
            # Geometric check on undistorted pixels; KLT alone happily follows
            # reflections and moving shadows.
            u0 = self._to_pixels(self.undistort(self._pts[idx]))
            u1 = self._to_pixels(self.undistort(p1[idx]))
            _, mask = cv2.findFundamentalMat(u0, u1, cv2.FM_RANSAC, cfg.ransac_threshold_px, 0.99)
            if mask is not None:
                idx = idx[mask.ravel().astype(bool)]
        self._pts = p1[idx].astype(np.float32)
        self._ids = self._ids[idx]

    def _to_pixels(self, xy: np.ndarray) -> np.ndarray:
        K = self.camera.K
        return np.column_stack([xy[:, 0] * K[0, 0] + K[0, 2], xy[:, 1] * K[1, 1] + K[1, 2]])

    def _detect(self, gray: np.ndarray) -> None:
        cfg = self.config
        room = cfg.max_features - len(self._pts)
        if room <= 0:
            return
        h, w = gray.shape
        cols, rows = cfg.grid
        per_cell = int(np.ceil(cfg.max_features / (cols * rows)))
        mask = np.zeros_like(gray)
        mask[cfg.border_px:h - cfg.border_px, cfg.border_px:w - cfg.border_px] = 255
        for x, y in self._pts:
            cv2.circle(mask, (int(x), int(y)), int(cfg.min_distance_px), 0, -1)

        # Count existing tracks per cell so new corners go where tracks are thin.
        cell = lambda pts: (  # noqa: E731
            np.clip((pts[:, 0] * cols / w).astype(int), 0, cols - 1)
            + cols * np.clip((pts[:, 1] * rows / h).astype(int), 0, rows - 1)
        )
        occupancy = np.bincount(cell(self._pts), minlength=cols * rows) if len(self._pts) else np.zeros(cols * rows, int)

        new = []
        for r in range(rows):
            for c in range(cols):
                want = per_cell - occupancy[c + cols * r]
                if want <= 0:
                    continue
                x0, x1 = c * w // cols, (c + 1) * w // cols
                y0, y1 = r * h // rows, (r + 1) * h // rows
                corners = cv2.goodFeaturesToTrack(
                    gray[y0:y1, x0:x1], maxCorners=int(want), qualityLevel=cfg.quality_level,
                    minDistance=cfg.min_distance_px, mask=mask[y0:y1, x0:x1], blockSize=7,
                )
                if corners is not None:
                    new.append(corners.reshape(-1, 2) + [x0, y0])
        if not new:
            return
        new = np.concatenate(new)[:room].astype(np.float32)
        ids = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
        self._next_id += len(new)
        self._pts = np.vstack([self._pts, new])
        self._ids = np.concatenate([self._ids, ids])


def track_sequence(seq, config: TrackerConfig = TrackerConfig(), progress=None) -> list[Frame]:
    """Run the tracker over every cam0 frame. The result can be cached to disk."""
    tracker = KLTTracker(seq.camera, config)
    frames = []
    for i, t in enumerate(seq.cam_t):
        frames.append(tracker.track(seq.image(i), int(t)))
        if progress is not None:
            progress(i, len(seq.cam_t))
    return frames


def save_tracks(path, frames: list[Frame]) -> None:
    counts = np.array([len(f.ids) for f in frames])
    np.savez_compressed(
        path,
        t=np.array([f.t for f in frames], np.int64),
        counts=counts,
        ids=np.concatenate([f.ids for f in frames]) if frames else np.empty(0, np.int64),
        xy=np.concatenate([f.xy for f in frames]).astype(np.float32) if frames else np.empty((0, 2)),
        px=np.concatenate([f.px for f in frames]).astype(np.float32) if frames else np.empty((0, 2)),
    )


def load_tracks(path) -> list[Frame]:
    data = np.load(path)
    splits = np.cumsum(data["counts"])[:-1]
    ids = np.split(data["ids"], splits)
    xy = np.split(data["xy"].astype(float), splits)
    px = np.split(data["px"].astype(float), splits)
    return [Frame(int(t), i, x, p) for t, i, x, p in zip(data["t"], ids, xy, px)]
