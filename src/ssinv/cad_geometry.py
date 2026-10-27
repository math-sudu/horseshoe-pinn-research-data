"""Support contours and displacement monitoring locations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SupportSection:
    def __init__(self, path):
        self.path = None if isinstance(path, dict) else Path(path).resolve()
        self.raw = path if isinstance(path, dict) else json.loads(
            self.path.read_text(encoding="utf-8"))
        if self.raw["coordinate_unit"] != "m":
            raise ValueError("The support section must be expressed in metres")
        self.scale = float(self.raw["scale"])
        self.translation = np.asarray(self.raw["translation_m"], dtype=float)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("The section requires one positive similarity scale")
        self.boundaries = {b["role"]: b for b in self.raw["boundaries"]}
        self.inner = self.boundaries["initial_support_inner"]
        self.outer = self.boundaries["initial_support_outer"]
        self.crown_y = self.scale * self.inner["bounds_max"][1] + self.translation[1]
        self._validate()

    def _validate(self):
        thicknesses = []
        for inner, outer in zip(self.inner["circular_arcs"],
                                self.outer["circular_arcs"], strict=True):
            if inner["arc_id"] != outer["arc_id"]:
                raise ValueError("Inner and outer arc correspondence differs")
            if not np.allclose(inner["center"], outer["center"], atol=1e-10):
                raise ValueError("Paired support arcs must have the same centre")
            for key in ("start_rad", "sweep_rad"):
                if not np.isclose(inner[key], outer[key], atol=1e-10):
                    raise ValueError("Paired support arcs have different intervals")
            thicknesses.append(outer["radius"] - inner["radius"])
        if min(thicknesses) <= 0 or np.ptp(thicknesses) > 1e-10:
            raise ValueError("The supplied section has inconsistent normal thickness")
        self.thickness = self.scale * float(np.mean(thicknesses))
        for role, boundary in self.boundaries.items():
            arcs = boundary["circular_arcs"]
            for left, right in zip(arcs, arcs[1:] + arcs[:1]):
                p = self.point(role, left["arc_id"], 1.0)
                q = self.point(role, right["arc_id"], 0.0)
                if np.linalg.norm(p - q) > 1e-8:
                    raise ValueError("The supplied CAD boundary is not closed")

    def arc(self, role, arc_id):
        return next(a for a in self.boundaries[role]["circular_arcs"]
                    if a["arc_id"] == arc_id)

    def point(self, role, arc_id, fraction):
        arc = self.arc(role, arc_id)
        t = arc["start_rad"] + np.asarray(fraction) * arc["sweep_rad"]
        p = np.asarray(arc["center"]) + arc["radius"] * np.stack(
            (np.cos(t), np.sin(t)), axis=-1)
        return self.scale * p + self.translation

    def sample(self, role, points_per_metre=20):
        pieces = []
        for arc in self.boundaries[role]["circular_arcs"]:
            n = max(4, int(np.ceil(self.scale * arc["radius"]
                                   * arc["sweep_rad"] * points_per_metre)))
            pieces.append(self.point(role, arc["arc_id"],
                                     np.linspace(0, 1, n, endpoint=False)))
        p = np.concatenate(pieces)
        return np.vstack((p, p[0]))

    def horizontal_targets(self, depth):
        """Intersect a recorded horizontal line with the inner support contour."""
        y = self.crown_y - float(depth)
        hits = self.horizontal_intersections("initial_support_inner", y)
        if len(hits) != 2:
            raise ValueError(f"Depth {depth} m must intersect two inner-contour points")
        return hits

    def horizontal_intersections(self, role, y):
        """Exact horizontal cuts of the circular arcs forming a CAD contour."""
        y_source = (y - self.translation[1]) / self.scale
        hits = []
        for arc in self.boundaries[role]["circular_arcs"]:
            c = np.asarray(arc["center"])
            q = (y_source - c[1]) / arc["radius"]
            if abs(q) > 1 + 1e-12:
                continue
            a = np.arcsin(np.clip(q, -1, 1))
            for t in (a, np.pi - a):
                delta = (t - arc["start_rad"]) % (2 * np.pi)
                if delta <= arc["sweep_rad"] + 1e-10:
                    x = c[0] + arc["radius"] * np.cos(t)
                    hits.append(self.scale * x + self.translation[0])
        xs = np.unique(np.round(hits, decimals=11))
        return np.column_stack((xs, np.full(len(xs), y)))

    def local_frame(self, arc_id, fraction):
        arc = self.arc("initial_support_outer", arc_id)
        angle = arc["start_rad"] + fraction * arc["sweep_rad"]
        normal = np.array([np.cos(angle), np.sin(angle)])
        return normal, np.array([-normal[1], normal[0]])

    def monitoring_points(self):
        points = {"GD": self.point("initial_support_inner", "crown", 0.5)}
        for pair, depth in self.raw["monitoring"]["chord_depth_m"].items():
            left, right = pair.split("-")
            p = self.horizontal_targets(depth)
            points[left], points[right] = p[0], p[1]
        return points

    def locate(self, role, point, tolerance=1e-8):
        """Return the CAD arc and fraction of a point on the named contour."""
        p = (np.asarray(point)-self.translation)/self.scale
        for arc in self.boundaries[role]["circular_arcs"]:
            relative = p-np.asarray(arc["center"])
            if abs(np.linalg.norm(relative)-arc["radius"])*self.scale > tolerance:
                continue
            angle = np.arctan2(relative[1], relative[0])
            delta = (angle-arc["start_rad"]) % (2*np.pi)
            if abs(delta-2*np.pi) < tolerance:
                delta = 0.
            if delta <= arc["sweep_rad"]+tolerance:
                return arc["arc_id"], min(delta/arc["sweep_rad"], 1.)
        raise ValueError("Point does not lie on the specified CAD contour")
