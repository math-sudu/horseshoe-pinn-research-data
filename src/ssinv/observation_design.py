"""Reflection-closed locations and explicit new-reflector installation actions.

Geometry and installation inputs determine this design before response scoring.
Physical geometry defines the seed grid and its reflected locations.
"""
from __future__ import annotations

import numpy as np

from .survey_geometry import layouts, stations
from .measurements import baseline_points, coordinate_rows


def contour_position(section, position, role="initial_support_inner"):
    """Unwrapped arclength in [0, L], including the final interval endpoint."""
    fraction = float(position["fraction"])
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("An arc fraction must lie in [0, 1]")
    offset = 0.
    for arc in section.boundaries[role]["circular_arcs"]:
        length = section.scale * arc["radius"] * arc["sweep_rad"]
        if arc["arc_id"] == position["arc"]:
            return offset + fraction * length
        offset += length
    raise ValueError(f"Unknown contour arc: {position['arc']}")


def contour_distance(s, other, perimeter):
    """Shortest separation along the closed contour, in metres."""
    delta = np.abs(np.asarray(s) - np.asarray(other)) % perimeter
    return np.minimum(delta, perimeter - delta)


def reflection_closed_layout(section, seed_count=24, n_eval=600):
    """Keep the original seed locations and add their physical reflections."""
    layout = layouts(section, seed_count, n_eval)
    perimeter = layout["inner_perimeter_m"]
    crown_length = contour_position(section, dict(arc="crown", fraction=1.))
    seeds = np.array([p["s_m"] for p in layout["candidates"]])
    positions = np.sort(np.r_[seeds, (crown_length - seeds) % perimeter])
    positions = positions[np.r_[True, np.diff(positions) > 1e-9]]
    if len(positions) > 1 and contour_distance(positions[0], positions[-1], perimeter) < 1e-9:
        positions = positions[:-1]
    candidates = stations(section, "initial_support_inner", positions)
    for i, point in enumerate(candidates):
        mirror = int(np.argmin(contour_distance(positions, crown_length - point["s_m"], perimeter)))
        point.update(id=f"C{i+1:02d}", kind="normal", depth=section.thickness,
                     mirror_index=mirror, mirror_id=f"C{mirror+1:02d}")
    layout["candidates"] = candidates

    # Physical locations, rather than N/C labels, identify installed points.
    layout["baseline"][0]["s_m"] = contour_position(section, layout["baseline"][0])
    for chord in layout["baseline"][1:]:
        for endpoint in chord["endpoints"]:
            endpoint["s_m"] = contour_position(section, endpoint)

    # An equal grid starting at the springline is not reflection-closed on a
    # horseshoe. Centre the quadrature on a symmetry axis as well as the pool.
    for key, role in (("outer", "initial_support_outer"), ("inner", "initial_support_inner")):
        length = layout[f"{key}_perimeter_m"]
        crown = contour_position(section, dict(arc="crown", fraction=.5), role)
        grid = np.sort((crown + (np.arange(n_eval) + .5) * length / n_eval) % length)
        layout[key] = stations(section, role, grid)
    return layout


def resolve_intervals(section, intervals):
    """Resolve geometric endpoints; start > end crosses the springline seam."""
    return [[contour_position(section, interval["start"]),
             contour_position(section, interval["end"])] for interval in intervals]


def in_intervals(s, intervals, perimeter):
    """Intervals are closed; an empty list covers no locations."""
    s %= perimeter
    for start, end in intervals:
        if end - start >= perimeter:
            return True
        start, end = start % perimeter, end % perimeter
        if (start <= s <= end) if start <= end else (s >= start or s <= end):
            return True
    return False


def installation_actions(layout, installed, accessible, visibility_exclusions, minimum_clearance):
    """List every candidate's clearance and independent exclusion reasons."""
    if not np.isfinite(minimum_clearance) or minimum_clearance < 0:
        raise ValueError("Installation clearance must be finite and nonnegative")
    perimeter = layout["inner_perimeter_m"]
    installed_s = np.array([p["s_m"] for p in installed])
    actions = []
    for index, point in enumerate(layout["candidates"]):
        distances = contour_distance(point["s_m"], installed_s, perimeter)
        nearest = int(np.argmin(distances))
        clearance = float(distances[nearest])
        excluded = []
        if clearance < 1e-9:
            excluded.append("already_installed")
        elif clearance < minimum_clearance:
            excluded.append("installation_clearance")
        if not in_intervals(point["s_m"], accessible, perimeter):
            excluded.append("outside_access")
        if in_intervals(point["s_m"], visibility_exclusions, perimeter):
            excluded.append("visibility_exclusion")
        actions.append(dict(candidate_index=index, candidate_id=point["id"],
                            clearance_m=clearance, nearest_installed_id=installed[nearest]["id"],
                            excluded_by=excluded))
    return actions


def build_design(section, config, n_eval=600):
    """Return a response-ready layout and its four installation scenarios.

    Raw coordinate rows order the original seven reflectors then C candidates.
    Relocations resolve their physical position into those rows. A relocated
    reflector keeps its installed identity and is excluded from new actions.
    """
    layout = reflection_closed_layout(section, config["candidate_seed_count"], n_eval)
    original = baseline_points(layout)
    original_indices = {p["id"]: i for i, p in enumerate(original)}
    perimeter = layout["inner_perimeter_m"]
    positions = np.array([p["s_m"] for p in layout["candidates"]])
    baselines = {}
    for name, spec in config["baseline_layouts"].items():
        installed = [dict(p) for p in original]
        indices = list(range(len(original)))
        for identifier, location in spec["relocations"].items():
            index = original_indices[identifier]
            s = contour_position(section, location)
            matches = np.flatnonzero(contour_distance(positions, s, perimeter) < 1e-9)
            if len(matches) != 1:
                raise ValueError("A relocated baseline point must occur once in the response candidate grid")
            candidate = int(matches[0])
            point = layout["candidates"][candidate]
            installed[index] = dict(id=identifier, **{key: point[key] for key in
                                    ("arc", "fraction", "depth", "xy_m", "s_m")})
            indices[index] = len(original) + candidate
        baselines[name] = (installed, coordinate_rows(indices).tolist())

    cases = []
    for spec in config["layouts"]:
        installed, rows = baselines[spec["baseline"]]
        region = config["access_regions"][spec["access"]]
        accessible = resolve_intervals(section, region["accessible_intervals"])
        exclusions = resolve_intervals(section, region["visibility_exclusions"])
        actions = installation_actions(layout, installed, accessible, exclusions,
                                       config["minimum_contour_clearance_m"])
        cases.append(dict(name=spec["name"], baseline=spec["baseline"],
                          candidate_access=spec["access"], baseline_rows=rows,
                          installed_points=installed, accessible_intervals_m=accessible,
                          visibility_exclusions_m=exclusions,
                          allowed_indices=[a["candidate_index"] for a in actions if not a["excluded_by"]],
                          actions=actions))
    return layout, cases
