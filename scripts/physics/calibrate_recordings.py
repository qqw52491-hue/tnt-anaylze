#!/usr/bin/env python3
"""Offline shot audit and frozen-candidate comparison. Requires NumPy.

Input is export_ballistics JSONL from immutable raw snapshots. Never reads a
device, changes recordings, rewrites Rust constants, or starts the HUD/game.
Schema: verified embedded protobuf descriptors in the reconstructed client ELF.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np


def varint(data, offset):
    result = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("varint exceeds u64")
        result |= (byte & 127) << shift
        if byte < 128:
            return result, offset
    raise ValueError("overlong varint")


def parse(data):
    result = {}
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if not 0 < number <= 0x1fffffff:
            raise ValueError("invalid field number")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if offset + size > len(data):
                raise ValueError("truncated fixed field")
            value = struct.unpack_from("<d" if wire == 1 else "<f", data, offset)[0]
            offset += size
        elif wire == 2:
            size, offset = varint(data, offset)
            if size > len(data) - offset:
                raise ValueError("truncated bytes")
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError("unsupported wire type")
        result.setdefault(number, []).append((wire, value))
    return result


def one(fields, number, wire, default=None):
    return next((v for w, v in fields.get(number, []) if w == wire), default)


def subs(fields, number):
    return [parse(value) for wire, value in fields.get(number, []) if wire == 2]


def signed32(value):
    value = int(value) & 0xffffffff
    return value - 0x100000000 if value & 0x80000000 else value


def integers(fields, number):
    result = []
    for wire, value in fields.get(number, []):
        if wire == 0:
            result.append(value)
        elif wire == 2:
            offset = 0
            while offset < len(value):
                v, offset = varint(value, offset)
                result.append(v)
    return result


def xy(fields):
    x, y = one(fields, 1, 0), one(fields, 2, 0)
    if x is None or y is None:
        raise ValueError("missing required XY")
    return signed32(x), signed32(y)


def wind_action(fields, framing):
    if framing not in ("le32", "be16"):
        raise ValueError("unknown envelope framing")
    raw, round_number, speed = (one(fields, n, 0) for n in (4, 1, 3))
    order = integers(fields, 2)
    if raw is None or round_number is None or not order:
        raise ValueError("missing wind/round/order")
    wind = signed32(raw)
    if framing == "le32":
        if speed is None:
            raise ValueError("missing legacy speed")
        wind ^= signed32(round_number) ^ signed32(speed)
        for player in order:
            wind ^= signed32(player)
    return signed32(wind) / 10.0, signed32(round_number), order[0]


def extract(path):
    with path.open() as lines:
        return extract_rows(lines)


def extract_rows(lines):
    states, shots, counts, fingerprints = {}, [], Counter(), set()
    for line in lines:
        row = json.loads(line)
        key = row["source"], row["pid"], row["stream"]
        state = states.setdefault(key, {"battle": 0, "wind": None, "pending": None, "ambiguous": False})
        fields = parse(bytes.fromhex(row["body_hex"]))
        if row["name"] == "FireReq":
            counts["requests"] += 1
            if not row["c2s"]:
                counts["request_wrong_direction"] += 1
                continue
            if state["pending"] is not None or state["ambiguous"]:
                counts["overlapping_requests"] += 1
                state["pending"] = None
                state["ambiguous"] = True
                continue
            required = [one(fields, 1, 0), one(fields, 2, 0),
                        one(fields, 3, 0), one(fields, 4, 1)]
            if any(v is None for v in required):
                counts["request_missing_required"] += 1
                state["pending"] = None
                continue
            round_number, left, angle, force = required
            angle = signed32(angle)
            if not (left in (0, 1) and 0 <= angle <= 180 and
                    math.isfinite(force) and 0 < force <= 150):
                counts["invalid_request"] += 1
                state["pending"] = None
                continue
            state["pending"] = dict(row, round=signed32(round_number), left=left,
                                    angle=angle, force=force, wind=state["wind"])
            continue
        if row["name"] != "BattleNotify" or row["c2s"]:
            continue
        kind = one(fields, 2, 0)
        if kind == 0:
            state["battle"] += 1
            state["pending"] = None
            state["wind"] = None
            state["ambiguous"] = False
            counts["battle_inits"] += 1
        elif kind == 8:
            try:
                action = subs(fields, 8)
                state["wind"] = wind_action(action[0], row["framing"]) if len(action) == 1 else None
            except ValueError:
                state["wind"] = None
            # A request never carries across the next turn.
            state["pending"] = None
            state["ambiguous"] = False
        elif kind == 15:
            counts["fire_notices"] += 1
            request = state["pending"]
            state["pending"] = None
            if request is None:
                counts["no_pending_request"] += 1
                continue
            fire_list = subs(fields, 14)
            if len(fire_list) != 1 or request["wind"] is None or not state["battle"]:
                counts["missing_context"] += 1
                continue
            fire = fire_list[0]
            wind, round_number, active = request["wind"]
            delay = row["game_ms"] - request["game_ms"]
            # The two schema fields are both named round, but the client
            # request counter is not the Action display-round counter. Use
            # same stream, uninterrupted turn, active UID, left and angle.
            if not (0 <= delay <= 1500
                    and one(fire, 1, 0) == active and one(fire, 2, 0) == request["left"]):
                counts["pair_context_mismatch"] += 1
                continue
            counts["paired_requests"] += 1
            if (one(fire, 4, 0, 0) != 0 or one(fire, 5, 0, 0) != 0
                    or one(fire, 6, 0, 0) != 0):
                counts["special_fire_flags"] += 1
                continue
            combos = subs(fire, 10)
            if not combos:
                counts["no_combo"] += 1
                continue
            # Only the first primary branch. Extra salvos/fans do not get
            # treated as independent observations of the same request force.
            primary = [b for b in subs(combos[0], 1)
                       if one(b, 1, 0) == 0 and
                       one(b, 3, 1) == float(request["angle"])]
            if len(primary) != 1:
                counts["no_unique_primary_angle"] += 1
                continue
            tracks = [t for t in subs(primary[0], 2) if one(t, 1, 0, 0) == 0]
            if len(tracks) != 1:
                counts["no_unique_projectile_track"] += 1
                continue
            track = tracks[0]
            try:
                points = [xy(p) for p in subs(track, 2)]
                origin_list = subs(fire, 3)
                origin = xy(origin_list[0]) if len(origin_list) == 1 else None
            except ValueError:
                counts["missing_coordinates"] += 1
                continue
            event_indices = [one(e, 1, 0) for e in subs(track, 3)]
            if any(v is None for v in event_indices):
                counts["missing_event_index"] += 1
                continue
            event_indices = [signed32(v) for v in event_indices if v is not None]
            end = min([len(points)] + [v for v in event_indices if v >= 0])
            prefix = points[:end]
            if len(prefix) < 12:
                counts["short_pre_event_track"] += 1
                continue
            if not all(abs(v) <= 1000000 for p in prefix for v in p) or abs(wind) > 100:
                counts["outside_domain"] += 1
                continue
            fingerprint = hashlib.sha256(json.dumps(
                [request["angle"], request["force"], wind, prefix],
                separators=(",", ":")).encode()).hexdigest()
            if fingerprint in fingerprints:
                counts["duplicate_trajectory"] += 1
                continue
            fingerprints.add(fingerprint)
            shots.append({"source": row["source"], "pid": row["pid"],
                          "stream": row["stream"], "battle": state["battle"],
                          "framing": row["framing"], "round": round_number,
                          "request_round": request["round"],
                          "angle": request["angle"], "force": request["force"],
                          "left": request["left"], "wind": wind, "origin": origin,
                          "points": prefix, "all_points": len(points),
                          "first_event": min(event_indices) if event_indices else None,
                          "request_offset": request["raw_offset"],
                          "fire_offset": row["raw_offset"], "pair_delay_ms": delay,
                          "fingerprint": fingerprint})
            counts["accepted"] += 1
    return shots, counts


FROZEN = dict(k=0.9792415487505262, gravity=4.202064201553961,
              wind_accel=0.02668859228049719, slope=1.4830798630263653,
              intercept=0.06399350205524457, initial_wind=0.013021726383376776,
              initial_down=2.109151970666793)


def coefficients(k, t):
    distance = -np.expm1(t * math.log(k)) / (1 - k)
    acceleration = (t - distance) / (1 - k)
    return distance, acceleration


def predict(shot, model):
    t = np.arange(len(shot["points"]), dtype=float)
    travel, accel = coefficients(model["k"], t)
    angle = math.radians(shot["angle"])
    speed = model["slope"] * shot["force"] + model["intercept"]
    vx = speed * math.cos(angle) + model["initial_wind"] * shot["wind"]
    vy = -speed * math.sin(angle) + model["initial_down"]
    return np.column_stack((travel * vx + accel * model["wind_accel"] * shot["wind"],
                            travel * vy + accel * model["gravity"]))


def metrics(shots, model):
    per_shot, final, biases = [], [], []
    for shot in shots:
        points = np.asarray(shot["points"], dtype=float)
        errors = predict(shot, model) - (points - points[0])
        radial = np.linalg.norm(errors, axis=1)
        per_shot.append(float(np.sqrt(np.mean(radial ** 2))))
        final.append(float(radial[-1]))
        biases.extend(errors.tolist())
    if not shots:
        return {"shots": 0}
    return {"shots": len(shots), "battles": len({(s["source"], s["pid"], s["stream"], s["battle"]) for s in shots}),
            "points": sum(len(s["points"]) for s in shots),
            "shot_rmse_median": float(np.median(per_shot)),
            "shot_rmse_mean": float(np.mean(per_shot)),
            "shot_rmse_p95": float(np.percentile(per_shot, 95)),
            "prefix_end_error_median": float(np.median(final)),
            "prefix_end_error_p95": float(np.percentile(final, 95)),
            "mean_xy_bias": np.mean(biases, axis=0).tolist()}


def fit(shots):
    # Variable projection: k is nonlinear; the other six coefficients are
    # linear for fixed k. Every shot has equal total weight despite its length.
    prepared = []
    for shot in shots:
        points = np.asarray(shot["points"], dtype=float)
        t = np.arange(1, len(points), dtype=float)
        angle = math.radians(shot["angle"])
        prepared.append((t, points[1:] - points[0], math.cos(angle),
                         math.sin(angle), shot["force"], shot["wind"]))

    def solve(k):
        matrices, observed = [], []
        for t, points, co, si, power, wind in prepared:
            travel, accel = coefficients(k, t)
            z = np.zeros_like(t)
            x = np.column_stack((travel * co * power, travel * co,
                                 travel * wind, z, accel * wind, z))
            y = np.column_stack((-travel * si * power, -travel * si,
                                 z, travel, z, accel))
            weight = 1 / math.sqrt(len(t))
            matrices.extend((x * weight, y * weight))
            observed.extend((points[:, 0] * weight, points[:, 1] * weight))
        matrix, target = np.vstack(matrices), np.concatenate(observed)
        values, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        if rank != 6:
            raise ValueError("insufficient parameter identifiability")
        # A tiny negative fitted intercept is rounding noise, not evidence of
        # backwards launch velocity at zero force. Solve the constrained face.
        if values[1] < 0:
            keep = [0, 2, 3, 4, 5]
            bounded, _, bounded_rank, _ = np.linalg.lstsq(matrix[:, keep], target, rcond=None)
            if bounded_rank != 5:
                raise ValueError("insufficient constrained parameter identifiability")
            values = np.zeros(6)
            values[keep] = bounded
        residual = matrix @ values - target
        return float(residual @ residual / len(shots)), values

    # Fixed bounds and deterministic search; validation/test never enter fit.
    grid = np.linspace(0.95, 0.995, 91)
    scored = [(solve(float(k))[0], float(k)) for k in grid]
    _, best = min(scored)
    lo, hi = max(0.95, best - 0.0005), min(0.995, best + 0.0005)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fc, fd = solve(c)[0], solve(d)[0]
    for _ in range(45):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = solve(c)[0]
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = solve(d)[0]
    k = (lo + hi) / 2
    loss, values = solve(k)
    model = dict(zip(["slope", "intercept", "initial_wind", "initial_down", "wind_accel", "gravity"], map(float, values)))
    return dict(model, k=k), loss


def partition(shot):
    # Keep complete sessions/dates together. The newest client is the domain
    # to calibrate; old framing is an external compatibility check only.
    if shot["framing"] != "be16":
        return "legacy_external"
    stamp = int(shot["source"].split("-")[0])
    # 2026-10-18 / 2026-10-19 00:00:00 Asia/Shanghai.
    if stamp < 1792252800000:
        return "train"
    if stamp < 1792339200000:
        return "validation"
    return "test"


def scale_family(shot):
    # Geometry from independently decoded notification origin and track start,
    # not a threshold on model residual. Cause/pre-shot selector is unresolved.
    if shot["origin"] is None:
        return "unknown"
    radius = math.dist(shot["points"][0], shot["origin"])
    if 27 <= radius <= 30:
        return "ordinary_origin_scale"
    if 55 <= radius <= 58:
        return "double_origin_scale"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("envelopes", type=Path)
    parser.add_argument("new_output_directory", type=Path)
    args = parser.parse_args()
    args.new_output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    shots, counts = extract(args.envelopes)
    report = {"counts": dict(counts), "units": "server world coordinates, pre-event prefixes, first-point aligned",
              "frozen": FROZEN, "families": {}, "rust_sources_modified_by_this_tool": False}
    for name in ["ordinary_origin_scale", "double_origin_scale"]:
        rows = [s for s in shots if scale_family(s) == name]
        groups = {group: [s for s in rows if partition(s) == group]
                  for group in ["train", "validation", "test", "legacy_external"]}
        if len(groups["train"]) < 30:
            report["families"][name] = {"shots": len(rows), "error": "fewer than 30 training shots"}
            continue
        candidate, loss = fit(groups["train"])
        report["families"][name] = {"shots": len(rows), "candidate": candidate, "training_loss": loss,
                                   "groups": {group: {"frozen": metrics(g, FROZEN), "candidate": metrics(g, candidate)}
                                              for group, g in groups.items()}}
    for filename, obj in [("shots.json", shots), ("report.json", report)]:
        path = args.new_output_directory / filename
        path.write_text(json.dumps(obj, indent=2, allow_nan=False))
        path.chmod(0o600)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
