"""Compute BD-rate against the included VTM reference curves."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

VTM = {
    "kodak": ([0.17486630792194274, 0.28609626622750733, 0.4401069404436023,
               0.6176470678952847, 0.8133689654602636, 0.8973261584388547],
              [30.014109002463602, 31.91181633679285, 33.82363325487289,
               35.55908270121682, 37.08994710686285, 37.66843023142387]),
    "clic": ([0.13803363212626746, 0.20595083920424098, 0.3565329804348678,
              0.44812421336677555, 0.6452781966788921],
             [32.02797220728088, 33.503496850025975, 35.643356734018425,
              36.60139856714859, 38.22377621773211]),
    "tecnick": ([0.07895187613822979, 0.15884876748211593,
                 0.3027491158378586, 0.5480239867184374],
                [29.791077346085483, 32.36779127316333,
                 35.07508158995802, 37.64309025702291]),
}
EXPECTED_IMAGES = {"clic": 41, "kodak": 24, "tecnick": 100}
EXPECTED_QPS = (0, 15, 31, 47, 63)


def bd_rate(reference, candidate):
    curves = []
    for rates, psnrs in (reference, candidate):
        order = np.argsort(psnrs)
        x = np.asarray(psnrs)[order]
        y = np.asarray(rates)[order]
        if np.any(np.diff(x) <= 0) or np.any(y <= 0):
            raise ValueError("Invalid rate-distortion curve")
        curves.append((x, y))
    low = max(curve[0][0] for curve in curves)
    high = min(curve[0][-1] for curve in curves)
    if low >= high:
        raise ValueError("Rate-distortion curves do not overlap")
    functions = [PchipInterpolator(x, np.log(y)) for x, y in curves]
    value = 100 * np.expm1(
        (functions[1].integrate(low, high) - functions[0].integrate(low, high))
        / (high - low)
    )
    return float(value), [float(low), float(high)]


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = (row["dataset"], row["image"], int(row["qp"]))
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        if row["max_codec_error"] != 0:
            raise ValueError(f"Invalid codec result: {key}")
    return rows


def score(path):
    grouped = defaultdict(list)
    for row in load_rows(path):
        grouped[(row["dataset"], int(row["qp"]))].append(row)
    report = {"evaluation": str(path), "datasets": {}}
    for dataset in sorted({name for name, _ in grouped}):
        qps = tuple(sorted(qp for name, qp in grouped if name == dataset))
        if qps != EXPECTED_QPS:
            raise ValueError(f"{dataset}: expected QPs {EXPECTED_QPS}, got {qps}")
        images = {row["image"] for qp in qps for row in grouped[(dataset, qp)]}
        if len(images) != EXPECTED_IMAGES[dataset]:
            raise ValueError(f"{dataset}: incomplete image coverage")
        points = {}
        rates, psnrs = [], []
        for qp in qps:
            rows = grouped[(dataset, qp)]
            if {row["image"] for row in rows} != images:
                raise ValueError(f"{dataset}: image coverage differs by QP")
            bpp = float(np.mean([row["bpp"] for row in rows]))
            psnr = float(np.mean([row["psnr"] for row in rows]))
            rates.append(bpp)
            psnrs.append(psnr)
            points[str(qp)] = {"bpp": bpp, "psnr": psnr}
        value, interval = bd_rate(VTM[dataset], (rates, psnrs))
        report["datasets"][dataset] = {
            "images": len(images),
            "points": len(images) * len(qps),
            "bd_rate_vs_vtm_percent": value,
            "psnr_interval": interval,
            "rd_points": points,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(args.evaluation)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
