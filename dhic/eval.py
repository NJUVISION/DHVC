"""Evaluate the codec on an image directory."""
import argparse
import json
from pathlib import Path

from dhic import Codec
from dhic.evaluation import evaluate_image
from dhic.paths import EXPECTED_IMAGES, image_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("weights/model.pth"))
    parser.add_argument("--dataset", choices=tuple(EXPECTED_IMAGES), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 15, 31, 47, 63])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    paths = image_files(args.root)
    expected = EXPECTED_IMAGES[args.dataset]
    if len(paths) != expected:
        parser.error(f"Expected {expected} images, found {len(paths)}")
    tasks = [(path, qp) for qp in args.qps for path in paths]
    tasks = [task for index, task in enumerate(tasks)
             if index % args.num_shards == args.shard_id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (row["image"], int(row["qp"]))
            for row in map(json.loads, args.output.read_text().splitlines())
        }

    codec = Codec(str(args.model), device=args.device)
    with args.output.open("a") as output:
        for index, (path, qp) in enumerate(tasks, 1):
            if (str(path), qp) in completed:
                continue
            row = evaluate_image(codec, path, qp)
            row["dataset"] = args.dataset
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps({
                "progress": f"{index}/{len(tasks)}", "dataset": args.dataset,
                "qp": qp, "bpp": row["bpp"], "psnr": row["psnr"],
            }), flush=True)


if __name__ == "__main__":
    main()
