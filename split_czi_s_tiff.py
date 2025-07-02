"""
Split every scene in one or many .czi files into separate OME-TIFFs.

Usage
-----
    # one file
    python split_czi_folder_to_ometiff.py /path/to/slide.czi

    # all .czi in a folder (non-recursive)
    python split_czi_folder_to_ometiff.py /path/to/czi_folder

    # choose a destination folder for the exported TIFFs
    python split_czi_folder_to_ometiff.py /path/to/czi_folder --outdir /tmp/out

Dependencies
------------
    pip install pylibCZIrw==5.0.0 tifffile numpy
"""

from pathlib import Path
import argparse
import numpy as np
from tifffile import imwrite
from pylibCZIrw import czi as pyczi

_AXES = "TZCYX"  # order expected by tifffile


def _range_from_bbox(bbox: dict, key: str):
    start, stop = bbox.get(key, (0, 1))
    return range(start, stop)


def _channel_count(reader) -> int:
    if hasattr(reader, "dim_sizes"):       # ≥ 5.1
        return reader.dim_sizes.get("C", 1)
    if hasattr(reader, "dims"):            # 5.0.x
        return reader.dims.get("C", 1)
    start, stop = reader.total_bounding_box.get("C", (0, 1))
    return stop - start or 1


def read_full_scene(reader, scene_idx: int) -> np.ndarray:
    bbox = reader.total_bounding_box
    t_rng = _range_from_bbox(bbox, "T")
    z_rng = _range_from_bbox(bbox, "Z")
    c_rng = _range_from_bbox(bbox, "C")

    y0, x0, h, w = reader.scenes_bounding_rectangle[scene_idx]

    sample = reader.read(
        plane={"T": t_rng[0] if t_rng else 0, "Z": z_rng[0] if z_rng else 0},
        scene=scene_idx,
        roi=(x0, y0, 1, 1),
    )
    dtype = sample.dtype
    if sample.ndim == 3 and sample.shape[-1] in (3, 4) and len(c_rng) <= 1:
        interleaved = True
        n_channels = sample.shape[-1]
    else:
        interleaved = False
        n_channels = len(c_rng) or 1

    stack = np.empty((len(t_rng), len(z_rng), n_channels, h, w), dtype=dtype)

    for t_i, T in enumerate(t_rng):
        for z_i, Z in enumerate(z_rng):
            if interleaved:
                img = reader.read(
                    plane={"T": T, "Z": Z},
                    scene=scene_idx,
                    roi=(x0, y0, w, h),
                )                       # h×w×nC
                stack[t_i, z_i] = img.transpose(2, 0, 1)
            else:
                for c_i, C in enumerate(c_rng):
                    img = reader.read(
                        plane={"T": T, "Z": Z, "C": C},
                        scene=scene_idx,
                        roi=(x0, y0, w, h),
                    )                   # h×w
                    stack[t_i, z_i, c_i] = img
    return stack


def write_ome_tiff(arr, out_path: Path, pixel_size=None, channel_names=None):
    meta = {"axes": _AXES}
    if channel_names and len(channel_names) >= arr.shape[2]:
        meta["Channel"] = [
            {"Name": n} for n in channel_names[: arr.shape[2]]
        ]
    if pixel_size and all(pixel_size):
        meta.update(
            dict(
                PhysicalSizeX=pixel_size[0],
                PhysicalSizeY=pixel_size[1],
                PhysicalSizeZ=pixel_size[2],
            )
        )
    imwrite(out_path, arr, ome=True, metadata=meta)  # BigTIFF auto


def split_czi_file(in_file: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    print(f"\n=== {in_file.name} ===")

    with pyczi.open_czi(str(in_file)) as doc:        # str() for old wheels
        scenes = doc.scenes_bounding_rectangle
        if not scenes:
            print("  No scenes found – skipped.")
            return

        md = doc.metadata
        scales = md.get("Scaling", {}).get("Items", {}).get("Distance", {})
        pixel_size = (scales.get("X"), scales.get("Y"), scales.get("Z"))

        for s in scenes:
            data = read_full_scene(doc, s)
            channel_names = [f"Ch{c}" for c in range(data.shape[2])]
            out_file = out_dir / f"{in_file.stem}_scene{s}.ome.tiff"
            write_ome_tiff(data, out_file, pixel_size, channel_names)
            print(f"  scene {s} → {out_file.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path,
                    help="*.czi file OR folder containing *.czi files")
    ap.add_argument("--outdir", type=Path,
                    help="destination folder (default: alongside each CZI)")
    args = ap.parse_args()

    if args.input.is_dir():
        czi_files = sorted(p for p in args.input.iterdir()
                           if p.suffix.lower() == ".czi")
        if not czi_files:
            print("No .czi files found in folder.")
            return
        for f in czi_files:
            target_dir = args.outdir or f.with_suffix("")
            split_czi_file(f, target_dir)
    else:
        if args.input.suffix.lower() != ".czi":
            print("Input must be a .czi file or a folder containing .czi files.")
            return
        target_dir = args.outdir or args.input.with_suffix("")
        split_czi_file(args.input, target_dir)


if __name__ == "__main__":
    main()
