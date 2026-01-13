import argparse
import os
import zipfile


def iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-hub-cache", default=os.environ.get("HF_HUB_CACHE", ""))
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd(), "mocr_offline.zip"),
        help="Output zip path",
    )
    args = parser.parse_args()

    if not args.hf_hub_cache:
        raise SystemExit("HF_HUB_CACHE is empty. Pass --hf-hub-cache or set env HF_HUB_CACHE")

    model_dir_name = "models--kha-white--manga-ocr-base"
    src_root = os.path.join(args.hf_hub_cache, model_dir_name)

    if not os.path.isdir(src_root):
        raise SystemExit(f"Model cache not found: {src_root}")

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prefix = model_dir_name + "/"

    count = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp in iter_files(src_root):
            rel = os.path.relpath(fp, src_root).replace(os.sep, "/")
            arcname = prefix + rel
            zf.write(fp, arcname)
            count += 1

    print(f"OK: wrote {count} files -> {out_path}")


if __name__ == "__main__":
    main()
