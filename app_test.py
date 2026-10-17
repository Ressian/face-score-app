# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_CPU_ONLY"] = "1"
os.environ["TORCH_DYNAMO_ENABLED"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MPLBACKEND"] = "Agg"

if sys.stdout is None or sys.stderr is None:
    import io

    class _NullWriter(io.TextIOBase):
        def write(self, s):
            return len(s)

        def flush(self):
            return None

    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_cli(args):
    from app.engine import ScoreEngine, format_result

    eng = ScoreEngine(tta=args.tta, threads=args.threads)
    collected = []
    try:
        for path in args.image:
            result = eng.predict(path, track=args.track, gender=args.gender)
            payload = {k: v for k, v in result.items()
                       if k not in ("crop", "crop_model", "landmarks", "geo")}
            payload["image"] = path
            collected.append(payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print("=" * 60)
                print(path)
                print(format_result(result))
    finally:
        eng.close()
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="颜值打分 v2：单图测试 / GUI")
    ap.add_argument("--image", action="append", default=[], help="图片路径，可重复")
    ap.add_argument("--track", default="scut",
                    choices=["scut", "meb", "crowd", "ensemble"])
    ap.add_argument("--gender", default="generic",
                    choices=["female", "male", "generic"],
                    help="SCUT 主模型性别权重")
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out-json", default="", help="把结果写入 JSON 文件（窗口版 exe 自检用）")
    ap.add_argument("--cli", action="store_true", help="强制命令行模式")
    args = ap.parse_args()

    if args.cli or args.image:
        run_cli(args)
        return
    from app.ui import run
    run()


if __name__ == "__main__":
    main()

