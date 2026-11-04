import os
import sys

# cuDNN 9 不在默认动态库搜索路径。LD_LIBRARY_PATH 必须在进程启动前生效，
# 所以这里改完环境后重跑一次自身（此时 torch 还没导入）。
_CUDNN_LIB = "/root/rivermind-data/cudnn9/nvidia/cudnn/lib"
_LD_PATH = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_CUDNN_LIB) and _CUDNN_LIB not in _LD_PATH.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB + (f":{_LD_PATH}" if _LD_PATH else "")
    os.execv(sys.executable, [sys.executable, *sys.argv])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _ensure_tty_size(cols=120, rows=40):
    """Jupyter/网页终端常把 pty 尺寸报成 0x0；tqdm 会算出 nrows=-1 而不渲染任何进度条。"""
    try:
        import fcntl
        import struct
        import termios

        for fd in (1, 2):
            try:
                c, r = os.get_terminal_size(fd)
            except OSError:
                continue
            if c >= 2 and r >= 2:
                return  # 终端尺寸正常
            try:
                fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            except OSError:
                pass
        os.environ.setdefault("COLUMNS", str(cols))
        os.environ.setdefault("LINES", str(rows))
    except Exception:
        pass


_ensure_tty_size()

# 二级关系训练默认配置（环境变量仍可覆盖）；`python train.py v2` 直接切 V2。
# 想跑原来的纯一级训练：`YOLOVO_RELATE_DATA= python train.py`。
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLOVO_RELATE_DATA", os.path.join(_ROOT, "RA-4M/data_proxy50k.yaml"))
os.environ.setdefault("YOLOVO_PRIMARY", "/root/rivermind-fs/.Trash-0/files/code/best.pt")
os.environ.setdefault("YOLOVO_RELATE_INPUTS", "P4")
os.environ.setdefault("YOLOVO_EPOCHS", "100")

_field = (sys.argv[1].lower() if len(sys.argv) > 1 else os.environ.get("YOLOVO_FIELD", "glp"))
if _field == "v1":
    _field = "glp"
os.environ["YOLOVO_FIELD"] = _field
os.environ.setdefault("YOLOVO_NAME",
                      "glp_proxy50k_v2_e100" if _field == "v2" else "glp_proxy50k_v1_e100")

from ultralytics import YOLOVO
# 创建 YOLOVO 模型
if __name__ == "__main__" :
    # 关系二级训练（GLP）：设置 YOLOVO_RELATE_DATA 后启用；默认行为保持不变。
    rel_data = os.environ.get("YOLOVO_RELATE_DATA")
    if rel_data:
        model = YOLOVO(os.environ.get("YOLOVO_PRIMARY", "yolo11n.pt"))
        model.rel.train(
            inputs=os.environ.get("YOLOVO_RELATE_INPUTS", "P4").split(","),
            data=rel_data,
            field=os.environ.get("YOLOVO_FIELD", "glp"),
            epochs=int(os.environ.get("YOLOVO_EPOCHS", "1")),
            imgsz=640,
            # batch 32 峰值显存约 10–11 GiB；卡上还有别人的大进程时用 YOLOVO_BATCH=16 降级。
            batch=int(os.environ.get("YOLOVO_BATCH", "32")),
            workers=8,
            # 图像缓存进 RAM：proxy50k 原图约 91 GiB（Ultralytics 按 imgsz 缩放估算，实际存原图）。
            # 不想要缓存时：YOLOVO_CACHE=0 python train.py
            cache=os.environ.get("YOLOVO_CACHE", "1").lower() not in ("0", "false", ""),
            lr0=1e-3,
            project="runs/test",
            name=os.environ.get("YOLOVO_NAME", "glp-relate"),
            exist_ok=True,
        )
    else:
        model = YOLOVO("yolovo11n.yaml")
        # model = YOLOVO('yolovo11n.pt')
        # model.load("yolovo11n.pt")

        # 训练配置
        model.train(
            data="/root/datasets/M3FD_split/data.yaml",
            epochs=100,
            cache=True,
            imgsz=640,
            batch=16,
            workers=8,
            lr0=5e-4,
            # lrf=0.01,
            # device = '0,
            # optimize='AdamW',
            project="runs/test",
            name="yolovo-train-1",
            close_mosaic=10,
            # exist_ok=True,
            resume=False,
            # fraction=0.4,
            yolovo_debug=True,
        )
