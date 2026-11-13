#!/usr/bin/env python3
"""训练看门狗：MPS/数值熔断或崩溃后自动 --resume，高损失熔断则停。

用法（在项目根目录）：

  python scripts/watch_train.py \\
    --log checkpoints/train_hf_mid200_v2.log \\
    --pid-prefix checkpoints/train_hf_mid200_v2 \\
    -- train.py --resume --tier mid200 --device mps ... --checkpoint ...

`--` 后面的参数原样传给训练脚本（默认 scripts 同级的 train.py）。
看门狗本身不改模型超参；收到 SIGINT / SIGTERM 时会停掉子进程并退出，不再拉起。

退出码约定（与 train.py 保持一致）：
  0  正常跑完
  1  用法 / 文件错误（不自动续训）
  2  MPS/数值熔断（loss≈0 或 NaN/Inf）——等待数秒后 --resume
  3  高损失熔断（loss 持续 >20）——不自动续训
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 与 train.py 中的常量保持一致
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FUSE_NUMERIC = 2
EXIT_FUSE_HIGH = 3

# 视为「用户要停」的信号：不自动续训
STOP_SIGNALS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}

# 进程被这几个信号打死，按崩溃处理，可以自动续训
CRASH_SIGNALS = {
    signal.SIGSEGV,
    signal.SIGABRT,
    signal.SIGBUS,
    signal.SIGILL,
    signal.SIGFPE,
    signal.SIGKILL,
    signal.SIGSYS,
    signal.SIGTRAP,
}

_stop_requested = False
_stop_signal: Optional[int] = None


def _now() -> str:
    # 机器时区已是 Asia/Shanghai
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stdout_is_same_file(path: Path) -> bool:
    """stdout 已经重定向到这份日志时，避免再写一遍造成重复行。"""
    try:
        if not sys.stdout.isatty() and path.is_file():
            return os.fstat(sys.stdout.fileno()).st_ino == path.stat().st_ino
    except OSError:
        return False
    return False


def _log(log_path: Path, msg: str) -> None:
    line = f"[看门狗 {_now()}] {msg}"
    print(line, flush=True)
    if _stdout_is_same_file(log_path):
        return
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except OSError as e:
        print(f"[看门狗] 写日志失败 {log_path}: {e}", file=sys.stderr, flush=True)


def _handle_stop(signum: int, _frame: Any) -> None:
    global _stop_requested, _stop_signal
    _stop_requested = True
    _stop_signal = signum


def _write_pid(path: Path, pid: int) -> None:
    path.write_text(f"{pid}\n", encoding="utf-8")


def _arg_value(argv: list, flag: str) -> Optional[str]:
    """从 argv 里取 --flag VALUE 或 --flag=VALUE。"""
    prefix = flag + "="
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(prefix):
            return a[len(prefix) :]
    return None


def _has_flag(argv: list, flag: str) -> bool:
    return flag in argv or any(a.startswith(flag + "=") for a in argv)


def _ensure_resume(argv: list) -> list:
    if _has_flag(argv, "--resume"):
        return list(argv)
    return list(argv) + ["--resume"]


def inspect_ckpt(path: Path) -> Tuple[int, Optional[float], Optional[float]]:
    """读主 ckpt 的 iter 与 train/val loss；失败则抛异常。"""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    try:
        it = int(ckpt.get("iter", -1))
        metrics = ckpt.get("metrics") or {}
        train = metrics.get("train")
        val = metrics.get("val")
        train_f = float(train) if train is not None else None
        val_f = float(val) if val is not None else None
    finally:
        del ckpt
    return it, train_f, val_f


def ckpt_is_sane(
    path: Path,
    *,
    eps: float = 1e-6,
) -> Tuple[bool, str, Optional[int], Optional[float], Optional[float]]:
    """主 ckpt 必须存在，iter 有效，loss 有限且不接近 0。"""
    if not path.is_file():
        return False, f"找不到主 checkpoint: {path}", None, None, None
    try:
        it, train, val = inspect_ckpt(path)
    except Exception as e:  # noqa: BLE001 — 读坏文件时要停，不要续训
        return False, f"无法读取 {path}: {e}", None, None, None
    if it < 0:
        return False, f"{path} 没有有效的 iter（得到 {it}）", it, train, val
    losses = [x for x in (train, val) if x is not None]
    if not losses:
        return False, f"{path} 没有 train/val 指标，拒绝续训", it, train, val
    for name, x in (("train", train), ("val", val)):
        if x is None:
            continue
        if not math.isfinite(x):
            return False, f"{path} 的 {name}={x!r} 非有限值", it, train, val
        if abs(x) < eps:
            return False, f"{path} 的 {name}={x!r} 接近 0（疑似损坏）", it, train, val
    return True, "ok", it, train, val


def classify_exit(rc: int) -> str:
    """把子进程退出码分类成: ok / numeric / high / error / crash / stop。"""
    if rc == EXIT_OK:
        return "ok"
    if rc == EXIT_FUSE_NUMERIC:
        return "numeric"
    if rc == EXIT_FUSE_HIGH:
        return "high"
    if rc == EXIT_ERROR:
        return "error"
    if rc < 0:
        sig = -rc
        if sig in STOP_SIGNALS:
            return "stop"
        return "crash"
    # 其它正数退出码：按意外崩溃处理，允许续训
    return "crash"


def describe_exit(rc: int, kind: str) -> str:
    if rc < 0:
        sig = -rc
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        extra = {
            "stop": "视为手动停止，不自动续训",
            "crash": "视为崩溃，可自动续训",
        }.get(kind, "")
        return f"被信号终止 {name}（退出状态 {rc}）{('，' + extra) if extra else ''}"
    labels = {
        "ok": f"正常结束（退出码 {rc}）",
        "numeric": f"MPS/数值熔断（退出码 {rc}）",
        "high": f"高损失熔断（退出码 {rc}），不自动续训",
        "error": f"训练脚本报错（退出码 {rc}），不自动续训",
        "crash": f"意外退出码 {rc}，按崩溃处理",
    }
    return labels.get(kind, f"退出码 {rc}")


def last_healthy_iter(log_path: Path, start_offset: int, fallback: int) -> int:
    """从本次运行新写入的日志里找最后一次健康的 iter（loss 有限、非 0、不超过 20）。"""
    try:
        data = log_path.read_bytes()[start_offset:].decode("utf-8", errors="replace")
    except OSError:
        return fallback
    last = fallback
    for line in data.splitlines():
        if "熔断" in line or "[看门狗" in line:
            continue
        m = re.search(r"iter\s+(\d+)\s+\|\s+loss\s+([0-9.eE+-]+)", line)
        if m:
            it, loss = int(m.group(1)), float(m.group(2))
            if math.isfinite(loss) and abs(loss) >= 1e-6 and loss <= 20.0:
                last = max(last, it)
            continue
        m = re.search(
            r"iter\s+(\d+)\s+\|\s+train\s+([0-9.eE+-]+)\s+\|\s+val\s+([0-9.eE+-]+)",
            line,
        )
        if m:
            it = int(m.group(1))
            tr, va = float(m.group(2)), float(m.group(3))
            if all(math.isfinite(x) and abs(x) >= 1e-6 and x <= 20.0 for x in (tr, va)):
                last = max(last, it)
    return last


def terminate_child(proc: subprocess.Popen, log_path: Path, grace: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    _log(log_path, f"向训练进程 PID {proc.pid} 发送 SIGTERM")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.4)
    _log(log_path, f"训练进程 PID {proc.pid} 未退出，发送 SIGKILL")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    proc.wait(timeout=10)


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练看门狗：MPS 熔断/崩溃后自动续训",
    )
    p.add_argument(
        "--log",
        required=True,
        help="训练日志路径（追加写入；重启记录也写到这里）",
    )
    p.add_argument(
        "--pid-prefix",
        required=True,
        help="pid/cmd 文件前缀，例如 checkpoints/train_hf_mid200_v2",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=25.0,
        help="MPS 熔断或崩溃后，等待多少秒再续训（默认 25）",
    )
    p.add_argument(
        "--max-consecutive",
        type=int,
        default=3,
        help="连续无进展的自动重启次数上限（默认 3）",
    )
    p.add_argument(
        "--max-total-restarts",
        type=int,
        default=10,
        help="本轮战役自动重启总次数上限（默认 10，不含第一次启动）",
    )
    p.add_argument(
        "--progress-iters",
        type=int,
        default=500,
        help="视为「有进展」的最少健康步数，或主 ckpt iter 往前走（默认 500）",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="用来启动 train.py 的 Python（默认：当前解释器）",
    )
    p.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="传给训练脚本的参数；请用 -- 分隔，例如 -- train.py --resume ...",
    )
    return p.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = Path(args.log)
    prefix = Path(args.pid_prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    train_argv = list(args.train_argv)
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        print("[错误] 请在 -- 后面给出训练命令，例如: -- train.py --resume ...", file=sys.stderr)
        return EXIT_ERROR

    # 第一项可以是 train.py 路径，也可以直接是参数（默认 ./train.py）
    if train_argv[0].endswith(".py") or train_argv[0] == "train.py":
        train_script = train_argv[0]
        train_args = train_argv[1:]
    else:
        train_script = "train.py"
        train_args = train_argv

    ckpt_s = _arg_value(train_args, "--checkpoint")
    if not ckpt_s:
        print("[错误] 训练参数里需要 --checkpoint，看门狗要靠它做续训前检查。", file=sys.stderr)
        return EXIT_ERROR
    ckpt_path = Path(ckpt_s)

    # 第一次启动就要求 --resume：避免看门狗误开「从头训」把好权重盖掉
    if ckpt_path.is_file() and not _has_flag(train_args, "--resume"):
        _log(
            log_path,
            f"主 ckpt 已存在（{ckpt_path}），但命令没有 --resume。"
            "为防止覆盖好权重，看门狗拒绝启动。请加上 --resume。",
        )
        return EXIT_ERROR

    watchdog_pid_path = Path(str(prefix) + ".watchdog.pid")
    train_pid_path = Path(str(prefix) + ".pid")
    _write_pid(watchdog_pid_path, os.getpid())

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    # 后台 nohup 通常忽略 SIGHUP；若有人显式发给看门狗，也按停止处理
    try:
        signal.signal(signal.SIGHUP, _handle_stop)
    except OSError:
        pass

    total_restarts = 0
    consecutive_no_progress = 0
    attempt = 0
    project_root = Path.cwd()

    _log(
        log_path,
        f"看门狗启动 PID={os.getpid()}，日志={log_path}，主 ckpt={ckpt_path}，"
        f"最多连续无进展重启 {args.max_consecutive} 次、"
        f"本轮总共重启 {args.max_total_restarts} 次，"
        f"熔断后等待 {args.settle_seconds:g} 秒。",
    )

    while True:
        if _stop_requested:
            name = (
                signal.Signals(_stop_signal).name
                if _stop_signal is not None
                else "signal"
            )
            _log(log_path, f"收到 {name}，停止且不再拉起训练。")
            return EXIT_OK

        ok, reason, it, train, val = ckpt_is_sane(ckpt_path)
        if not ok:
            _log(log_path, f"续训前检查未通过：{reason}。停止，不自动重启。")
            return EXIT_ERROR

        attempt += 1
        is_restart = attempt > 1
        metrics_s = (
            f"iter={it}, train={train:.4f}, val={val:.4f}"
            if train is not None and val is not None
            else f"iter={it}"
        )
        if is_restart:
            _log(
                log_path,
                f"自动重启第 {total_restarts}/{args.max_total_restarts} 次"
                f"（连续无进展 {consecutive_no_progress}/{args.max_consecutive}）："
                f"从 {metrics_s} 续训。",
            )
        else:
            _log(log_path, f"第一次启动训练：续训 {metrics_s}。")

        cmd = [args.python, "-u", train_script] + _ensure_resume(train_args)
        start_offset = log_path.stat().st_size if log_path.is_file() else 0
        start_iter = it if it is not None else 0

        # 子进程独立 session：看门狗被杀时我们可以给整个训练进程组发信号
        # 训练输出追加进同一份日志（与看门狗记录在一起）
        log_f = log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            log_f.close()
            _log(log_path, f"无法启动训练进程: {e}")
            return EXIT_ERROR
        _write_pid(train_pid_path, proc.pid)
        _log(log_path, f"训练进程已启动 PID={proc.pid}：{' '.join(cmd)}")

        rc: Optional[int] = None
        while True:
            if _stop_requested:
                terminate_child(proc, log_path)
                rc = proc.poll()
                name = (
                    signal.Signals(_stop_signal).name
                    if _stop_signal is not None
                    else "signal"
                )
                _log(
                    log_path,
                    f"收到 {name}，已停止训练进程（退出码 {rc!r}），不再拉起。",
                )
                log_f.close()
                return EXIT_OK
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.5)
        log_f.close()

        kind = classify_exit(rc)
        _log(log_path, f"训练进程 PID={proc.pid} 退出：{describe_exit(rc, kind)}。")

        if kind == "ok":
            _log(log_path, "训练正常结束，看门狗退出。")
            return EXIT_OK
        if kind == "high":
            _log(log_path, "高损失熔断，不自动续训。请人工检查日志和权重后再决定。")
            return EXIT_FUSE_HIGH
        if kind == "error":
            _log(log_path, "训练脚本以错误码 1 退出（多半是参数或文件问题），不自动续训。")
            return EXIT_ERROR
        if kind == "stop":
            _log(log_path, "训练进程被中断信号结束，视为手动停止，不自动续训。")
            return EXIT_OK
        # numeric 或 crash：可以尝试自动续训
        if total_restarts >= args.max_total_restarts:
            _log(
                log_path,
                f"已达到本轮自动重启上限 {args.max_total_restarts} 次，停止。"
                "若 GPU 仍异常，请先排查后再手动续训。",
            )
            return EXIT_FUSE_NUMERIC if kind == "numeric" else EXIT_ERROR

        # 判断这次有没有进展：主 ckpt iter 往前，或健康步进了若干步
        ok_after, after_reason, it_after, train_after, val_after = ckpt_is_sane(ckpt_path)
        if not ok_after:
            _log(log_path, f"重启前复查主 ckpt 失败：{after_reason}。停止，以免读到坏文件。")
            return EXIT_ERROR
        healthy = last_healthy_iter(log_path, start_offset, start_iter)
        advanced_ckpt = it_after is not None and it_after > start_iter
        advanced_steps = (healthy - start_iter) >= int(args.progress_iters)
        if advanced_ckpt or advanced_steps:
            consecutive_no_progress = 0
            why = []
            if advanced_ckpt:
                why.append(f"主 ckpt iter {start_iter} → {it_after}")
            if advanced_steps:
                why.append(f"健康步进到 iter {healthy}（+{healthy - start_iter}）")
            _log(log_path, "本轮有进展（" + "；".join(why) + "），连续无进展计数清零。")
        else:
            consecutive_no_progress += 1
            _log(
                log_path,
                f"本轮没有明显进展（ckpt iter 仍为 {it_after}，"
                f"健康 iter={healthy}）。连续无进展 {consecutive_no_progress}/"
                f"{args.max_consecutive}。",
            )
            if consecutive_no_progress >= args.max_consecutive:
                _log(
                    log_path,
                    f"连续 {consecutive_no_progress} 次重启仍无进展，停止自动续训。"
                    "可能是 GPU 已卡死，请稍后再手动启动。",
                )
                return EXIT_FUSE_NUMERIC if kind == "numeric" else EXIT_ERROR

        total_restarts += 1
        wait = max(0.0, float(args.settle_seconds))
        _log(log_path, f"等待 {wait:g} 秒，让 MPS/GPU 状态平复后再续训…")
        waited = 0.0
        while waited < wait:
            if _stop_requested:
                name = (
                    signal.Signals(_stop_signal).name
                    if _stop_signal is not None
                    else "signal"
                )
                _log(log_path, f"等待期间收到 {name}，停止且不再拉起。")
                return EXIT_OK
            slice_s = min(0.5, wait - waited)
            time.sleep(slice_s)
            waited += slice_s

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
