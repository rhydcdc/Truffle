"""학습 루프. 끊겨도 잃지 않는 것이 이 파일의 목적.

랩탑은 절전·뚜껑·윈도우 업데이트로 끊긴다. 재시작은 부품이 아니라 전제다.
에피소드를 매 스텝 새로 만들기 때문에 **RNG 상태가 곧 데이터 위치**다 —
가중치만 저장하면 조용히 어긋난다. 커리큘럼 단계도 빠지면 soft 로 되돌아간다.

    python -m src.train --config routed
    python -m src.train --config smoke --steps 200 --no-resume
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import (Config, SMOKE, BASELINE_N32, ROUTED_N32, ROUTED_N32_LATE, ROUTED_N32_LATE010,
                     ROUTED_N64, ROUTED_N64_EXT, ROUTED_N64_LR, ROUTED_N64_PW1)
from .data import EpisodeGen
from .model import CacheRouter
from .metrics import loss_fn, evaluate
from .logger import TrainLogger

try:
    from tqdm import tqdm
except Exception:                      # tqdm 이 없어도 학습은 돈다
    tqdm = None


# ---------------------------------------------------------------- 잡동사니
def _hms(s: float) -> str:
    if s != s or s < 0 or s == float("inf"):
        return "--:--:--"
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _med(w) -> float:
    """중앙값. 평균은 첫 스텝의 컴파일·할당 튐에 끌려간다."""
    if not w:
        return float("nan")
    v = sorted(w)
    return v[len(v) // 2]


def _say(bar, msg: str):
    """콘솔 인코딩(cp949 등) 때문에 학습이 죽지 않게 한다."""
    try:
        bar.write(msg) if bar else print(msg, flush=True)
    except UnicodeEncodeError:
        s = msg.encode("ascii", "backslashreplace").decode("ascii")
        bar.write(s) if bar else print(s, flush=True)


def _amp(use_cuda: bool):
    """bf16 오토캐스트. 지수부가 fp32 와 같아 GradScaler 가 필요 없다."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()


def _lr_at(cfg: Config, step: int) -> float:
    """선형 warmup -> cosine. 바닥은 lr 의 10 %.

    코사인 길이는 sched_steps (0 이면 max_steps). 연장 학습에서 sched_steps 를 원래 길이로
    두면 그 지점부터 p 가 1.0 에 클램프돼 LR 이 바닥에 머문다 — 스케줄이 되살아나지 않는다.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    total = cfg.sched_steps or cfg.max_steps
    p = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    p = min(1.0, max(0.0, p))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p)))


# ---------------------------------------------------------------- 시드 / RNG
def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    """넷 다 저장한다. 하나라도 빠지면 같은 에피소드를 다시 보거나 분포가 어긋난다."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def _rng_load(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    cu = s.get("cuda") or []
    if cu and torch.cuda.is_available() and len(cu) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([t.cpu() for t in cu])


# ---------------------------------------------------------------- 원자적 쓰기
def _save_atomic(obj, path: Path):
    """tmp 에 쓰고 os.replace. 저장 중 죽어도 기존 체크포인트는 산다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_atomic(text: str, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- metrics.csv
_BASE = ["step", "wall_s", "loss", "loss_hit", "loss_pos", "loss_size", "lr", "hard"]
_LOSS_KEYS = ["loss", "loss_hit", "loss_pos", "loss_size"]
_P_KEYS = ["p2_key_sep", "p3_recall", "p4_pos_acc", "p4_size_acc"]


def _row(step, wall, lv, parts, lr, hard, metrics) -> dict:
    """학습 손실(그 스텝) + evaluate 가 준 전부. 이름이 겹치면 eval_ 접두사."""
    row = dict(step=step, wall_s=round(wall, 1), loss=lv,
               loss_hit=parts.get("loss_hit"), loss_pos=parts.get("loss_pos"),
               loss_size=parts.get("loss_size"), lr=lr, hard=int(hard))
    for k, v in metrics.items():
        row[("eval_" + k) if k in _BASE else k] = v
    return row


def _csv_append(path: Path, row: dict):
    """헤더는 첫 기록 때 실제 키에서 만들어 한 번만 쓴다."""
    fresh = (not path.exists()) or path.stat().st_size == 0
    if fresh:
        cols = _BASE + [k for k in row if k not in _BASE]
    else:
        with open(path, newline="", encoding="utf-8") as f:
            cols = next(csv.reader(f))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(cols)
        w.writerow([row.get(c, "") for c in cols])


def _csv_history(path: Path) -> list:
    """재시작해도 그래프가 이어지도록 지난 eval 을 되읽는다."""
    if not path.exists():
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if k is None:
                    continue
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            if isinstance(d.get("step"), float):
                out.append(d)
    return out


# ---------------------------------------------------------------- 학습
def train(cfg: Config, resume: bool = True) -> dict:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # 재실행 때 인덕터 캐시 재사용 (체크리스트 §0 가속)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str((run_dir / "inductor").resolve()))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _seed_all(cfg.seed)

    use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
    device = torch.device(cfg.device if use_cuda else "cpu")

    # 생성 순서를 고정한다 — RNG 소비 순서가 같아야 재시작이 맞아떨어진다
    gen = EpisodeGen(cfg, device)
    raw = CacheRouter(cfg).to(device)
    if cfg.grad_checkpoint:                # 기본 꺼짐. 활성값이 8GB 안에 든다
        if hasattr(raw, "set_grad_checkpoint"):
            raw.set_grad_checkpoint(True)
        else:
            _say(None, "경고: grad_checkpoint=True 지만 모델에 훅이 없다. 그냥 간다")
    # compile 은 감싸기만 한다. state_dict 는 항상 raw 에서 — 접두사가 안 붙는다
    model = torch.compile(raw) if cfg.compile else raw

    opt = torch.optim.AdamW(raw.parameters(), lr=cfg.lr, betas=tuple(cfg.betas),
                            weight_decay=cfg.weight_decay, fused=use_cuda)

    ck_path, gate_path = run_dir / "last.pt", run_dir / "gate.pt"
    global_step, hard_phase, last_metrics, wall_prev = 0, False, {}, 0.0

    # 분기는 이거 하나 — last.pt 가 있으면 이어서, 없으면 처음부터
    if resume and ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)  # RNG·지표가 텐서가 아니다
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])   # Adam 모멘트. 빠지면 재시작 직후 손실이 튄다
        _rng_load(ck["rng"])               # 데이터 위치 복원
        global_step = int(ck["global_step"])
        hard_phase = bool(ck["hard"])      # 안 하면 soft 로 되돌아간다
        last_metrics = dict(ck.get("metrics") or {})
        wall_prev = float(ck.get("wall_s", 0.0))
        _say(None, f"재시작: step {global_step}/{cfg.max_steps} | "
                   f"{'hard' if hard_phase else 'soft'} | 누적 {_hms(wall_prev)}")
    elif ck_path.exists():
        _say(None, "--no-resume: 기존 last.pt 를 무시하고 처음부터")

    csv_path = run_dir / "metrics.csv"
    hist = [h for h in _csv_history(csv_path) if h["step"] <= global_step]

    # train.log — 한 줄씩 이어쓰고 매 줄 flush 한다. tqdm 과 겹치지 않게 echo 는 끈다.
    lg = TrainLogger(cfg.run_dir, echo=(tqdm is None))
    lg.start(cfg, *raw.n_params(), resumed_from=global_step)

    hard_at = int(cfg.hard_switch_frac * cfg.max_steps)
    drop_on = cfg.group_dropout_p > 0
    first_step = global_step
    window = deque(maxlen=100)         # CUDA 동기화 타이머, 최근 100 스텝
    t0 = time.perf_counter() - wall_prev
    lv, lr, parts = float("nan"), _lr_at(cfg, global_step), {}
    interrupted = False

    def snapshot() -> dict:
        """하나라도 빠지면 재시작이 깨진다 — 목록은 체크리스트 §체크포인트와 재시작."""
        return dict(
            global_step=global_step, sched_step=global_step,   # LR 은 스텝의 함수다
            model=raw.state_dict(), optim=opt.state_dict(),
            rng=_rng_state(), hard=hard_phase, metrics=last_metrics,
            wall_s=time.perf_counter() - t0, lr=lr, cfg=cfg.dict(),
        )

    bar = tqdm(total=cfg.max_steps, initial=global_step, dynamic_ncols=True,
               desc=run_dir.name, unit="st") if tqdm else None
    model.train()

    try:
        while global_step < cfg.max_steps:
            if use_cuda:
                torch.cuda.synchronize()
            tick = time.perf_counter()

            if not hard_phase and global_step >= hard_at:
                hard_phase = True
                _say(bar, f"[step {global_step}] 커리큘럼 전환 soft -> hard")
                lg.switch(global_step)

            lr = _lr_at(cfg, global_step)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = gen.batch(cfg.batch_size)

            gdm = None
            if drop_on:
                # True = 지운다. 드롭은 질의별로 건다 (B,Q,N).
                # 에피소드 단위로 걸면 Q = 조합 수일 때 모든 방이 어느 질의의
                # 정답이라 마스크가 통째로 비어 드롭이 안 걸린다.
                # 질의 q 입장에선 자기 정답 방만 지키면 되고 나머지는 지워도 된다 —
                # 그래야 "그 방만 있어도 위치·크기를 복원하는가" 가 강제된다.
                hit = batch["hit"]                               # (B,Q,N) bool
                gdm = (torch.rand(hit.shape, device=device) < cfg.group_dropout_p) & ~hit
                # 살아남은 후보가 1라운드 k 보다 적으면 그 질의는 드롭을 걷는다.
                # 뒤 라운드에서 후보가 바닥나는 건 모델이 막는다 — 가려진 방을 뽑으면 valid=False.
                # 총 읽기(sum(ks)) 로 재면 최대 라운드가 방 전체를 덮을 때(k x R = N)
                # 방 하나만 드롭돼도 걷혀서 드롭아웃이 통째로 꺼진다.
                thin = (~gdm).sum(-1) < cfg.ks[0]                # (B,Q)
                gdm = gdm & ~thin[..., None]

            with _amp(use_cuda):
                out = model(batch["obs"], batch["qry"], hard=hard_phase, group_drop_mask=gdm)
                pw = torch.tensor(cfg.pos_weight, device=batch["hit"].device) if cfg.pos_weight > 0 else None
                loss, parts = loss_fn(out, batch, pw)   # None 이면 loss_fn 이 배치에서 자동 계산 (종전과 동일)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
            opt.step()

            global_step += 1
            if use_cuda:
                torch.cuda.synchronize()
            if global_step - first_step > 5:        # 초반 컴파일·할당 튐은 버린다
                window.append((time.perf_counter() - tick) * 1e3)
            lv = float(loss.detach())
            ms = _med(window)
            left = (cfg.max_steps - global_step) * (ms if ms == ms else 0.0) / 1e3

            msg = f"{ms:.1f} ms/step" if ms == ms else "-- ms/step"
            if bar:
                bar.update(1)
                bar.set_postfix_str(f"loss {lv:.4f} | {msg}")
            if global_step % cfg.log_every == 0:
                lg.step(global_step, lv, ms if ms == ms else 0.0, lr, hard_phase)

            # 평가는 학습 스텝 바깥에서. 잰 ms/step 이 오염되지 않는다
            if global_step % cfg.eval_every == 0 or global_step == cfg.max_steps:
                with _amp(use_cuda):
                    last_metrics = dict(evaluate(model, gen, cfg, cfg.eval_batches))
                wall = time.perf_counter() - t0
                row = _row(global_step, wall, lv, parts, lr, hard_phase, last_metrics)
                _csv_append(csv_path, row)
                hist.append(row)
                lg.eval(global_step, last_metrics)
                # eval 마다 모델을 따로 남긴다. 후반에 성능이 떨어져도 최고점을 고를 수 있다.
                # 재시작용이 아니라 가중치·설정만 — 옵티마이저·RNG 는 last.pt 에 있다
                step_path = run_dir / f"step_{global_step:05d}.pt"
                _save_atomic(dict(global_step=global_step, model=raw.state_dict(),
                                  cfg=cfg.dict(), metrics=last_metrics), step_path)

            if global_step % cfg.ckpt_every == 0:
                _save_atomic(snapshot(), ck_path)
                lg.ckpt(global_step, str(ck_path))

    except KeyboardInterrupt:
        interrupted = True
    finally:
        if bar:
            bar.close()
        _save_atomic(snapshot(), ck_path)    # 중단이든 완주든 저장부터. 로그는 그 다음

    if interrupted:
        _say(None, "중단 - 체크포인트를 저장하고 나간다")
        lg.interrupted(global_step)
    elif global_step >= cfg.max_steps:
        _save_atomic(snapshot(), gate_path)  # 완주 시 1회만. 그 외에는 남기지 않는다

    wall, ms = time.perf_counter() - t0, _med(window)
    _say(None, f"끝 | step {global_step} | {ms:.1f} ms/step | 총 {_hms(wall)} | {ck_path}")
    lg.done(global_step, ms if ms == ms else 0.0, last_metrics)
    return dict(steps=global_step, ms_per_step=ms, wall_s=wall, hard=hard_phase,
                loss=lv, metrics=last_metrics, interrupted=interrupted,
                run_dir=str(run_dir))


# ---------------------------------------------------------------- CLI
CONFIGS = {"smoke": SMOKE, "baseline": BASELINE_N32, "routed": ROUTED_N32, "routed_late": ROUTED_N32_LATE,
           "routed_late010": ROUTED_N32_LATE010, "routed_n64": ROUTED_N64,
           "routed_n64_ext": ROUTED_N64_EXT, "routed_n64_lr": ROUTED_N64_LR,
           "routed_n64_pw1": ROUTED_N64_PW1}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m src.train")
    ap.add_argument("--config", choices=list(CONFIGS), default="smoke")
    ap.add_argument("--no-resume", action="store_true", help="last.pt 를 무시하고 처음부터")
    ap.add_argument("--steps", type=int, default=None, help="max_steps 덮어쓰기")
    a = ap.parse_args()

    cfg = CONFIGS[a.config]
    if a.steps:
        cfg = replace(cfg, max_steps=a.steps)
    out = train(cfg, resume=not a.no_resume)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
