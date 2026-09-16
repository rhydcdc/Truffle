"""5단계 측정 — 예산-품질 곡선과 A2(RoPE) 판정.

    python measure.py [runs/routed_n32_r2k4]
"""
import sys
from dataclasses import fields
from pathlib import Path

import torch

from src.config import Config, ROUTED_N32
from src.data import EpisodeGen
from src.model import CacheRouter
from src import metrics as M

BATCHES = 8


def load(run_dir: Path, dev):
    """run_dir 이면 gate.pt -> last.pt 순. .pt 파일을 직접 주면 그걸 쓴다 (step_XXXXX.pt)."""
    if run_dir.is_file():
        ck = run_dir
    else:
        ck = run_dir / "gate.pt"
        if not ck.exists():
            ck = run_dir / "last.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    # 런마다 k·라운드 수가 다르다. 체크포인트에 저장된 설정으로 모델을 세운다
    names = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in st["cfg"].items() if k in names})
    m = CacheRouter(cfg).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    return m, cfg, int(st["global_step"]), ck.name


@torch.no_grad()
def sweep_k(m, gen, cfg, dev, ks):
    """k 만 바꿔가며 재현 정확도를 잰다. 재학습 없이 예산만 조절.
    라운드 수는 고정이라 총 읽기는 k x 라운드다."""
    base = cfg.top_k
    R = cfg.n_rounds
    rows = []
    for k in ks:
        m.cfg.top_k = k
        acc = M.MeanAcc()
        for _ in range(BATCHES):
            b = gen.batch(cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(b["obs"], b["qry"], True)
            r, p = M.P3_recall(o, b)
            acc.add(dict(recall=r, precision=p, **M.accuracy(o, b)))
        d = acc.mean()
        d["k"] = k
        d["read_pct"] = 100.0 * d["read_frac"]         # 실측. 정지 조건이 있으면 k x 라운드보다 적다
        # 이론 상한: 정답이 h 개면 min(h, k x 라운드)/h 까지만 담을 수 있다
        hs = b["hit"].sum(-1).float()
        d["recall_ceil"] = float((torch.clamp(hs, max=k * R) / hs.clamp(min=1)).mean())
        rows.append(d)
    m.cfg.top_k = base
    return rows


@torch.no_grad()
def by_room(m, gen, cfg, dev):
    """방 인덱스별 위치 재현 정확도 — A2(RoPE 불일치) 판정용.

    슬롯 질의에 RoPE 를 안 걸어서 방 위치에 따라 읽기가 달라진다면
    앞쪽 방과 뒤쪽 방의 정확도가 기울어야 한다. 평평하면 무해.
    """
    N = cfg.n_rooms
    hit_n = torch.zeros(N, device=dev)
    ok_n = torch.zeros(N, device=dev)
    for _ in range(BATCHES * 2):
        b = gen.batch(cfg.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(b["obs"], b["qry"], True)
        th, tp, _ = M.gather_targets(b, o["sel"])
        true = (th > 0.5) & o["valid"]                  # 실제로 읽은 슬롯만
        ok = (o["pos_logit"].argmax(-1) == tp) & true
        idx = o["sel"].reshape(-1)
        hit_n.scatter_add_(0, idx, true.reshape(-1).float())
        ok_n.scatter_add_(0, idx, ok.reshape(-1).float())
    return (ok_n / hit_n.clamp(min=1)).cpu(), hit_n.cpu()


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROUTED_N32.run_dir)
    dev = "cuda"
    m, cfg, step, src = load(run_dir, dev)
    gen = EpisodeGen(cfg, dev)
    print(f"\n{run_dir} · {src} · step {step:,} · {cfg.n_rounds}라운드 x k={cfg.top_k}\n")

    # ---- 예산-품질 곡선 ----
    # k x 라운드가 방 수를 넘으면 뒤 라운드가 이미 읽은 방을 다시 뽑는다
    ks = [k for k in (2, 4, 8, 16, cfg.n_rooms) if k * cfg.n_rounds <= cfg.n_rooms]
    print(f"예산-품질 곡선  (k 만 바꿈 · 라운드 {cfg.n_rounds} 고정 · 재학습 없음)")
    print(f"  {'k':>3} {'읽기':>7} {'recall':>8} {'상한':>7} {'재현':>7} "
          f"{'히트':>7} {'★완수':>7} {'집합정확':>9}")
    for d in sweep_k(m, gen, cfg, dev, ks):
        print(f"  {d['k']:>3} {d['read_pct']:>6.1f}% {d['recall']:>8.3f} "
              f"{d['recall_ceil']:>7.3f} {d['pos_acc']:>7.3f} "
              f"{d['hit_acc']:>7.3f} {d['task_recall']:>7.3f} "
              f"{d.get('set_exact', float('nan')):>9.3f}")

    # ---- A2 판정 ----
    acc, cnt = by_room(m, gen, cfg, dev)
    N = cfg.n_rooms
    h1, h2 = acc[: N // 2].mean().item(), acc[N // 2:].mean().item()
    print(f"\nA2 판정 — 방 인덱스별 위치 재현")
    print(f"  앞쪽 {N//2}개 평균 {h1:.4f}   뒤쪽 {N//2}개 평균 {h2:.4f}   "
          f"차 {abs(h1-h2):.4f}")
    print(f"  전체 최소 {acc.min():.4f}  최대 {acc.max():.4f}  표준편차 {acc.std():.4f}")
    verdict = "무해 — 그대로 간다" if abs(h1 - h2) < 0.02 and acc.std() < 0.03 \
              else "버그 — RoPE 를 슬롯 읽기에도 걸고 재학습"
    print(f"  판정: {verdict}")


if __name__ == "__main__":
    main()
