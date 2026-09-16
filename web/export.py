"""브라우저용 자산을 만든다 — ONNX 그래프 둘 + 미리 계산한 에피소드.

    python web/export.py

왜 둘로 쪼개나: 우리 모델은 한 번의 순전파가 아니라 데이터에 따라 도는 반복 루프다
(top-k · 0히트 정지 · q 보정 · patience). 그 동적 제어 흐름은 ONNX 로 빼기 고약하다.
대신 **정적인 두 조각만** 내보내고 루프는 JS 가 몬다:

    qenc.onnx   그림(one-hot) -> q, e_q          제어 흐름 없음
    slot.onnx   (e_q, 고른 그룹의 KV) -> 히트·위치·크기   가져오기는 JS, 그래프는 정적

관찰 인코딩은 에피소드가 고정이라 **여기서 한 번만** 하고 결과(층별 KV · 키 64개)를 파일로 싣는다.
→ 브라우저는 질의 인코딩부터만 하면 된다.

산출물은 web/assets/ 에 떨어진다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import measure                                    # noqa: E402
from src.data import EpisodeGen                   # noqa: E402

CKPT = ROOT / "frontier" / "N64" / "n64_lr_step15000.pt"
OUT = Path(__file__).resolve().parent / "assets"
EPISODE_SEED = 20260916                           # demo/serve.py 와 같은 판
DEV = "cpu"

COLORS = [("파랑", "--o-blue"), ("빨강", "--o-red"), ("초록", "--o-green"), ("노랑", "--o-yellow")]
SHAPES = ["원", "삼각", "사각", "십자"]


class QEnc(nn.Module):
    """그림 one-hot -> 정규화된 q, 그리고 슬롯 시작 상태 e_q."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, oh):
        e = self.m.embed(oh)
        h = self.m._encode_query(e)
        return F.normalize(self.m.Wq(h), dim=-1), e


class Slot(nn.Module):
    """고른 그룹의 KV 를 12층으로 읽어 히트·위치·크기 로짓."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, e, kg, vg):
        x = e
        for blk, k_, v_ in zip(self.m.blocks, kg.unbind(0), vg.unbind(0)):
            x = blk.forward_slot(x, k_, v_)
        z = self.m.norm_f(x)
        return self.m.head_hit(z).squeeze(-1), self.m.head_pos(z), self.m.head_size(z)


@torch.no_grad()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m, cfg, step, src = measure.load(CKPT, DEV)
    m.eval()
    N, L, H, dh, n = cfg.n_rooms, cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.frames_per_room
    n_sym = cfg.n_colors + 1

    # ---------------------------------------------------------------- 에피소드 (고정)
    torch.manual_seed(EPISODE_SEED)
    gen = EpisodeGen(cfg, DEV)
    b = gen.batch(1, noise_p=0.008)

    # 관찰 인코딩 — 여기서 한 번만. 브라우저는 이 결과를 받아 쓴다
    x = m._embed_frames(b["obs"])
    cos, sin = m._rope_for(x.shape[1], DEV)
    kvs = []
    for blk in m.blocks:
        x, kv = blk.forward_obs(x, cos, sin)
        kvs.append(kv)
    h_obs = m.norm_f(x)
    hg = h_obs.view(1, N, n, -1)
    pool_w = (hg * m.u).sum(-1).softmax(-1)
    keys = F.normalize(m.W((hg * pool_w[..., None]).sum(2)), dim=-1)      # (1,N,d_key)

    K = torch.stack([kv[0][0] for kv in kvs])      # (L, T, H, dh)
    V = torch.stack([kv[1][0] for kv in kvs])
    K = K.view(L, N, n, H, dh)
    V = V.view(L, N, n, H, dh)

    # ---------------------------------------------------------------- ONNX
    # fp16 으로 내보낸다 — 크기가 절반이다. 결정이 바뀌는지는 web/verify.py 가 센다.
    # (바꾸려면 HALF=False. 그때 자산이 두 배가 된다)
    HALF = True
    mh = m.half() if HALF else m
    dt = torch.float16 if HALF else torch.float32

    oh = torch.zeros(1, 1, cfg.n_cells * n_sym, dtype=dt)
    oh[..., ::n_sym] = 1.0
    torch.onnx.export(QEnc(mh), (oh,), OUT / "qenc.onnx",
                      input_names=["oh"], output_names=["q", "e_q"],
                      opset_version=17, dynamo=False)

    M = cfg.top_k
    e = torch.randn(M, 1, cfg.d_model, dtype=dt)
    kg = torch.randn(L, M, n, H, dh, dtype=dt)
    vg = torch.randn(L, M, n, H, dh, dtype=dt)
    torch.onnx.export(Slot(mh), (e, kg, vg), OUT / "slot.onnx",
                      input_names=["e", "kg", "vg"], output_names=["hit", "pos", "size"],
                      dynamic_axes={"e": {0: "slots"}, "kg": {1: "slots"}, "vg": {1: "slots"},
                                    "hit": {0: "slots"}, "pos": {0: "slots"}, "size": {0: "slots"}},
                      opset_version=17, dynamo=False)

    # ---------------------------------------------------------------- 자산
    # KV 는 fp16 으로 싣는다 (fp32 대비 절반). 결정이 바뀌는지는 verify.py 가 센다
    (OUT / "kv.bin").write_bytes(
        torch.cat([K.flatten(), V.flatten()]).to(torch.float16).numpy().tobytes())
    (OUT / "keys.bin").write_bytes(keys[0].to(torch.float32).numpy().tobytes())

    combo = b["room_combo"][0].tolist()
    pos = b["pos"][0].tolist()
    size = b["size"][0].tolist()
    meta = dict(
        n_rooms=N, grid=cfg.grid, k=cfg.top_k, max_rounds=cfg.n_rounds, n_layers=L,
        n_heads=H, d_head=dh, d_model=cfg.d_model, d_key=cfg.d_key,
        frames_per_room=n, n_colors=cfg.n_colors, n_shapes=cfg.n_shapes, n_sizes=cfg.n_sizes,
        n_sym=n_sym, n_cells=cfg.n_cells, radii=[float(r) for r in gen.radii],
        colors=[dict(name=c, var=v) for c, v in COLORS], shapes=SHAPES,
        step=step, src=src, seed=EPISODE_SEED,
        rooms=[dict(id=i, combo=c, color=c // cfg.n_shapes, shape=c % cfg.n_shapes,
                    cy=p // cfg.grid, cx=p % cfg.grid, size=s)
               for i, (c, p, s) in enumerate(zip(combo, pos, size))],
    )
    (OUT / "episode.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"내보내기 완료 — {OUT}")
    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"  {f.name:16s} {f.stat().st_size/1048576:7.1f} MB")
    print(f"  합계 {total/1048576:.1f} MB   <- 첫 방문 다운로드")


if __name__ == "__main__":
    main()
