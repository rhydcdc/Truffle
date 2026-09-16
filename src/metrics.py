"""점검 수치 P2 ~ P5 와 손실. DOCS/04-CHECKLIST.md "점검 수치" 절 그대로.

원칙: 각 수치가 파이프라인을 절반으로 가른다.
    P2  키 분리도      키가 내용을 담았나   (top-k 도 읽기도 거치지 않는다)
    P3  recall@k       라우터가 고르나
    P4  오라클 라우팅  읽기·표현이 되나     (완벽한 라우팅일 때의 천장)
    P5  w 엔트로피 · 선택 위치 CV   풀링 포화 · 최근성 붕괴

최종 정확도 ~= recall x 오라클 재현 x 판정 정확도.
어느 항이 끌어내리는지는 diagnose() 가 문자열로 답한다.

전부 GPU 텐서 위에서 계산하고 반환 직전에만 float 으로 내린다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:                      # 런타임 import 없음. 순환 의존 방지
    from config import Config
    from data import EpisodeGen

Batch = dict[str, torch.Tensor]        # data.EpisodeGen.batch 의 반환
Out = dict[str, torch.Tensor]          # model.CacheRouter.forward 의 반환

__all__ = [
    "gather_targets", "loss_fn", "accuracy",
    "P2_key_separation", "P3_recall", "P4_oracle", "P5_health",
    "build_force_groups", "diagnose", "evaluate", "MeanAcc",
]


# ---------------------------------------------------------------- 내부 도구
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 가 True 인 원소만의 평균. 하나도 없으면 0 (NaN 아님)."""
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def _valid(out: Out) -> torch.Tensor:
    """실제로 읽은 슬롯 (B,Q,S). 멈춘 뒤 라운드 · 후보가 바닥나 가려진 방을 뽑은 슬롯은 False."""
    v = out.get("valid")
    return torch.ones_like(out["sel"], dtype=torch.bool) if v is None else v


def _read_rooms(sel: torch.Tensor, mask: torch.Tensor, N: int) -> torch.Tensor:
    """mask 인 슬롯이 가리키는 방 (B,Q,N). 나머지 슬롯은 덤프 칸 N 에 써서 버린다."""
    B, Q, _ = sel.shape
    idx = torch.where(mask, sel, N)
    m = torch.zeros(B, Q, N + 1, dtype=torch.bool, device=sel.device)
    m.scatter_(2, idx, torch.ones_like(idx, dtype=torch.bool))
    return m[..., :N]


def _n_rooms(out: Out) -> int:
    """출력 딕트에서 N 을 읽는다. score 가 없으면 keys, 그것도 없으면 pool_w."""
    for k, dim in (("score", -1), ("keys", 1), ("pool_w", 1)):
        if k in out:
            return out[k].shape[dim]
    raise KeyError("N 을 알 수 없다 — score/keys/pool_w 중 하나가 필요하다")


# ---------------------------------------------------------------- 타깃 정렬
def gather_targets(
    batch: Batch, sel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """선택된 슬롯 기준으로 라벨을 모은다. sel (B,Q,k) -> (B,Q,k) 셋.

    슬롯 j 의 방이 r = sel[b,q,j] 이면
        hit  = batch["hit"][b,q,r]   위치·크기와 달리 질의마다 다르다
        pos  = batch["pos"][b,r]
        size = batch["size"][b,r]
    """
    B, Q, k = sel.shape
    flat = sel.reshape(B, Q * k)                       # pos/size 는 q 와 무관

    tgt_hit = batch["hit"].gather(2, sel).to(torch.float32)         # (B,Q,k)
    tgt_pos = batch["pos"].gather(1, flat).reshape(B, Q, k)         # (B,Q,k)
    tgt_size = batch["size"].gather(1, flat).reshape(B, Q, k)       # (B,Q,k)
    return tgt_hit, tgt_pos, tgt_size


# ---------------------------------------------------------------- 손실
def loss_fn(
    out: Out, batch: Batch, pos_weight: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """히트 판정(전 슬롯) + 위치·크기 재현(히트 슬롯만).

    라우터에 직접 주는 손실은 없다 — 잘못 고르면 아래 셋이 틀리고
    그 손실이 score 를 타고 흘러든다.
    """
    tgt_hit, tgt_pos, tgt_size = gather_targets(batch, out["sel"])
    valid = _valid(out)
    hit_m = (tgt_hit > 0.5) & valid                     # 읽지 않은 슬롯은 재현을 배울 수 없다

    # 히트 판정은 방 N 개 전체에서 잰다. 선택된 것만 재면 "정답인데 안 고른"
    # 방이 손실에 아예 안 보여서, 라우터가 고른 것에 확신만 키우는 해로 간다.
    # (실측: loss 0.0017 인데 recall 0.70. DOCS/05-DEBUG-LOG.md 참조)
    full = out["hit_full"].float()                      # (B,Q,N)
    tgt_full = batch["hit"].to(full.dtype)
    if pos_weight is None:                              # 양성이 N 대비 소수라 균형을 맞춘다
        n_pos = tgt_full.sum().clamp(min=1.0)
        pos_weight = ((tgt_full.numel() - n_pos) / n_pos).detach()
    # bf16 AMP 아래서도 손실은 fp32 로. 포화 구간에서 값이 뭉개진다
    l_hit = F.binary_cross_entropy_with_logits(full, tgt_full, pos_weight=pos_weight)

    n_cells = out["pos_logit"].shape[-1]
    n_sizes = out["size_logit"].shape[-1]
    ce_pos = F.cross_entropy(
        out["pos_logit"].float().reshape(-1, n_cells), tgt_pos.reshape(-1),
        reduction="none",
    )
    ce_size = F.cross_entropy(
        out["size_logit"].float().reshape(-1, n_sizes), tgt_size.reshape(-1),
        reduction="none",
    )
    # 히트 아닌 슬롯은 마스크가 0 이라 값도 그래디언트도 0. 히트가 없어도 0/1 이다
    l_pos = _masked_mean(ce_pos, hit_m.reshape(-1))
    l_size = _masked_mean(ce_size, hit_m.reshape(-1))

    total = l_hit + l_pos + l_size

    # ---- 늦게 찾은 정답 벌점 ----
    # 2라운드 이후에 찾은 정답 하나마다 λ · (찾은 라운드 - 1) · BCE(1라운드 점수의 affine, 1).
    # 늦게 찾은 정답의 1라운드 점수를 밀어 올려 다음번엔 1라운드에 잡히게 한다 → 라운드 수 감소.
    # 1라운드에 찾았거나 끝내 못 찾은 정답은 0 — 못 찾은 정답은 이미 히트 판정 BCE 가 더 크게 벌한다.
    # 정답 수로 나눈다. 정답이 k 보다 많으면 일부는 늘 벌점을 받으므로 λ 는 작게 둔다.
    l_late = total.new_zeros(())
    lam = out.get("late_cost", 0.0)
    if lam > 0:
        sel = out["sel"]
        N = batch["hit"].shape[-1]
        ans_read = (valid & hit_m.new_ones(()) & (tgt_hit > 0.5)).float()          # (B,Q,S) 읽은 정답 칸
        r0 = out["round_of_slot"].to(torch.float32)                               # (S,) 0부터
        late_w = torch.zeros(*sel.shape[:2], N + 1, device=sel.device) \
            .scatter_add(-1, torch.where(valid, sel, N), r0 * ans_read)[..., :N]  # (B,Q,N) 방마다 찾은 라운드-1
        bce1 = F.binary_cross_entropy_with_logits(
            out["late_logit"].float(), torch.ones_like(late_w), reduction="none")
        l_late = lam * (late_w * bce1).sum() / batch["hit"].sum().clamp(min=1)
        total = total + l_late

    h, p, s, lt = torch.stack([l_hit, l_pos, l_size, l_late]).detach().tolist()   # 동기화 1회
    return total, dict(loss_hit=h, loss_pos=p, loss_size=s, loss_late=lt)


# ---------------------------------------------------------------- 정확도
def accuracy(out: Out, batch: Batch) -> dict[str, float]:
    """판정·재현 정확도. 위치·크기는 진짜 히트 슬롯에서만 잰다."""
    tgt_hit, tgt_pos, tgt_size = gather_targets(batch, out["sel"])
    valid = _valid(out)                                # 실제로 읽은 슬롯만 출력이다
    true_hit = (tgt_hit > 0.5) & valid
    pred_hit = (out["hit_logit"] > 0.0) & valid        # sigmoid > 0.5 와 같다

    same = pred_hit == true_hit                        # (B,Q,S)  무효 슬롯은 둘 다 False 라 True
    hit_acc = _masked_mean(same.float(), valid)
    exact_set = same.all(-1).float().mean()            # 읽은 슬롯으로 제한한 집합 일치

    pos_ok = (out["pos_logit"].argmax(-1) == tgt_pos).float()
    size_ok = (out["size_logit"].argmax(-1) == tgt_size).float()
    pos_acc = _masked_mean(pos_ok, true_hit)
    size_acc = _masked_mean(size_ok, true_hit)

    # 최종 답은 방 N 개 전체에 대한 집합이다. 선택 슬롯만 보면 놓친 방이 안 보인다.
    pred_full = out["hit_full"] > 0.0                  # (B,Q,N)
    true_full = batch["hit"]
    tp = (pred_full & true_full).sum(-1).float()
    set_rec = _masked_mean(tp / true_full.sum(-1).clamp(min=1).float(),
                           true_full.sum(-1) > 0)
    set_prec = _masked_mean(tp / pred_full.sum(-1).clamp(min=1).float(),
                            pred_full.sum(-1) > 0)
    set_exact = (pred_full == true_full).all(-1).float().mean()

    # ---- 실제 태스크 완수 ----
    # 찾기만 한 건 완수가 아니다. 정답 방을 top-k 안에 넣고 · 히트로 판정하고 ·
    # 위치와 크기까지 맞혀야 그 방을 푼 것이다. 안 읽은 방은 재현이 불가능하므로
    # 자동으로 미완이다.
    solved_slot = (true_hit & pred_hit
                   & (out["pos_logit"].argmax(-1) == tgt_pos)
                   & (out["size_logit"].argmax(-1) == tgt_size))      # (B,Q,k)
    n_solved = solved_slot.sum(-1).float()                            # (B,Q)
    n_true = true_full.sum(-1).float()
    task_recall = _masked_mean(n_solved / n_true.clamp(min=1.0), n_true > 0)
    # 전부완수 = 물어본 것에 정확히 답했는가. 두 가지만 본다.
    #   1) 출력한 방 집합 == 정답 방 집합  (더도 덜도 없이)
    #   2) 그 정답 방 전부의 위치·크기를 맞혔다
    # 출력 = 읽고 히트로 판정한 방. 안 읽은 방의 hit_full 은 학습 신호일 뿐 출력이 아니다
    # — 넣으면 읽지도 않은 방이 답에 섞이거나(헛것) 점수만으로 맞은 걸로 셈해진다.
    n_fp = (pred_hit & ~true_hit).sum(-1).float()                     # 읽은 비정답을 답으로 냄
    task_exact = _masked_mean(((n_solved == n_true) & (n_fp == 0)).float(), n_true > 0)

    # 읽기량 — 정답 수에 따라 달라져야 한다
    read_frac = valid.sum(-1).float().mean() / true_full.shape[-1]    # 실제로 읽은 방 비율
    ra = out.get("round_active")
    rounds = ra.sum(-1).float().mean() if ra is not None else torch.ones((), device=valid.device)

    v = torch.stack([hit_acc, pos_acc, size_acc, exact_set,
                     set_rec, set_prec, set_exact,
                     task_recall, task_exact, read_frac, rounds]).tolist()
    return dict(hit_acc=v[0], pos_acc=v[1], size_acc=v[2], exact_set=v[3],
                set_recall=v[4], set_precision=v[5], set_exact=v[6],
                task_recall=v[7], task_exact=v[8], read_frac=v[9], rounds=v[10])


# ---------------------------------------------------------------- P2 키 분리도
def P2_key_separation(keys: torch.Tensor, room_combo: torch.Tensor) -> float:
    """같은 조합끼리의 평균 코사인 - 다른 조합끼리의 평균 코사인.

    keys (B,N,d_key) 는 이미 L2 정규화 -> 코사인은 행렬곱. 대각선은 뺀다.
    무작위 초기화에서 0 근처, 학습이 되면 올라간다.
    """
    k = keys.float()
    B, N, _ = k.shape
    cos = k @ k.transpose(1, 2)                                    # (B,N,N)

    same = room_combo[:, :, None] == room_combo[:, None, :]        # (B,N,N)
    off = ~torch.eye(N, dtype=torch.bool, device=k.device)[None]   # 자기 자신 제외
    m_same = (same & off).to(cos.dtype)
    m_diff = (~same & off).to(cos.dtype)

    c_same = m_same.sum((1, 2))
    c_diff = m_diff.sum((1, 2))
    mu_same = (cos * m_same).sum((1, 2)) / c_same.clamp(min=1.0)
    mu_diff = (cos * m_diff).sum((1, 2)) / c_diff.clamp(min=1.0)

    # 한쪽 쌍이 아예 없는 에피소드는 뺀다 (조합이 전부 유일하거나 전부 같을 때)
    ok = (c_same > 0) & (c_diff > 0)
    return float(_masked_mean(mu_same - mu_diff, ok))


# ---------------------------------------------------------------- P3 recall@k
def P3_recall(out: Out, batch: Batch) -> tuple[float, float]:
    """(recall, precision). 히트가 0 인 질의는 recall 평균에서 제외한다.

    recall    질의별 [정답 방 중 sel 에 든 비율] 의 평균. 복구 불가능한 실패를 잰다
    precision sel 슬롯 중 진짜 히트의 비율. 읽고 버리면 되므로 치명적이지 않다
    """
    hit = batch["hit"]                                             # (B,Q,N) bool
    sel = out["sel"]                                               # (B,Q,S)
    valid = _valid(out)

    picked = _read_rooms(sel, valid, hit.shape[-1])                # 실제로 읽은 방만

    n_hit = hit.sum(-1).float()                                    # (B,Q)
    found = (picked & hit).sum(-1).float()
    recall = _masked_mean(found / n_hit.clamp(min=1.0), n_hit > 0)

    tgt_hit, _, _ = gather_targets(batch, sel)
    precision = _masked_mean(tgt_hit, valid)

    r, p = torch.stack([recall, precision.to(recall.dtype)]).tolist()
    return r, p


# ---------------------------------------------------------------- P4 오라클
def build_force_groups(batch: Batch, cfg: "Config") -> torch.Tensor:
    """정답 그룹 G* 를 True 로 둔 (B,Q,N) 마스크. 나머지는 무작위 방으로 채운다.

    슬롯 수를 k 로 맞춰야 모양이 같아진다. rand 는 [0,1) 이라 히트(+1.0)가
    항상 앞선다 — topk 가 정답을 먼저 다 담고 남는 자리만 무작위로 채운다.
    """
    hit = batch["hit"]
    B, Q, N = hit.shape
    k = min(cfg.ks[0], N)                  # 오라클은 1라운드 선택을 강제한다

    key = hit.to(torch.float32) + torch.rand(B, Q, N, device=hit.device)
    idx = key.topk(k, dim=-1).indices                              # (B,Q,k)

    force = torch.zeros_like(hit)
    force.scatter_(2, idx, torch.ones_like(idx, dtype=torch.bool))
    return force


@torch.no_grad()
def P4_oracle(model: torch.nn.Module, batch: Batch, cfg: "Config") -> dict[str, float]:
    """라우터를 끄고 정답 그룹을 직접 먹였을 때의 위치·크기 재현 = 천장."""
    force = build_force_groups(batch, cfg)
    out = model(batch["obs"], batch["qry"], True, force_groups=force)

    tgt_hit, tgt_pos, tgt_size = gather_targets(batch, out["sel"])
    true_hit = (tgt_hit > 0.5) & _valid(out)
    pos_ok = (out["pos_logit"].argmax(-1) == tgt_pos).float()
    size_ok = (out["size_logit"].argmax(-1) == tgt_size).float()

    v = torch.stack([
        _masked_mean(pos_ok, true_hit), _masked_mean(size_ok, true_hit)
    ]).tolist()
    return dict(p4_pos_acc=v[0], p4_size_acc=v[1])


# ---------------------------------------------------------------- P5 건강도
def P5_health(out: Out) -> dict[str, float]:
    """풀링 포화와 최근성 붕괴. 매 스텝 찍어도 공짜다.

    pool_entropy  w 의 엔트로피(nats). 0 으로 가면 softmax 포화 = 전용 슬롯 퇴화
    sel_pos_cv    방 인덱스별 선택 빈도의 변동계수. 높으면 특정 위치로 쏠린 것
    """
    w = out["pool_w"].float()                                      # (B,N,n)
    # 단항 마이너스가 메서드 체인보다 나중에 묶인다. 한 줄로 쓰면
    # clamp 가 음수인 Σ w·log w 를 0 으로 잘라버린 뒤 부호를 뒤집어 -0.0 이 된다.
    ent = -(w * w.clamp(min=1e-9).log()).sum(-1)                   # (B,N)
    ent = ent.mean().clamp(min=0.0)

    N = _n_rooms(out)
    sel = torch.where(_valid(out), out["sel"], N).reshape(-1)      # 무효 슬롯은 덤프 칸으로
    cnt = torch.zeros(N + 1, device=sel.device, dtype=torch.float32)
    cnt.scatter_add_(0, sel, torch.ones_like(sel, dtype=torch.float32))
    cnt = cnt[:N]
    cv = cnt.std() / cnt.mean().clamp(min=1e-6)                    # 총합 > 0 이면 안전

    v = torch.stack([ent, cv]).tolist()
    return dict(pool_entropy=v[0], sel_pos_cv=v[1])


# ---------------------------------------------------------------- 진단표
def diagnose(
    p2: float, p3_recall: float, p4_pos_acc: float,
    p2_thr: float = 0.15, recall_thr: float = 0.9, pos_thr: float = 0.8,
    routing: bool = True,
) -> str:
    """04-CHECKLIST.md 진단표. 어디를 고칠지 한 줄로.

    recall 이 먼저다. P2 는 recall 이 낮을 때 *원인을 가르는* 지표이지
    그 자체가 목표가 아니다 — recall 이 좋으면 키가 얼마나 뭉쳤든 상관없다.
    (스모크에서 P2 0.228 · recall 0.906 이 나왔다. 0.3 임계값이 높았다.)

    p2_thr 0.15 도 아직 찍은 값이다. 라우팅판 곡선이 나오면 recall 이
    꺾이는 지점의 P2 로 다시 잡는다.
    """
    if not routing:
        # 전체를 읽으므로 recall 은 항상 1, precision 은 |G*|/N,
        # 키는 선택에 안 쓰이니 뭉칠 이유가 없다. 셋 다 아무것도 재지 않는다.
        return "기준선 — 라우팅 미사용. P2·recall 은 의미 없다"

    read_ok = p4_pos_acc >= pos_thr
    if p3_recall >= recall_thr:
        return "정상" if read_ok else "읽기·표현 문제 — 그룹이 안 갈렸다. 드롭아웃 p 상향"

    # 여기부터 recall 이 낮다. 원인이 키냐 질의냐를 P2 가 가른다.
    if p2 < p2_thr:
        return "키가 내용을 못 담았다 — 풀링에서 정보가 죽음. 멀티 프로브 필요"
    if read_ok:
        return "라우터만 문제 — 키는 뭉쳤다. W_q 확대 / tau 조정 / hard 전환 늦춤"
    return "라우터·읽기 동시 문제 — 읽기부터 고친다. 드롭아웃 p 상향 후 재측정"


# ---------------------------------------------------------------- 집계
class MeanAcc:
    """스칼라 딕트를 받아 키마다 평균낸다. 배치 수가 달라도 안전하다."""

    def __init__(self):
        self._sum: dict[str, float] = {}
        self._cnt: dict[str, int] = {}

    def add(self, d: dict[str, float]) -> None:
        for k, v in d.items():
            self._sum[k] = self._sum.get(k, 0.0) + float(v)
            self._cnt[k] = self._cnt.get(k, 0) + 1

    def mean(self) -> dict[str, float]:
        return {k: s / max(self._cnt[k], 1) for k, s in self._sum.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, gen: "EpisodeGen", cfg: "Config",
    n_batches: int = 0, hard: bool = True,
) -> dict[str, float | str]:
    """P2 ~ P5 · 손실 · 정확도를 배치 평균으로. diagnosis 문자열까지 한 딕트에.

    n_batches 가 0 이면 cfg.eval_batches 를 쓴다. 모드는 반드시 되돌린다.
    hard · 정지 규칙이면 인내의 보너스 라운드를 켠 순전파를 한 번 더 해 bonus_* 지표를 더한다.
    평가 지표일 뿐 손실 · 학습 신호와 무관하다 (보너스는 학습에 넣지 않는다).
    """
    n = n_batches if n_batches > 0 else cfg.eval_batches
    was_training = model.training
    model.eval()
    acc = MeanAcc()
    try:
        for _ in range(n):
            batch = gen.batch(cfg.batch_size)
            out = model(batch["obs"], batch["qry"], hard)

            total, comp = loss_fn(out, batch)
            acc.add(comp)
            acc.add(dict(loss=float(total.detach())))
            acc.add(accuracy(out, batch))

            recall, precision = P3_recall(out, batch)
            acc.add(dict(p3_recall=recall, p3_precision=precision))
            acc.add(dict(p2_key_sep=P2_key_separation(out["keys"], batch["room_combo"])))
            acc.add(P5_health(out))
            acc.add(P4_oracle(model, batch, cfg))

            # 인내의 보너스 라운드를 켠 추론 지표 — 게이트는 이 값으로 본다. 난수를 쓰지 않아 위 지표의 데이터 순서를 바꾸지 않는다
            if cfg.routing and cfg.adaptive_stop and hard:
                mcfg = getattr(model, "cfg", cfg)
                prev = mcfg.bonus_round
                mcfg.bonus_round = True
                try:
                    a = accuracy(model(batch["obs"], batch["qry"], hard), batch)
                finally:
                    mcfg.bonus_round = prev
                acc.add(dict(bonus_task_exact=a["task_exact"], bonus_task_recall=a["task_recall"],
                             bonus_read_frac=a["read_frac"], bonus_rounds=a["rounds"]))
    finally:
        model.train(was_training)

    m = acc.mean()
    m["diagnosis"] = diagnose(m["p2_key_sep"], m["p3_recall"], m["p4_pos_acc"],
                              routing=cfg.routing)
    return m
