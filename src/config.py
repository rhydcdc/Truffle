"""하이퍼파라미터 한 곳에. DOCS/04-CHECKLIST.md 부록과 같은 값."""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- 데이터 ----
    n_rooms: int = 32          # N
    n_colors: int = 4          # 32방 = 4색 x 2모양 = 8조합
    n_shapes: int = 2
    frames_per_room: int = 4   # n
    grid: int = 16             # 16 x 16
    n_sizes: int = 4           # 크기 등급
    min_rooms_per_combo: int = 2   # 조합당 방 수 하한
    max_rooms_per_combo: int = 7   # 상한. k 보다 작아야 한다
    n_queries: int = 0         # Q. 0 이면 조합 수와 동일하게 자동 설정

    # ---- 모델 ----
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 1024
    d_key: int = 64            # d_k

    # ---- 라우팅 ----
    top_k: int = 8             # 라운드당 읽는 그룹 수
    n_rounds: int = 2          # 반복 라우팅. 2라운드의 q 는 1라운드가 읽은 것으로 보정된다
    round_ks: tuple = ()       # 라운드별 k. 비우면 (top_k,) * n_rounds. 예: (3, 5)
    adaptive_stop: bool = False    # True 면 새 히트가 0개인 라운드에서 멈춘다. n_rounds 는 최대 라운드
    late_cost: float = 0.0         # 늦게 찾은 정답 벌점 λ. 정답 하나당 λ·(찾은 라운드-1)·BCE(1라운드 점수 affine, 1)
    bonus_round: bool = False      # 인내의 보너스 라운드 — 0히트 라운드가 처음 나오면 질의당 한 번은 더 읽는다 (adaptive_stop · hard 에서만)
                                   # 학습 설정에서는 켜지 않는다. 학습이 끝난 모델에 추론 때 켠다 (m.cfg.bonus_round = True)
    tau: float = 0.1           # soft 온도
    group_dropout_p: float = 0.5
    probe_init_scale: float = 0.02   # u ~ 0 => 시작점이 평균 풀링

    # ---- 학습 ----
    batch_size: int = 64
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 60000
    sched_steps: int = 0       # LR 코사인의 길이. 0 이면 max_steps 를 쓴다.
                               # 연장 학습에서 원래 길이를 그대로 두면 LR 이 바닥(lr 의 10%)에 머문다
    pos_weight: float = 0.0    # 히트 BCE 의 양성 가중. 0 이면 배치에서 자동 계산 (비정답/정답 — N=64 에서 약 15)
                               # 15 는 "정답을 놓치는 것이 헛것보다 15배 비싸다" 라, 로짓 0 에서 모델의 진짜 믿음이
                               # p = 1/(1+w) = 6.25 % 다. 구조적으로 헛것(E)을 만든다. 실측 최적 임계값 +3.0 ≈ log(15) = 2.708
    grad_clip: float = 1.0
    hard_switch_frac: float = 0.30   # 전체 스텝의 30% 지점에서 soft -> hard

    # ---- 실행 ----
    seed: int = 0
    device: str = "cuda"
    amp_dtype: str = "bf16"
    grad_checkpoint: bool = False     # 활성값이 8GB 안에 들어간다. 기본 꺼짐
    compile: bool = False             # Windows inductor 가 불안정하면 False 유지
    eval_every: int = 500
    ckpt_every: int = 2000
    eval_batches: int = 8
    run_dir: str = "runs/default"

    # ---- 모드 ----
    routing: bool = True       # False 면 full-attention 기준선
    log_every: int = 50

    def __post_init__(self):
        if self.n_queries == 0:
            self.n_queries = self.n_combos
        # 총 읽기 용량은 k x 라운드 수다. 이게 정답 상한보다 커야
        # recall 실패를 "예산이 모자라서" 로 돌릴 수 없다.
        if self.round_ks:
            assert len(self.round_ks) == self.n_rounds, "round_ks 길이는 n_rounds 와 같아야 한다"
        cap = sum(self.ks)
        assert self.max_rooms_per_combo < cap, (
            f"상한 {self.max_rooms_per_combo} 은 k x 라운드 = {cap} 보다 작아야 "
            "recall 실패를 예산 탓으로 돌릴 수 없다"
        )
        assert self.n_rooms % self.n_combos == 0 or True
        assert self.d_model % self.n_heads == 0

    # ---- 파생값 ----
    @property
    def n_combos(self) -> int:
        return self.n_colors * self.n_shapes

    @property
    def ks(self) -> tuple:
        """라운드별 읽기 수. 총 읽기는 sum(ks)."""
        return tuple(self.round_ks) if self.round_ks else (self.top_k,) * self.n_rounds

    @property
    def seq_obs(self) -> int:
        return self.n_rooms * self.frames_per_room

    @property
    def seq_total(self) -> int:
        return self.seq_obs + self.n_queries

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def dict(self):
        d = asdict(self)
        d.update(n_combos=self.n_combos, seq_obs=self.seq_obs,
                 seq_total=self.seq_total, n_cells=self.n_cells)
        return d


# 0.5 스모크 — 배선 검증용. 몇 초 만에 돈다.
SMOKE = Config(
    n_rooms=8, n_colors=2, n_shapes=2,
    min_rooms_per_combo=1, max_rooms_per_combo=3,
    d_model=64, n_layers=2, n_heads=4, d_ff=256, d_key=16,
    top_k=2, n_rounds=3, adaptive_stop=True,
    batch_size=4, max_steps=2000, warmup_steps=50,
    eval_every=200, ckpt_every=500, eval_batches=2,
    run_dir="runs/smoke",
)
#   라운드당 2칸 x 최대 3라운드 = 6칸 (방 8개). 정답 최대 3개라 1~2라운드에서 멈출 수 있어야 한다.

# 2단계 기준선 — 라우팅 없이 상한선을 잰다.
BASELINE_N32 = Config(routing=False, max_steps=2000, run_dir="runs/baseline_n32")
#   실측: 1,000 스텝에서 재현 1.000 도달. 상한선 확보에 10k 는 과했다.

# 3~4단계 본 노선
ROUTED_N32 = Config(routing=True, top_k=4, n_rounds=8, adaptive_stop=True, max_steps=6000,
                    run_dir="runs/routed_n32_k4_r8")
#   라운드당 4칸 x 최대 8라운드 = 방 32개 전부까지 읽을 수 있다. 정답이 몇 개든
#   "새 히트 0개 라운드" 가 나올 때까지 읽는다 — 최대 라운드가 먼저 막지 않게 N / k 로 둔다.
#   q 보정은 히트로 판정한 방들의 키 평균. 옛 run_dir(routed_n32_r2k4*) 은 Wr 보정 판이라 섞지 않는다.
#   늦게 찾은 정답 벌점 λ = 0 (비교 기준).

# 늦게 찾은 정답 벌점 비교판 — ROUTED_N32 와 λ 만 다르다. 0.05 는 측정값이 아니라 "작게" 잡은 시작값
ROUTED_N32_LATE = Config(routing=True, top_k=4, n_rounds=8, adaptive_stop=True, late_cost=0.05,
                         max_steps=6000, run_dir="runs/routed_n32_k4_r8_late005")

# 벌점을 높인 판 — ROUTED_N32_LATE 와 λ 만 다르다 (0.05 → 0.1). 인내의 보너스 라운드는 학습에 넣지 않는다.
#   학습 때 보너스가 있으면 5~8위에 있다 못 읽힌 정답을 히트 판정 BCE 가 밀지 않고 λ 만 남아 순위 압력이 약해진다.
#   보너스는 학습 이후 추론에 켠다. 프론티어: frontier/N32/n32_lam005_step06000.pt (λ=0.05 + 보너스 → 전부완수 0.9990).
#   λ = 0.1 은 측정값이 아니라 "0.05 의 두 배" 로 잡은 시작값
ROUTED_N32_LATE010 = Config(routing=True, top_k=4, n_rounds=8, adaptive_stop=True, late_cost=0.1,
                            max_steps=6000, run_dir="runs/routed_n32_k4_r8_late010")

# N = 64 — 알고리즘은 그대로, 규모 파라미터만 바꾼다 (DOCS/07-SCALE-PLAN.md).
#   조합 16 = 색 4 × 모양 4 (원 · 삼각 · 사각 · 십자) · 조합당 방 2~14 (사용자 결정 — 학습 때 큰 정답 수도 보여 준다.
#   조합이 16개라 평균은 늘 4 → 실측 분포: 정답 8개 이상 질의 약 5 %, 10개 이상 약 1 %)
#   라운드당 k = 4 · 최대 16라운드 (= N/k) · λ = 0.05 · 모델 크기 그대로. 보너스는 학습에 없고 평가 지표(bonus_*)로만
#   게이트: 보너스를 켠 전부완수 99.9 %. 손실의 pos_weight 는 비정답/정답 으로 계산돼 7 → 15 로 저절로 바뀐다
#   배치 32 (실측 runs/n64_timing*.txt): 64 는 VRAM 12.8 GB 로 8 GB 를 넘어 스텝당 10초. 32 · 24 · 16 은 스텝 시간이 거의 같아
#   (soft 약 0.95 s) 들어가는 가장 큰 32 — 예약 6.68 GB. 스텝당 질의 512 로 N=32 런(64 × 8)과 같다
ROUTED_N64 = Config(n_rooms=64, n_colors=4, n_shapes=4, max_rooms_per_combo=14,
                    routing=True, top_k=4, n_rounds=16, adaptive_stop=True, late_cost=0.05,
                    batch_size=32, max_steps=6000, run_dir="runs/routed_n64_k4_r16_late005")

# N = 64 연장 — 위 런(6,000)이 수렴해서 멈춘 게 아니라 스텝이 떨어져 끝났다.
#   마지막 500스텝에도 히트 .999 · 보너스 전부완수 +0.0024 로 오르는 중이었다 (train.log).
#   같은 run_dir 이라 last.pt 에서 이어 간다. sched_steps=6000 이므로 step 6,000 부터 p 가 1.0 에
#   클램프돼 LR 이 바닥(3e-5)에 머문다 — "같은 학습을 더 돌린 것" 이 되게 하려는 것이다.
#   max_steps 를 그냥 9,000 으로 바꾸면 코사인이 재계산돼 LR 이 1.13e-4 로 3.8배 되살아난다 (다른 실험이 된다).
#   hard_switch_frac 0.20 은 0.20 × 9,000 = 1,800 으로 원래와 같은 전환 지점 (재시작에서는 체크포인트가 hard 를 복원하므로 무관).
ROUTED_N64_EXT = Config(n_rooms=64, n_colors=4, n_shapes=4, max_rooms_per_combo=14,
                        routing=True, top_k=4, n_rounds=16, adaptive_stop=True, late_cost=0.05,
                        batch_size=32, max_steps=9000, sched_steps=6000, hard_switch_frac=0.20,
                        run_dir="runs/routed_n64_k4_r16_late005")

# N = 64 LR 되살리기 — 위 연장(6,000 -> 9,000, LR 바닥 고정)은 6,000~7,000 에서만 개선되고
#   (32,768 질의 실패 287 -> 193) 그 뒤 2,000스텝은 평평했다. 바닥 LR(3e-5)에서 짜낼 건 끝났다는 뜻이다.
#   그래서 이번엔 sched_steps 를 비워 코사인을 15,000 기준으로 재계산한다 — step 9,000 의 LR 이
#   3e-5 -> 약 1.35e-4 로 되살아나고 15,000 에서 다시 바닥에 닿는다.
#   같은 run_dir 이라 last.pt (step 9,000) 에서 이어 간다. hard_switch_frac 0.12 = 0.12 x 15,000 = 1,800 (원래와 같은 전환 지점).
#   노리는 것: 판정 정밀도. 실패의 병목이 헛것(E)이고, 추론 임계값(+3.0)까지 다 써도 81~101 건으로 게이트(33)에 못 닿았다.
ROUTED_N64_LR = Config(n_rooms=64, n_colors=4, n_shapes=4, max_rooms_per_combo=14,
                       routing=True, top_k=4, n_rounds=16, adaptive_stop=True, late_cost=0.05,
                       batch_size=32, max_steps=15000, hard_switch_frac=0.12,
                       run_dir="runs/routed_n64_k4_r16_late005")

# N = 64 · pos_weight = 1 싼 시험 — "커리큘럼 15 -> 1" 의 뒷부분 그 자체를 1,500스텝만 재 본다.
#   가설: pos_weight 15 아래서 모델의 실효 동작점이 p=6.25 % 라 진짜 결정이 나는 50 % 부근을 한 번도 안 다듬는다.
#         1 로 내리면 그 자리에서 기울기를 받아 로짓 분리가 날카로워진다.
#   반증 가능: 임계값 +3.0 (= log 15) 이 이미 경계를 옳은 자리로 옮겨 실패 81건을 냈다. 손실을 바꿔도 81 근처면
#         "경계 이동이 전부였다" 가 확정되고, 남은 후보는 용량(head_hit 단층 · 모델 크기)으로 좁혀진다.
#   **LR 은 일부러 안 바꾼다** — sched_steps = max_steps 라 step 15,000 에서 3.6e-5 -> 3.0e-5 로 바닥 고정.
#         이번 시험의 변수는 손실 하나여야 한다. run_dir 도 분리한다 (손실이 다른 구간이 섞이면 곡선을 못 읽는다).
#   실행 전에 runs/routed_n64_k4_r16_late005/last.pt 를 runs/routed_n64_pw1/ 로 복사한다.
ROUTED_N64_PW1 = Config(n_rooms=64, n_colors=4, n_shapes=4, max_rooms_per_combo=14,
                        routing=True, top_k=4, n_rounds=16, adaptive_stop=True, late_cost=0.05,
                        batch_size=32, max_steps=16500, sched_steps=16500, hard_switch_frac=0.11,
                        pos_weight=1.0, run_dir="runs/routed_n64_pw1")
