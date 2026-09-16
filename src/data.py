"""GPU 위에서 에피소드를 생성한다. CPU 는 건드리지 않는다.

에피소드 하나
    방 N 개, 방마다 도형 1개 = (색, 모양) 조합. 위치와 크기는 방마다 랜덤.
    같은 조합이 여러 방에 중복 등장한다 (조합당 2 ~ 7 개, 합 = N).
    방 하나가 관찰 프레임 n 개를 만든다. 프레임끼리는 잡음만 다르다 —
    각 프레임에 정해진 역할은 없다.
    질의 Q 개 = 조합 하나씩. 도형만 그려져 있고 위치·크기는 무작위다.

라벨
    hit  (B, Q, N)  질의 q 의 조합을 방 n 이 가졌는가
    pos  (B, N)     방 n 의 도형이 있던 칸 (0 .. grid*grid-1)
    size (B, N)     방 n 의 도형 크기 등급
"""
import torch


class EpisodeGen:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        g = cfg.grid

        # 좌표 격자 — 한 번만 만들어 상주시킨다
        yy, xx = torch.meshgrid(
            torch.arange(g, device=device),
            torch.arange(g, device=device),
            indexing="ij",
        )
        self.yy = yy.float()
        self.xx = xx.float()

        # 조합당 방 수 후보를 미리 만들어 캐싱한다.
        # 매 스텝 제약을 만족하는 벡터를 새로 찾는 대신 여기서 뽑아 쓴다.
        self.count_pool = self._build_count_pool(n_pool=2048)

        # 크기 등급 -> 반지름
        self.radii = torch.arange(cfg.n_sizes, device=device).float() + 2.0

    # ---------------------------------------------------------------- 조합 배정
    def _build_count_pool(self, n_pool: int) -> torch.Tensor:
        """합이 N 이고 각 원소가 [lo, hi] 인 count 벡터 풀. (n_pool, C)"""
        cfg = self.cfg
        C, N = cfg.n_combos, cfg.n_rooms
        lo, hi = cfg.min_rooms_per_combo, cfg.max_rooms_per_combo
        assert lo * C <= N <= hi * C, f"N={N} 을 [{lo},{hi}] x {C} 로 만들 수 없다"

        base = torch.full((n_pool, C), N // C, dtype=torch.long)
        rem = N - (N // C) * C
        if rem:
            # 나머지를 무작위 위치에 하나씩 얹는다
            idx = torch.rand(n_pool, C).argsort(-1)[:, :rem]
            base.scatter_add_(1, idx, torch.ones_like(idx))

        # 경계 안에서 +1/-1 스왑을 반복해 분포를 흩는다
        for _ in range(64):
            i = torch.randint(0, C, (n_pool, 1))
            j = torch.randint(0, C, (n_pool, 1))
            ci = base.gather(1, i)
            cj = base.gather(1, j)
            ok = (ci > lo) & (cj < hi) & (i != j)
            base.scatter_(1, i, torch.where(ok, ci - 1, ci))
            base.scatter_(1, j, torch.where(ok, cj + 1, cj))

        assert (base.sum(-1) == N).all()
        assert (base >= lo).all() and (base <= hi).all()
        return base.to(self.device)

    # ---------------------------------------------------------------- 렌더러
    def _mask(self, cy, cx, rad, shape_id):
        """도형 마스크. cy/cx/rad/shape_id 는 (...) 이고 결과는 (..., g, g).

        스프라이트 뱅크 대신 좌표에서 바로 계산한다 — 원소별 연산 대여섯 번이라
        마스크를 색인하고 옮기는 것보다 싸고, 위치·크기가 연속이라 뱅크가 커진다.
        """
        dy = self.yy - cy[..., None, None]
        dx = self.xx - cx[..., None, None]
        r = rad[..., None, None]
        s = shape_id[..., None, None]

        circle = dy * dy + dx * dx <= r * r
        tri = (dy >= -r) & (dy <= r) & (2 * dx.abs() <= dy + r)
        square = torch.maximum(dy.abs(), dx.abs()) <= r
        thin = torch.clamp(r / 3.0, min=1.0)
        cross = ((dy.abs() <= r) & (dx.abs() <= thin)) | \
                ((dx.abs() <= r) & (dy.abs() <= thin))

        m = torch.where(s == 0, circle, tri)
        m = torch.where(s == 2, square, m)
        m = torch.where(s == 3, cross, m)
        return m

    def _paint(self, cy, cx, rad, shape_id, color_id, noise_p):
        """마스크를 색 인덱스 격자로. 0 = 빈칸, 1.. = 색."""
        g = self.cfg.grid
        m = self._mask(cy, cx, rad, shape_id)
        frame = m.long() * (color_id[..., None, None] + 1)

        if noise_p > 0:
            nz = torch.rand(frame.shape, device=self.device) < noise_p
            nz = nz & ~m
            ncol = torch.randint(1, self.cfg.n_colors + 1, frame.shape, device=self.device)
            frame = torch.where(nz, ncol, frame)
        return frame

    def _rand_place(self, shape, sizes):
        """크기에 맞게 도형이 격자 안에 들어가는 중심 좌표를 뽑는다."""
        g = self.cfg.grid
        rad = self.radii[sizes]
        lo = rad
        hi = (g - 1) - rad
        u = torch.rand(shape + (2,), device=self.device)
        cy = lo + u[..., 0] * (hi - lo)
        cx = lo + u[..., 1] * (hi - lo)
        return cy.round(), cx.round(), rad

    # ---------------------------------------------------------------- 배치
    @torch.no_grad()
    def batch(self, B: int, noise_p: float = 0.008):
        cfg = self.cfg
        dev = self.device
        N, C, Q, n = cfg.n_rooms, cfg.n_combos, cfg.n_queries, cfg.frames_per_room

        # 1) 조합별 방 수를 풀에서 뽑는다
        pick = torch.randint(0, self.count_pool.shape[0], (B,), device=dev)
        counts = self.count_pool[pick]                       # (B, C)

        # 2) 방 -> 조합 배정. 정렬된 multiset 을 무작위 치환한다
        cum = counts.cumsum(-1)                              # (B, C)
        ar = torch.arange(N, device=dev)
        combo_sorted = (ar[None, :, None] >= cum[:, None, :]).sum(-1)   # (B, N)
        perm = torch.rand(B, N, device=dev).argsort(-1)
        room_combo = combo_sorted.gather(1, perm)            # (B, N)

        room_color = room_combo // cfg.n_shapes
        room_shape = room_combo % cfg.n_shapes

        # 3) 방마다 위치·크기
        room_size = torch.randint(0, cfg.n_sizes, (B, N), device=dev)
        cy, cx, rad = self._rand_place((B, N), room_size)

        # 4) 관찰 프레임 n 개 — 같은 방을 n 번 본 것. 잡음만 다르다
        cy_f = cy[:, :, None].expand(B, N, n)
        cx_f = cx[:, :, None].expand(B, N, n)
        rad_f = rad[:, :, None].expand(B, N, n)
        sh_f = room_shape[:, :, None].expand(B, N, n)
        co_f = room_color[:, :, None].expand(B, N, n)
        obs = self._paint(cy_f, cx_f, rad_f, sh_f, co_f, noise_p)   # (B,N,n,g,g)
        obs = obs.reshape(B, N * n, cfg.grid, cfg.grid)

        # 5) 질의 Q 개 — 조합 하나씩. 위치·크기는 무작위(관찰과 무관)
        q_combo = torch.arange(Q, device=dev)[None, :].expand(B, Q) % C
        q_color = q_combo // cfg.n_shapes
        q_shape = q_combo % cfg.n_shapes
        q_size = torch.randint(0, cfg.n_sizes, (B, Q), device=dev)
        qy, qx, qrad = self._rand_place((B, Q), q_size)
        qry = self._paint(qy, qx, qrad, q_shape, q_color, noise_p)   # (B,Q,g,g)

        # 6) 라벨
        hit = (room_combo[:, None, :] == q_combo[:, :, None])        # (B,Q,N)
        pos = (cy.long() * cfg.grid + cx.long())                     # (B,N)

        return dict(
            obs=obs, qry=qry,
            hit=hit, pos=pos, size=room_size,
            room_combo=room_combo, counts=counts,
        )


def p0_report(gen, B=512):
    """P0 — 생성기 검증. 히트 수 분포와 정답 방 위치 분포."""
    b = gen.batch(B)
    hits = b["hit"].sum(-1)                       # (B, Q)
    flat = hits.reshape(-1).float()
    hist = torch.bincount(hits.reshape(-1), minlength=gen.cfg.top_k + 2)
    # 정답 방이 시퀀스 어디에 흩어져 있는지
    idx = torch.arange(gen.cfg.n_rooms, device=hits.device).float()
    room_hist = b["hit"].any(1).float().mean(0)   # (N,)
    return dict(
        hit_min=int(flat.min()), hit_max=int(flat.max()), hit_mean=float(flat.mean()),
        hist=hist.tolist(),
        room_bias=float(room_hist.std() / room_hist.mean().clamp(min=1e-6)),
    )
