"""캐시 라우팅 모델.

구조
    1) 관찰 프레임 T = N*n 개를 인과 어텐션으로 인코딩하고 층별 K,V 를 남긴다.
       방끼리 서로 볼 수 있게 둔다 — 블록 대각으로 막으면 그룹 분리가
       구조상 공짜로 생겨서 검증할 게 없어진다. 분리는 그룹 드롭아웃이 만든다.
    2) 방이 닫히면 어텐션 풀링 -> 투영 -> 정규화로 라우팅 키를 만든다.
    3) 질의 임베딩에서 q 를 만들어 키와 코사인을 재고 top-k 그룹을 고른다.
       읽기 전에 정해지므로 키를 미리 계산해 두는 구조가 성립한다.
    4) 선택된 슬롯마다 그 그룹의 KV n 개만 gather 해서 읽는다.
       슬롯 하나가 그룹 하나를 읽으므로 판정·재현이 그룹에 귀속된다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ RoPE
def rope_cache(seq: int, dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (..., S, H, dh)
    x1, x2 = x.float().chunk(2, dim=-1)
    c = cos[:, None, :].to(x.device)
    s = sin[:, None, :].to(x.device)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


# ------------------------------------------------------------------ 블록
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.dh = H, cfg.d_head
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.q_slot = nn.Linear(d, d, bias=False)   # 슬롯은 q 만 필요하다 (A3)
        self.n2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, d, bias=False)

    def _split(self, t, B, S):
        return t.view(B, S, self.H, self.dh)

    def forward_obs(self, x, cos, sin):
        """관찰 자기어텐션(인과). 반환: x, (k, v) — k,v 는 (B, S, H, dh)"""
        B, S, _ = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = apply_rope(self._split(q, B, S), cos, sin)
        k = apply_rope(self._split(k, B, S), cos, sin)
        v = self._split(v, B, S)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2).reshape(B, S, -1)
        x = x + self.proj(o)
        x = x + self.fc2(F.gelu(self.fc1(self.n2(x))))
        return x, (k, v)

    def forward_slot(self, s, kg, vg):
        """슬롯이 자기 그룹의 KV 만 읽는다.
        s: (M, 1, d) · kg, vg: (M, n, H, dh)   M = B*Q*k
        """
        M = s.shape[0]
        h = self.n1(s)
        # qkv(3d) 를 돌리고 k,v 를 버리면 이 경로 비용의 2/3 가 낭비다.
        # 슬롯은 M = B*Q*k 개라 비용 지배적이므로 전용 투영을 쓴다.
        q = self._split(self.q_slot(h), M, 1)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2)
        ).transpose(1, 2).reshape(M, 1, -1)
        s = s + self.proj(o)
        s = s + self.fc2(F.gelu(self.fc1(self.n2(s))))
        return s


# ------------------------------------------------------------------ 본체
class CacheRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n_sym = cfg.n_colors + 1                      # 0 = 빈칸
        self.embed = nn.Linear(cfg.n_cells * n_sym, d, bias=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(d)

        # 라우터 — 모델 전체에 u 하나
        self.u = nn.Parameter(torch.randn(d) * cfg.probe_init_scale)
        self.W = nn.Linear(d, cfg.d_key, bias=False)
        self.Wq = nn.Linear(d, cfg.d_key, bias=False)

        # 출력 헤드 3종
        self.head_hit = nn.Linear(d, 1)
        self.head_pos = nn.Linear(d, cfg.n_cells)
        self.head_size = nn.Linear(d, cfg.n_sizes)

        # 안 읽은 방의 히트 예측을 점수에서 만든다. 이게 있어야 "정답인데
        # 안 골랐다"가 손실에 잡힌다 — 없으면 놓친 방이 손실에 안 보여서
        # 라우터가 "고른 것에 확신만 키우는" 해로 수렴한다.
        self.route_a = nn.Parameter(torch.tensor(5.0))
        self.route_b = nn.Parameter(torch.tensor(-2.5))

        self.n_sym = n_sym
        self._rope = {}

    # -------------------------------------------------------------- 유틸
    def _rope_for(self, S, device):
        key = (S, device)
        if key not in self._rope:
            self._rope[key] = rope_cache(S, self.cfg.d_head, device)
        return self._rope[key]

    def _embed_frames(self, x):
        """(B, S, g, g) long -> (B, S, d)"""
        oh = F.one_hot(x, self.n_sym).flatten(-3).to(self.embed.weight.dtype)
        return self.embed(oh)

    def _encode_query(self, e_q):
        """질의 프레임을 관찰과 같은 블록에 통과시킨 마지막 층 상태. (B, Q, d)

        각 질의는 자기 자신만 본다 — 캐시를 안 읽으므로 라우팅이 읽기 전에
        정해진다는 구조는 그대로다. 임베딩에 선형 투영만 걸면 위치·크기가
        무작위인 도형의 모양을 못 가른다 (반지름 2 에서 원·삼각형 둘 다 13픽셀).
        실측: q 를 조합 오라클로 바꾸면 recall 0.953 -> 1.000.
        """
        B, Q, d = e_q.shape
        x = e_q.reshape(B * Q, 1, d)
        cos, sin = self._rope_for(1, e_q.device)
        for blk in self.blocks:
            x, _ = blk.forward_obs(x, cos, sin)
        return self.norm_f(x).view(B, Q, d)

    # -------------------------------------------------------------- 순전파
    def forward(self, obs, qry, hard: bool,
                group_drop_mask=None, force_groups=None):
        cfg = self.cfg
        B, T = obs.shape[0], obs.shape[1]
        N, n, Q, k = cfg.n_rooms, cfg.frames_per_room, qry.shape[1], cfg.ks[0]
        dev = obs.device

        # 1) 관찰 인코딩 + 층별 KV
        x = self._embed_frames(obs)
        cos, sin = self._rope_for(T, dev)
        kvs = []
        for blk in self.blocks:
            x, kv = blk.forward_obs(x, cos, sin)
            kvs.append(kv)
        h_obs = self.norm_f(x)                                  # (B, T, d)

        # 2) 어텐션 풀링 -> 라우팅 키
        hg = h_obs.view(B, N, n, -1)
        a = (hg * self.u).sum(-1)                               # (B, N, n)
        pool_w = a.softmax(-1)
        pooled = (hg * pool_w[..., None]).sum(2)                # (B, N, d)
        keys = F.normalize(self.W(pooled), dim=-1)              # (B, N, dk)

        # 3) 질의 -> q -> 점수. 읽기 전에 정해진다
        e_q = self._embed_frames(qry)                           # (B, Q, d)
        q = F.normalize(self.Wq(self._encode_query(e_q)), dim=-1)   # (B, Q, dk)
        score = torch.einsum("bqd,bnd->bqn", q, keys)           # (B, Q, N)

        # 학습 중 그룹 드롭아웃. (B,Q,N) — True 인 그룹은 이 질의에서 안 보인다.
        # 질의별인 이유: Q = 조합 수면 모든 방이 어느 질의의 정답이라
        # 에피소드 단위 마스크는 아무것도 못 지운다.
        if group_drop_mask is not None:
            if group_drop_mask.dim() == 2:                      # (B,N) 도 받아준다
                group_drop_mask = group_drop_mask[:, None, :].expand(B, Q, N)
            # -inf 가 아니라 큰 음수를 쓴다. score 는 hit_full 의 affine 입력으로도
            # 쓰이는데, route_a 가 음수로 가면 -inf 가 +inf 로 뒤집혀 BCE 가 NaN 이 된다.
            # -30 이면 코사인 범위(-1..1) 밖이라 top-k 에 절대 안 뽑히고
            # softmax(-30/0.1) 도 0 이라 실질 효과는 같다.
            score = score.masked_fill(group_drop_mask, -30.0)

        base_score = score
        score_first = score                                     # 1라운드 점수 — 늦게 찾은 정답 벌점이 민다
        gate = (score / cfg.tau).softmax(-1)                    # (B, Q, N)

        # 4) 선택. routing=False 면 모든 그룹을 읽는다 = full-attention 기준선.
        #    출력 구조가 같으므로 라우팅판과 통제된 비교가 된다.
        if not cfg.routing:
            k = N
            sel = torch.arange(N, device=dev)[None, None, :].expand(B, Q, N)
        elif force_groups is not None:                          # P4 오라클
            sel = force_groups.float().topk(k, dim=-1).indices
        else:
            sel = score.topk(k, dim=-1).indices                 # (B, Q, k)

        # 5) 라운드를 돌며 읽는다. 매 라운드 이미 읽은 그룹은 마스킹하고,
        #    다음 라운드의 q 는 이번에 읽은 것으로 보정한다.
        #    adaptive_stop 이면 새 히트가 0개인 라운드에서 멈춘다. bonus_round 면 그런 라운드가 처음 나왔을 때 한 번은 더 읽는다.
        #    bonus_round 는 학습 설정에서는 끄고, 학습이 끝난 모델에 추론 때 켠다 (순위 압력을 학습에서 약하게 만들지 않으려고).
        #    멈춘 질의의 뒤 라운드도 배치로 계산은 하지만 valid=False 라 "읽지 않은 것" 이다 —
        #    출력·손실·지표는 전부 valid 슬롯만 본다.
        H, dh = cfg.n_heads, cfg.d_head
        bi = torch.arange(B, device=dev)[:, None]
        ks = cfg.ks if cfg.routing else (N,)                    # 라운드별 읽기 수
        rounds = len(ks)
        adaptive = cfg.routing and cfg.adaptive_stop and rounds > 1
        read = torch.zeros(B, Q, N, dtype=torch.bool, device=dev)
        active = torch.ones(B, Q, dtype=torch.bool, device=dev)  # 이 라운드를 실제로 읽는 질의
        bonus_left = torch.full((B, Q), int(cfg.bonus_round), dtype=torch.long, device=dev)  # 인내의 보너스 라운드 — 질의당 남은 횟수
        score_aff = score           # 안 읽은 방 affine 입력 = 질의가 마지막으로 실제로 쓴 라운드의 점수
        sels, valids, outs, hits, actives, rids = [], [], [], [], [], []

        for rd, k in enumerate(ks):
            if rd > 0:
                score = base_score.masked_fill(read, -30.0)      # 읽은 건 다시 안 뽑는다
                gate = (score / cfg.tau).softmax(-1)
                sel = score.topk(k, dim=-1).indices
                score_aff = torch.where(active[..., None], score, score_aff)

            valid = active[..., None].expand(B, Q, k)
            if cfg.routing:
                # -30 으로 가려진 방(이미 읽음 · 드롭)이 뽑혔다면 후보가 바닥난 것이다. 읽지 않은 것으로 친다
                valid = valid & (score.gather(-1, sel) > -29.0)
            # scatter 로 쓰면 무효 슬롯이 이미 읽은 방을 False 로 덮는다. OR 로 누적한다
            read = read | torch.zeros_like(read).scatter(-1, sel, valid)
            actives.append(active)
            g_sel = gate.gather(-1, sel)
            if not cfg.routing:
                coef = torch.ones_like(g_sel)
            elif hard:
                coef = 1.0 + g_sel - g_sel.detach()             # STE: 순전파 1
            else:
                coef = k * g_sel / g_sel.sum(-1, keepdim=True).clamp_min(1e-9)

            M = B * Q * k
            sf = sel.reshape(B, Q * k)
            s = e_q[:, :, None, :].expand(B, Q, k, cfg.d_model).reshape(M, 1, -1)
            for blk, (kl, vl) in zip(self.blocks, kvs):
                kg = kl.view(B, N, n, H, dh)[bi, sf].reshape(M, n, H, dh)
                vg = vl.view(B, N, n, H, dh)[bi, sf].reshape(M, n, H, dh)
                s = blk.forward_slot(s, kg, vg)
            s = self.norm_f(s).view(B, Q, k, -1) * coef[..., None]

            # 판정은 라운드마다 낸다 — 정지 규칙이 보는 판정과 출력되는 판정이 같은 값이어야 한다
            hit_r = self.head_hit(s).squeeze(-1)                 # (B, Q, k)

            sels.append(sel)
            valids.append(valid)
            outs.append(s)
            hits.append(hit_r)
            rids.append(torch.full((k,), rd, dtype=torch.long, device=dev))
            if rd + 1 < rounds:
                # 다음 라운드용 q 보정.
                # 전제 — 정답 방들의 키는 코사인으로 서로 뭉쳐 있다. 그래서 이번 라운드에
                # 히트로 판정한 방들의 키 평균 쪽으로 q 를 당기면 아직 못 읽은 정답의 순위가 올라간다.
                # 비정답은 넣지 않는다 — 넣으면 엉뚱한 쪽으로 당긴다.
                # 추론엔 라벨이 없으므로 모델 자신의 판정으로 거르고, 학습도 똑같이 한다.
                # 보정 없이 마스킹만 하면 다음 라운드는 그냥 "다음 k 개" 라 k 를 키운 것과 같다.
                hm = ((hit_r > 0) & valid)[..., None].to(keys.dtype)            # (B, Q, k, 1)
                k_sel = keys[torch.arange(B, device=dev)[:, None, None], sel]   # (B, Q, k, dk)
                k_hit = (k_sel * hm).sum(2) / hm.sum(2).clamp_min(1.0)          # 히트 0개면 0 -> q 그대로
                q = F.normalize(q + k_hit, dim=-1)
                base_score = torch.einsum("bqd,bnd->bqn", q, keys)
                # 드롭된 방은 에피소드 내내 없는 방이다. 점수를 새로 매기면서
                # 마스크가 풀리면 2라운드가 지운 방을 읽고, hit_full 에도 새어 든다.
                if group_drop_mask is not None:
                    base_score = base_score.masked_fill(group_drop_mask, -30.0)
                if adaptive and hard:
                    # 정지 조건 — 이번 라운드에 실제로 읽은 방 중 히트로 판정한 게 하나도 없으면 멈춘다.
                    # 정답 개수를 모른 채 쓸 수 있는 사실만 본다. 후보를 점수 순으로 읽으므로
                    # 한 라운드를 통째로 읽고도 새 히트가 없으면 그 아래에도 없다고 본다.
                    # 학습하는 판단이 아니라서 "멈추지 않기" 를 고를 수 없고, 라운드 벌점도 필요 없다.
                    # soft 구간은 판정이 아직 안 배워졌으므로 전 라운드를 읽는다.
                    zero = ~((hit_r > 0) & valid).any(-1)
                    # 인내의 보너스 라운드 — 0히트 라운드가 처음 나오면 멈추지 않고 한 번은 더 읽는다 (질의당 1번).
                    # 히트가 0개라 q 보정이 없고 읽은 칸은 가려지므로, 이번 라운드 점수의 다음 순위(5~8위)를 읽는다.
                    # 근거 (DEBUG-LOG R5): 순위 실패로 멈춰 놓친 정답의 대부분이 컷 바로 아래였다.
                    use_bonus = active & zero & (bonus_left > 0)
                    bonus_left = bonus_left - use_bonus.long()
                    active = active & (~zero | use_bonus)
                    # 배치 전원이 멈췄으면 남은 라운드는 전부 valid=False 라 출력·손실·지표에 안 들어간다.
                    # 계산만 건너뛴다 — 결과는 똑같고 시간만 준다.
                    if not bool(active.any()):
                        break

        sel = torch.cat(sels, dim=-1)                           # (B, Q, S)  S = sum(ks)
        valid = torch.cat(valids, dim=-1)                       # (B, Q, S)  실제로 읽은 슬롯
        s = torch.cat(outs, dim=2)

        hit_slot = torch.cat(hits, dim=-1)                      # (B, Q, S)

        # 방 N 개 전체에 대한 히트 예측.
        #   읽은 방  -> 실제로 읽고 내린 판정으로 덮어쓴다
        #   안 읽은 방 -> 점수의 affine. 정답인데 점수가 낮으면 손실이 나고,
        #                그 기울기가 score 를 밀어 올린다
        # 무효 슬롯은 덤프 칸 N 에 써서 버린다 — 멈춘 뒤 라운드는 같은 방을 다시 뽑을 수 있어
        # 그대로 scatter 하면 읽은 방의 판정을 덮어쓴다.
        aff = self.route_a * score_aff + self.route_b
        idx = torch.where(valid, sel, N)
        hit_full = torch.cat([aff, aff.new_zeros(B, Q, 1)], dim=-1) \
            .scatter(-1, idx, hit_slot.to(aff.dtype))[..., :N]

        return dict(
            hit_logit=hit_slot,                                 # (B, Q, S)
            hit_full=hit_full,                                  # (B, Q, N)
            pos_logit=self.head_pos(s),                         # (B, Q, S, cells)
            size_logit=self.head_size(s),                       # (B, Q, S, sizes)
            sel=sel, valid=valid, score=score, gate=gate, keys=keys, pool_w=pool_w,
            round_active=torch.stack(actives, dim=-1),          # (B, Q, R')  그 라운드를 실제로 읽었나 (R' = 계산한 라운드)
            round_of_slot=torch.cat(rids),                      # (S,)  슬롯이 몇 번째 라운드인가 (0부터)
            # 늦게 찾은 정답 벌점용. affine 계수는 끊는다 — 벌점이 안 읽은 방 판정의 보정을 휘지 않고
            # 1라운드 점수(질의 인코더 · 키)만 밀게
            late_logit=self.route_a.detach() * score_first + self.route_b.detach(),   # (B, Q, N)
            late_cost=float(cfg.late_cost),
        )

    # -------------------------------------------------------------- 참고
    def n_params(self):
        body = sum(p.numel() for n, p in self.named_parameters()
                   if not n.startswith(("u", "W.", "Wq.")))
        router = sum(p.numel() for n, p in self.named_parameters()
                     if n.startswith(("u", "W.", "Wq.")))
        return body, router
