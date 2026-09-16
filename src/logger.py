"""사람이 읽는 텍스트 로그.

run_dir/train.log 에 이어쓴다. 재시작해도 같은 파일에 쌓이므로
어디서 끊겼고 어디서 이어졌는지가 한 파일에 남는다.

매 줄 flush 한다 — 크래시로 마지막 줄을 잃으면 끊긴 지점을 모른다.
실시간으로 보려면
    PowerShell : Get-Content runs/routed_n32/train.log -Wait -Tail 20
    Git Bash   : tail -f runs/routed_n32/train.log
"""
import os
import sys
import time
import unicodedata
from datetime import datetime

# Windows 기본 콘솔은 cp949 라 한글·em dash 에서 UnicodeEncodeError 로 죽는다.
# 파일은 항상 utf-8 로 쓰고, 콘솔만 안전하게 바꿔둔다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def hms(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def _w_disp(s: str) -> int:
    """표시 폭. 한글·전각은 2칸을 차지한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """표시 폭 기준 왼쪽 정렬. format 의 :<6s 는 한글을 1칸으로 세서 정렬이 깨진다."""
    return s + " " * max(0, width - _w_disp(s))


class TrainLogger:
    def __init__(self, run_dir: str, echo: bool = True):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "train.log")
        self.echo = echo
        self.t0 = time.time()
        self.total = 0

    # ------------------------------------------------------------ 기본
    def _w(self, kind: str, body: str):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {_pad(kind, 7)} | {body}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        if self.echo:
            print(line, flush=True)

    def event(self, body: str):
        self._w("이벤트", body)

    # ------------------------------------------------------------ 구간
    def start(self, cfg, n_body: int, n_router: int, resumed_from: int = 0):
        self.t0 = time.time()
        self.total = cfg.max_steps
        self._w("", "=" * 96)
        kind = "재시작" if resumed_from else "시작"
        self._w(kind, f"{cfg.run_dir} · routing={cfg.routing} · "
                      f"N={cfg.n_rooms} Q={cfg.n_queries} k={'+'.join(map(str, cfg.ks))} "
                      f"n={cfg.frames_per_room} seq={cfg.seq_obs}"
                      + (" · 정지: 새 히트 0개 라운드" if cfg.adaptive_stop else "")
                      + (f" · 늦은정답 벌점 λ={cfg.late_cost}" if cfg.late_cost else ""))
        self._w("모델", f"d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
                        f"d_key={cfg.d_key} · 본체 {n_body/1e6:.2f}M · "
                        f"라우터 {n_router/1e3:.1f}K ({n_router/(n_body+n_router)*100:.2f}%)")
        self._w("설정", f"batch={cfg.batch_size} lr={cfg.lr:.1e} "
                        f"max_steps={cfg.max_steps} tau={cfg.tau} "
                        f"p_drop={cfg.group_dropout_p} "
                        f"hard전환={int(cfg.hard_switch_frac*cfg.max_steps)}")
        if resumed_from:
            self._w("재개", f"step {resumed_from} 부터 이어서")

    def step(self, step: int, loss: float, ms: float, lr: float, hard: bool):
        done = step / max(1, self.total)
        el = time.time() - self.t0
        eta = ms / 1000 * (self.total - step)
        self._w("step", f"{step:>7,}/{self.total:,} ({done*100:5.1f}%) | "
                        f"loss {loss:7.4f} | {ms:6.1f} ms/it | lr {lr:.2e} | "
                        f"{'hard' if hard else 'soft'} | "
                        f"경과 {hms(el)} | 남음 {hms(eta)}")

    def eval(self, step: int, m: dict):
        g = lambda k, d=float("nan"): m.get(k, d)
        self._w("eval", f"{step:>7,}/{self.total:,} | "
                        f"★완수 {g('task_recall'):.3f} | 전부완수 {g('task_exact'):.3f} | "
                        f"찾기 {g('p3_recall'):.3f} | 읽기 {g('read_frac')*100:.1f}% | "
                        f"라운드 {g('rounds'):.2f} | 재현 {g('pos_acc'):.3f} | "
                        f"히트 {g('hit_acc'):.3f} | P2 {g('p2_key_sep'):.3f} | "
                        f"엔트로피 {g('pool_entropy'):.3f} | cv {g('sel_pos_cv'):.2f}")
        if "bonus_task_exact" in m:
            self._w("보너스", f"{step:>7,}/{self.total:,} | 추론 보너스 켬 — 전부완수 {g('bonus_task_exact'):.4f} | "
                             f"★완수 {g('bonus_task_recall'):.4f} | 읽기 {g('bonus_read_frac')*100:.1f}% | "
                             f"라운드 {g('bonus_rounds'):.2f}")
        if "diagnosis" in m:
            self._w("진단", str(m["diagnosis"]))

    def ckpt(self, step: int, path: str):
        self._w("체크", f"step {step:,} 저장 -> {os.path.basename(path)}")

    def switch(self, step: int):
        self._w("전환", f"soft -> hard (step {step:,})")

    def gate(self, name: str, passed: bool, detail: str = ""):
        self._w("게이트", f"{name} · {'통과' if passed else '실패'}"
                          + (f" · {detail}" if detail else ""))

    def done(self, step: int, ms: float, m: dict = None):
        el = time.time() - self.t0
        self._w("완료", f"{step:,} step · 총 {hms(el)} · {ms:.1f} ms/it 평균")
        if m:
            self._w("최종", " · ".join(
                f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, (int, float))))

    def interrupted(self, step: int):
        self._w("중단", f"step {step:,} 에서 사용자 중단 — 체크포인트 저장 후 종료")

    def fail(self, msg: str):
        self._w("오류", msg)
