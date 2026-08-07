"""自己対戦でヒューリスティックのパラメータを調整する (CEM).

    python3 -m modernart.tune --iters 12 --pop 16 --games 480

勝率の測定は毎回ノイズを含むので、世代内の最良をそのまま採用すると「たまたま
上振れた候補」を掴む。交代は独立した再測定で有意差が出たときだけにしている。

各世代で候補パラメータを ``pop`` 個サンプルし、それぞれを「現行チャンピオン」と
同じ卓で戦わせて勝率を測る。上位を残して分布を更新する、を繰り返す。

結果は ``params/tuned.json`` に書き出され、以降 ``Params.load_tuned()`` が拾う。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from multiprocessing import Pool

from .arena import SEATINGS, make_agent
from .params import BOUNDS, Params

# 調整対象。value_noise は「相手モデルのばらつき」で強さの指標ではないので固定する
TUNED_FIELDS = [
    "gamma",
    "shade_open",
    "shade_once",
    "shade_sealed",
    "shade_fixed",
    "once_seller_bonus",
    "risk",
    "future_weight",
    "fixed_margin",
    "second_min_revenue",
    "cash_weight_r1",
    "cash_weight_r2",
    "cash_weight_r3",
    "cash_weight_r4",
]


def _clip(name: str, v: float) -> float:
    lo, hi = BOUNDS[name]
    return min(hi, max(lo, v))


def sample_params(base: Params, mean: dict, sigma: dict, rng: random.Random) -> Params:
    kw = {f: _clip(f, rng.gauss(mean[f], sigma[f])) for f in TUNED_FIELDS}
    return base.with_(**kw)


def _duel(job):
    """挑戦者1人 vs 相手(n-1)人で1ゲーム. 挑戦者の (勝ちポイント, 所持金) を返す."""
    from . import rules

    game_id, n, params_by_entry, seed = job
    perm = SEATINGS[n][game_id % len(SEATINGS[n])]
    agents = [
        make_agent("heuristic", seed + game_id * 97 + seat, params_by_entry[perm[seat]])
        for seat in range(n)
    ]
    s = rules.play_game(agents, random.Random(seed + game_id))
    hero_seat = perm.index(0)
    best = max(s.money)
    tied = sum(1 for m in s.money if m == best)
    win = (1.0 / tied) if s.money[hero_seat] == best else 0.0
    return win, s.money[hero_seat]


def evaluate(challenger: Params, opponents: list, n: int, games: int, seed: int, workers) -> float:
    """挑戦者の勝率を返す. 期待値は 1/n.

    ``opponents`` は歴代チャンピオンのプール。1人の相手にだけ効く小細工を掴まない
    よう、卓の相手はここから選ぶ。誰と当たるかは game_id だけで決まるので、
    候補どうしの比較は同じ条件になる。
    """
    k = len(SEATINGS[n])
    games = max(k, ((games + k - 1) // k) * k)
    jobs = []
    for g in range(games):
        seats = [challenger] + [opponents[(g + i) % len(opponents)] for i in range(n - 1)]
        jobs.append((g, n, seats, seed))
    results = workers.map(_duel, jobs, chunksize=8) if workers else [_duel(j) for j in jobs]
    return sum(w for w, _ in results) / games


def stderr_of(rate: float, games: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 1e-9) / games)


def cem(
    iters: int = 12,
    pop: int = 16,
    elite_frac: float = 0.3,
    games: int = 480,
    n_players: int = 4,
    seed: int = 2024,
    jobs: int = 0,
    start: Params | None = None,
    pool_size: int = 5,
):
    rng = random.Random(seed)
    champion = start or Params()
    mean = {f: getattr(champion, f) for f in TUNED_FIELDS}
    sigma = {f: max(0.06, (BOUNDS[f][1] - BOUNDS[f][0]) * 0.18) for f in TUNED_FIELDS}
    n_elite = max(2, int(pop * elite_frac))
    jobs = jobs or max(1, min(8, os.cpu_count() or 1))

    #: 歴代チャンピオン。最新だけを相手にすると「その相手にだけ効く手」に寄ってしまう
    gauntlet = [champion]

    workers = Pool(jobs) if jobs > 1 else None
    try:
        base = 1.0 / n_players
        sanity = evaluate(champion, gauntlet, n_players, games, seed, workers)
        print(
            f"基準（同じパラメータ同士）: {sanity:.1%} ± {1.96 * stderr_of(sanity, games):.1%}"
            f"  ※理論値 {base:.1%}"
        )

        for it in range(1, iters + 1):
            t0 = time.time()
            cands = [sample_params(champion, mean, sigma, rng) for _ in range(pop)]
            scored = []
            for i, c in enumerate(cands):
                rate = evaluate(c, gauntlet, n_players, games, seed + 1000 * it + i, workers)
                scored.append((rate, c))
            scored.sort(key=lambda x: -x[0])
            elite = [c for _, c in scored[:n_elite]]

            for f in TUNED_FIELDS:
                vals = [getattr(c, f) for c in elite]
                m = sum(vals) / len(vals)
                var = sum((v - m) ** 2 for v in vals) / len(vals)
                mean[f] = m
                # 収縮しすぎないよう下限を残す
                sigma[f] = max(math.sqrt(var), (BOUNDS[f][1] - BOUNDS[f][0]) * 0.04)

            best_rate, best = scored[0]
            print(
                f"世代 {it:>2}/{iters}  最良 {best_rate:.1%}  "
                f"上位{n_elite}件で分布を更新  ({time.time() - t0:.0f}秒)"
            )
            # 世代内の最良は「たまたま上振れた候補」なので、必ず別サンプルで測り直す。
            # 交代は、独立した再測定が誤差2つぶんを超えて基準を上回ったときだけ。
            if best_rate > base + 2 * stderr_of(best_rate, games):
                trials = games * 4
                confirm = evaluate(best, gauntlet, n_players, trials, seed + 77 * it, workers)
                margin = 2 * stderr_of(confirm, trials)
                print(f"     再測定 {confirm:.1%} ± {1.96 * stderr_of(confirm, trials):.1%} …", end=" ")
                if confirm - margin > base:
                    champion = best
                    mean = {f: getattr(champion, f) for f in TUNED_FIELDS}
                    gauntlet.append(best)
                    if len(gauntlet) > pool_size:
                        gauntlet.pop(1)  # 最初の1つ（既定値）は残しておく
                    print(f"チャンピオン交代（プール {len(gauntlet)} 体）")
                else:
                    print("有意差なし。据え置き")

        final = evaluate(champion, gauntlet, n_players, games * 4, seed + 999, workers)
        print(f"\n最終確認（歴代プール相手）: {final:.1%} ± {1.96 * stderr_of(final, games * 4):.1%}")
    finally:
        if workers:
            workers.close()
            workers.join()
    return champion


def main(argv=None):
    ap = argparse.ArgumentParser(description="自己対戦でパラメータを調整する")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--games", type=int, default=480, help="1候補あたりの対戦数")
    ap.add_argument("--players", type=int, default=4, choices=(3, 4, 5))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("-j", "--jobs", type=int, default=0)
    ap.add_argument("--from-tuned", action="store_true", help="既存の調整済みパラメータから始める")
    ap.add_argument("--out", default=None, help="書き出し先 (既定: params/tuned.json)")
    args = ap.parse_args(argv)

    start = Params.load_tuned() if args.from_tuned else Params()
    best = cem(
        iters=args.iters,
        pop=args.pop,
        games=args.games,
        n_players=args.players,
        seed=args.seed,
        jobs=args.jobs,
        start=start,
    )
    from pathlib import Path

    from .params import TUNED_PATH

    path = Path(args.out) if args.out else TUNED_PATH
    best.save(path)
    print(f"\n{path} に保存しました")
    for f in TUNED_FIELDS:
        print(f"  {f:<20} {getattr(best, f):.3f}")


if __name__ == "__main__":
    main()
