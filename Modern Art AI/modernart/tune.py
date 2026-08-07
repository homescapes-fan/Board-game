"""自己対戦でヒューリスティックのパラメータを調整する (CEM).

    python3 -m modernart.tune --iters 12 --pop 14 --games 540

各世代で候補パラメータを ``pop`` 個サンプルし、それぞれを「歴代チャンピオンの
プール」と同じ卓で戦わせて強さを測る。上位を残して分布を更新する、を繰り返す。

測定を歪めないための工夫が3つある。

* **人数を混ぜる** — 4人卓だけで調整すると4人にだけ効くパラメータに寄る。
  3〜5人を回して、どの人数でも通用する値を探す。
* **強さ指数で測る** — 期待勝率は人数で変わる (3人=33%, 5人=20%) ので、
  勝ちを人数倍して基準を 1.00 に揃える。
* **交代は再測定で有意差が出たときだけ** — 世代内の最良は定義上「上振れた候補」。
  そのまま採用すると毎回ノイズを掴む。

結果はルールごとに別ファイルへ書き出す (``params/tuned.json`` /
``params/tuned_any_color.json``)。ダブルの2枚目の色制限で最適な打ち方が変わるため。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

from .arena import SEATINGS, make_agent, round_to_seatings
from .params import BOUNDS, TUNED_ANY_PATH, TUNED_PATH, Params

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

#: 強さ指数の基準。これを超えていれば平均より強い
EVEN = 1.0


def _clip(name: str, v: float) -> float:
    lo, hi = BOUNDS[name]
    return min(hi, max(lo, v))


def sample_params(base: Params, mean: dict, sigma: dict, rng: random.Random) -> Params:
    kw = {f: _clip(f, rng.gauss(mean[f], sigma[f])) for f in TUNED_FIELDS}
    return base.with_(**kw)


def _duel(job):
    """挑戦者1人 vs 相手(n-1)人で1ゲーム. 挑戦者の「強さ指数」を返す.

    勝ちを人数倍しているので、実力が同じなら期待値は 1.00 になる。
    """
    from . import rules

    game_id, n, params_by_entry, seed, any_colour, perm_idx = job
    perm = SEATINGS[n][perm_idx % len(SEATINGS[n])]
    agents = [
        make_agent("heuristic", seed + game_id * 97 + seat, params_by_entry[perm[seat]])
        for seat in range(n)
    ]
    rng = random.Random(seed + game_id)
    s = rules.play_out(rules.new_game(n, rng, double_any_artist=any_colour), agents, rng)

    hero_seat = perm.index(0)
    best = max(s.money)
    tied = sum(1 for m in s.money if m == best)
    win = (1.0 / tied) if s.money[hero_seat] == best else 0.0
    return win * n


def evaluate(
    challenger: Params,
    opponents: list,
    player_counts: list[int],
    games: int,
    seed: int,
    workers,
    any_colour: bool = False,
) -> tuple[float, float]:
    """``(強さ指数, その標準誤差)`` を返す. 1.00 が「相手と同じ強さ」.

    ``opponents`` は歴代チャンピオンのプール。1人の相手にだけ効く小細工を掴まない
    よう、卓の相手はここから選ぶ。誰と当たるかは game_id だけで決まるので、
    候補どうしの比較は同じ条件になる。
    """
    jobs = []
    gid = 0
    for n in player_counts:
        # 席順が均等になるよう、人数ごとに n! の倍数だけ回す
        count = round_to_seatings(max(1, games // len(player_counts)), n)
        for i in range(count):
            seats = [challenger] + [opponents[(gid + j) % len(opponents)] for j in range(n - 1)]
            jobs.append((gid, n, seats, seed, any_colour, i))
            gid += 1

    if workers:
        scores = workers.map(_duel, jobs, chunksize=8)
    else:
        scores = [_duel(j) for j in jobs]

    n_games = len(scores)
    mean = sum(scores) / n_games
    var = sum((x - mean) ** 2 for x in scores) / n_games
    return mean, math.sqrt(var / n_games)


def cem(
    iters: int = 12,
    pop: int = 14,
    elite_frac: float = 0.3,
    games: int = 540,
    player_counts: tuple[int, ...] = (3, 4, 5),
    seed: int = 2024,
    jobs: int = 0,
    start: Params | None = None,
    pool_size: int = 5,
    any_colour: bool = False,
):
    rng = random.Random(seed)
    champion = start or Params()
    mean = {f: getattr(champion, f) for f in TUNED_FIELDS}
    sigma = {f: max(0.06, (BOUNDS[f][1] - BOUNDS[f][0]) * 0.18) for f in TUNED_FIELDS}
    n_elite = max(2, int(pop * elite_frac))
    jobs = jobs or max(1, min(8, os.cpu_count() or 1))
    counts = list(player_counts)

    #: 歴代チャンピオン。最新だけを相手にすると「その相手にだけ効く手」に寄ってしまう
    gauntlet = [champion]
    #: 動かさない基準。プールだけで判定すると、プールごと弱い方へ流れていっても
    #: 「プールには勝っている」ので改善に見えてしまう（自己対戦の崩壊）。
    #: 出発点を固定の物差しとして残し、これにも勝てなければ採用しない。
    anchor = champion

    workers = Pool(jobs) if jobs > 1 else None
    try:
        rule = "ダブルの2枚目はどの色でも可" if any_colour else "ダブルの2枚目は同じ色のみ"
        print(f"ルール: {rule} / 人数: {'・'.join(map(str, counts))}人を混ぜる")
        m, se = evaluate(champion, gauntlet, counts, games, seed, workers, any_colour)
        print(f"基準（同じパラメータ同士）: 強さ {m:.3f} ± {1.96 * se:.3f}  ※ 1.000 が互角\n")

        for it in range(1, iters + 1):
            t0 = time.time()
            cands = [sample_params(champion, mean, sigma, rng) for _ in range(pop)]
            scored = []
            for i, c in enumerate(cands):
                s, _ = evaluate(
                    c, gauntlet, counts, games, seed + 1000 * it + i, workers, any_colour
                )
                scored.append((s, c))
            scored.sort(key=lambda x: -x[0])
            elite = [c for _, c in scored[:n_elite]]

            for f in TUNED_FIELDS:
                vals = [getattr(c, f) for c in elite]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / len(vals)
                mean[f] = mu
                # 収縮しすぎないよう下限を残す
                sigma[f] = max(math.sqrt(var), (BOUNDS[f][1] - BOUNDS[f][0]) * 0.04)

            best_score, best = scored[0]
            print(
                f"世代 {it:>2}/{iters}  最良 {best_score:.3f}  "
                f"上位{n_elite}件で分布を更新  ({time.time() - t0:.0f}秒)"
            )
            # 世代内の最良は「たまたま上振れた候補」なので、必ず別サンプルで測り直す。
            # さらに、動かない基準にも勝てることを確かめる。
            if best_score > EVEN:
                m2, se2 = evaluate(
                    best, gauntlet, counts, games * 3, seed + 77 * it, workers, any_colour
                )
                print(f"     再測定 プール {m2:.3f} ± {1.96 * se2:.3f}", end="")
                if m2 - 2 * se2 <= EVEN:
                    print(" … 有意差なし。据え置き")
                    continue

                m3, se3 = evaluate(
                    best, [anchor], counts, games * 3, seed + 55 * it, workers, any_colour
                )
                print(f" / 基準 {m3:.3f} ± {1.96 * se3:.3f}", end="")
                if m3 - 2 * se3 <= EVEN:
                    print(" … 基準に勝てていない。据え置き")
                    continue

                champion = best
                mean = {f: getattr(champion, f) for f in TUNED_FIELDS}
                gauntlet.append(best)
                if len(gauntlet) > pool_size:
                    gauntlet.pop(1)  # 最初の1つ（出発点）は残しておく
                print(f" … チャンピオン交代（プール {len(gauntlet)} 体）")

        # 最終確認は必ず「動かない基準」に対して行う。プール相手だと、
        # プールごと弱くなっていた場合に良い数字が出てしまう
        m, se = evaluate(champion, [anchor], counts, games * 4, seed + 999, workers, any_colour)
        print(f"\n最終確認（出発点のパラメータ相手）: 強さ {m:.3f} ± {1.96 * se:.3f}")
        if m - 2 * se <= EVEN:
            print("  ※ 出発点を有意に上回っていません。書き出しても改善にならない可能性があります")
        for n in counts:
            mn, sn = evaluate(champion, [anchor], [n], games, seed + 31 * n, workers, any_colour)
            print(f"    {n}人: {mn:.3f} ± {1.96 * sn:.3f}")
    finally:
        if workers:
            workers.close()
            workers.join()
    return champion, m, se


def main(argv=None):
    ap = argparse.ArgumentParser(description="自己対戦でパラメータを調整する")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--games", type=int, default=540, help="1候補あたりの対戦数（人数で割って使う）")
    ap.add_argument(
        "--players", default="3,4,5",
        help="調整に使う人数。カンマ区切り。既定は3〜5人を混ぜる",
    )
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("-j", "--jobs", type=int, default=0)
    ap.add_argument("--from-tuned", action="store_true", help="既存の調整済みパラメータから始める")
    ap.add_argument("--out", default=None, help="書き出し先 (既定: ルールに応じたファイル)")
    ap.add_argument(
        "--double-any-color", action="store_true",
        help="ダブルの2枚目に別の色も出せるルールで調整する",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="強くなっていなくても書き出す（通常は使わない）",
    )
    args = ap.parse_args(argv)

    counts = tuple(int(x) for x in args.players.split(",") if x.strip())
    for n in counts:
        if n not in SEATINGS:
            ap.error(f"人数は3〜5です: {n}")

    start = Params.load_for_rule(args.double_any_color) if args.from_tuned else Params()
    best, strength, stderr = cem(
        iters=args.iters,
        pop=args.pop,
        games=args.games,
        player_counts=counts,
        seed=args.seed,
        jobs=args.jobs,
        start=start,
        any_colour=args.double_any_color,
    )

    default_path = TUNED_ANY_PATH if args.double_any_color else TUNED_PATH
    path = Path(args.out) if args.out else default_path

    # 強くなったときだけ採用する。自己対戦は「プールごと弱くなる」形で
    # 崩れることがあり、無条件に上書きすると弱いパラメータが入り込む
    improved = strength - 2 * stderr > EVEN
    if not (improved or args.force):
        print(f"\n出発点を有意に上回らなかったので、{path} は更新しません。")
        print("  （強くなっていない値で上書きしないための安全装置です）")
        print("  どうしても書き出したい場合は --force を付けてください。")
        return 1

    best.save(path)
    print(f"\n{path} に保存しました" + ("" if improved else "（--force 指定）"))
    for f in TUNED_FIELDS:
        print(f"  {f:<20} {getattr(best, f):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
