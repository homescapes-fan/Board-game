"""自己対戦でエージェントの強さを比べる.

    python3 -m modernart.arena --agents heuristic,heuristic,greedy,random -n 400

席順の有利不利を消すため、ゲームごとにエージェントの座席を回す。
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from multiprocessing import Pool

from .params import Params
from .state import PHASE_AUCTION, PHASE_GAME_END

AGENT_NAMES = ("random", "greedy", "heuristic", "pimc")

#: 席順の全順列。単に回すだけだと「誰の左隣に座るか」が固定されてしまい、
#: 同じエージェント同士でも勝率に差が出る (モダンアートは隣人が誰かで有利不利が変わる)。
SEATINGS = {n: list(itertools.permutations(range(n))) for n in (3, 4, 5)}


def _params_from(source: str) -> Params:
    """"default" / "tuned" / json のパス からパラメータを読む."""
    if source == "default":
        return Params()
    if source == "tuned":
        return Params.load_tuned()
    from pathlib import Path

    return Params.load_tuned(Path(source))


def make_agent(spec: str, seed: int, params: Params | None = None, opts: dict | None = None):
    """名前からエージェントを作る. ワーカープロセス側で呼ぶ."""
    from .agents.heuristic import GreedyAgent, HeuristicAgent
    from .agents.random_agent import RandomAgent

    rng = random.Random(seed)
    opts = opts or {}
    head, _, tail = spec.partition(":")
    if head in ("heuristic", "greedy") and tail:
        params = _params_from(tail)  # "heuristic:default" / "heuristic:tuned" / "heuristic:path.json"
    P = params or Params.load_tuned()

    if spec == "random":
        return RandomAgent(rng, name="random")
    if head == "greedy":
        return GreedyAgent(P, rng, name=spec, seed=seed)
    if head == "heuristic":
        return HeuristicAgent(P, rng, name=spec, seed=seed)
    if head == "pimc":
        from .agents.pimc import PIMCAgent

        # "pimc" / "pimc:greedy" / "pimc:heuristic:0.3" (ロールアウト方策と秒数を指定できる)
        parts = spec.split(":")
        rollout = parts[1] if len(parts) > 1 else opts.get("pimc_rollout", "heuristic")
        budget = float(parts[2]) if len(parts) > 2 else opts.get("pimc_budget", 0.15)
        return PIMCAgent(P, rng, name=spec, seed=seed, budget=budget, rollout=rollout)
    raise ValueError(f"未知のエージェント: {spec} (使えるのは {', '.join(AGENT_NAMES)})")


def _play_one(job):
    """1ゲーム遊んで、各エントリの (順位ポイント, 最終所持金) を返す."""
    from . import rules

    game_id, specs, base_seed, params, verify, opts = job
    any_colour = bool(opts.get("double_any_artist"))
    n = len(specs)
    perm = SEATINGS[n][game_id % len(SEATINGS[n])]  # perm[座席] = specs のインデックス
    seats = [specs[perm[i]] for i in range(n)]
    agents = [
        make_agent(sp, base_seed + game_id * 97 + i, params, opts) for i, sp in enumerate(seats)
    ]
    rng = random.Random(base_seed + game_id)

    s = rules.new_game(n, rng, double_any_artist=any_colour)
    if verify:
        while s.phase != PHASE_GAME_END:
            if s.phase == PHASE_AUCTION:
                before, seller = sum(s.money), s.seller
                w, price = rules.resolve_auction(s, agents, rng)
                s.apply_auction_result(w, price)
                assert sum(s.money) == before - (price if w == seller else 0), "収支が合わない"
            else:
                rules.step(s, agents, rng)
    else:
        rules.play_out(s, agents, rng)

    best = max(s.money)
    tied = sum(1 for m in s.money if m == best)
    out = [(0.0, 0)] * n
    for seat in range(n):
        win = (1.0 / tied) if s.money[seat] == best else 0.0
        out[perm[seat]] = (win, s.money[seat])
    return out


def round_to_seatings(n_games: int, n: int) -> int:
    """席順が均等になるよう、ゲーム数を n! の倍数に切り上げる."""
    k = len(SEATINGS[n])
    return max(k, ((n_games + k - 1) // k) * k)


def run(specs, n_games=200, seed=12345, jobs=0, params=None, verify=False, progress=True, opts=None):
    n = len(specs)
    n_games = round_to_seatings(n_games, n)
    jobs = jobs or 1
    joblist = [(g, specs, seed, params, verify, opts or {}) for g in range(n_games)]

    wins = [0.0] * n
    money = [0] * n
    t0 = time.time()
    if jobs > 1:
        with Pool(jobs) as pool:
            results = pool.imap_unordered(_play_one, joblist, chunksize=4)
            for done, res in enumerate(results, 1):
                for i, (w, m) in enumerate(res):
                    wins[i] += w
                    money[i] += m
                if progress and done % 25 == 0:
                    print(f"\r  {done}/{n_games} ゲーム", end="", flush=True)
    else:
        for done, job in enumerate(joblist, 1):
            for i, (w, m) in enumerate(_play_one(job)):
                wins[i] += w
                money[i] += m
            if progress and done % 25 == 0:
                print(f"\r  {done}/{n_games} ゲーム", end="", flush=True)
    if progress:
        print(f"\r  {n_games} ゲーム完了 ({time.time() - t0:.1f}秒)")
    return wins, money, n_games


def report(specs, wins, money, n_games):
    n = len(specs)
    baseline = 1.0 / n
    print(f"\n{n}人 × {n_games}ゲーム (期待勝率 {baseline:.1%})")
    print(f"{'エージェント':<14}{'勝率':>12}{'±95%':>9}{'平均所持金':>12}")
    print("-" * 47)
    order = sorted(range(n), key=lambda i: -wins[i])
    for i in order:
        rate = wins[i] / n_games
        se = math.sqrt(max(rate * (1 - rate), 1e-9) / n_games)
        print(f"{specs[i]:<14}{rate:>11.1%}{1.96 * se:>9.1%}{money[i] / n_games:>12.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="モダンアート エージェント対戦")
    ap.add_argument(
        "--agents",
        default="heuristic,heuristic,greedy,random",
        help="カンマ区切り (3〜5個): " + ", ".join(AGENT_NAMES),
    )
    ap.add_argument("-n", "--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("-j", "--jobs", type=int, default=0, help="0 で自動")
    ap.add_argument("--verify", action="store_true", help="毎局 収支の不変条件を検査する")
    ap.add_argument(
        "--double-any-color", action="store_true",
        help="ダブルの2枚目に別の色も出せるルールで対戦する",
    )
    args = ap.parse_args(argv)

    specs = [x.strip() for x in args.agents.split(",") if x.strip()]
    if not 3 <= len(specs) <= 5:
        ap.error("エージェントは3〜5個指定してください")

    jobs = args.jobs or min(8, __import__("os").cpu_count() or 1)
    opts = {"double_any_artist": args.double_any_color}
    wins, money, played = run(specs, args.games, args.seed, jobs, verify=args.verify, opts=opts)
    report(specs, wins, money, played)


if __name__ == "__main__":
    main()
