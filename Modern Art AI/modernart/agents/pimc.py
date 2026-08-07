"""PIMC (Perfect Information Monte Carlo) による着手選択.

やっていることは単純で、

  1. 観測と矛盾しない相手の手札を1通りサンプルする (determinization)
  2. その完全情報の局面で、候補手それぞれを指してみる
  3. 以降はヒューリスティック方策で終局まで進め、結果を採点する
  4. 1〜3 を時間いっぱい繰り返し、平均が一番良い手を選ぶ

同じサンプルで全候補を試す (common random numbers) ので、候補どうしの比較の
分散が小さい。生き残っている候補には常に同じ回数を割り当て、「最良との差が
誤差では説明できない」候補だけを打ち切る。回数を揃えないと、たまたま少ない
サンプルで良い数字が出ただけの候補が上位に来てしまう。
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass

from .. import rules
from ..beliefs import determinize
from ..cards import SEALED
from ..params import Params
from ..state import PHASE_GAME_END, GameState
from .heuristic import GreedyAgent, HeuristicAgent

# 決定の種類
PLAY = "play"
SECOND = "second"
BID = "bid"  # 公開競り / 一声 の留保価格
SEALED_BID = "sealed"
PRICE = "price"  # 差し値の提示額
ACCEPT = "accept"  # 差し値を買うか


@dataclass
class OptionStat:
    option: object
    label: str
    n: int = 0
    score: float = 0.0  # 勝ち寄りの評価 (0〜1)
    money: float = 0.0  # 自分の所持金 − 2位の所持金
    wins: float = 0.0
    #: 基準手との「同じサンプル上での差」。候補同士の優劣はこちらで判定する
    diff: float = 0.0
    diff_sq: float = 0.0

    @property
    def mean_score(self) -> float:
        return self.score / self.n if self.n else 0.0

    @property
    def mean_money(self) -> float:
        return self.money / self.n if self.n else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def stderr(self) -> float:
        """平均そのものの誤差 (表示用)."""
        if self.n < 2:
            return 1.0
        m = self.mean_score
        return math.sqrt(max(m * (1 - m), 1e-9) / self.n)

    @property
    def mean_diff(self) -> float:
        return self.diff / self.n if self.n else 0.0

    @property
    def diff_stderr(self) -> float:
        """基準手との差の誤差.

        同じ determinization で両方を試しているので、差の分散は個々の分散より
        ずっと小さい。ここを「平均どうしの誤差の和」で代用すると閾値が過大になり、
        探索が何も上書きできなくなる。
        """
        if self.n < 2:
            return 1.0
        m = self.mean_diff
        var = self.diff_sq / self.n - m * m
        return math.sqrt(max(var, 0.0) / self.n) if var > 0 else 0.0


class _Override:
    """1回の競りだけ、自分の行動を固定値に差し替えるラッパ."""

    __slots__ = ("base", "kind", "value", "name")

    def __init__(self, base, kind: str, value):
        self.base, self.kind, self.value, self.name = base, kind, value, "override"

    def choose_play(self, s, p):
        return self.base.choose_play(s, p)

    def choose_second(self, s, p):
        return self.base.choose_second(s, p)

    def reservation(self, s, p, lot, lot_type):
        return self.value if self.kind == BID else self.base.reservation(s, p, lot, lot_type)

    def sealed_bid(self, s, p, lot):
        return self.value if self.kind == SEALED_BID else self.base.sealed_bid(s, p, lot)

    def fixed_price(self, s, p, lot):
        return self.value if self.kind == PRICE else self.base.fixed_price(s, p, lot)

    def fixed_accept(self, s, p, lot, price):
        if self.kind == ACCEPT:
            return bool(self.value)
        return self.base.fixed_accept(s, p, lot, price)


def make_rollout_agents(spec: str, params: Params, n: int, seed: int) -> list:
    cls = HeuristicAgent if spec == "heuristic" else GreedyAgent
    return [cls(params, random.Random(seed + 7919 * i), seed=seed + i) for i in range(n)]


def terminal_score(s: GameState, me: int, scale: float) -> tuple[float, float, float]:
    """終局の評価. ``(勝ち寄りのスコア, 2位との金差, 勝ったか)``."""
    mine = s.money[me]
    best_other = max(s.money[j] for j in range(s.n) if j != me)
    diff = mine - best_other
    score = 1.0 / (1.0 + math.exp(-diff / scale))
    if diff > 0:
        win = 1.0
    elif diff < 0:
        win = 0.0
    else:
        win = 1.0 / sum(1 for j in range(s.n) if s.money[j] == mine)
    return score, float(diff), win


def _apply_option(t: GameState, me: int, kind: str, option, agents, rng) -> None:
    if kind == PLAY:
        t.apply_play(option)
    elif kind == SECOND:
        if option is None:
            t.apply_decline_second()
        else:
            t.apply_second(option)
    else:
        override = list(agents)
        override[me] = _Override(agents[me], kind, option)
        winner, price = rules.resolve_auction(t, override, rng)
        t.apply_auction_result(winner, price)


def evaluate_options(job):
    """1ワーカー分の評価.

    候補ごとに ``[スコア和, 金差和, 勝ち数, 回数, 基準手との差の和, 差の二乗和]`` を返す。
    1つの determinization で全候補を試すので、候補どうしの差は同じ配牌の上で測れる。
    """
    (
        state,
        me,
        kind,
        options,
        params,
        seed,
        n_dets,
        rollout_spec,
        weights,
        scale,
        base_index,
    ) = job
    rng = random.Random(seed)
    agents = make_rollout_agents(rollout_spec, params, state.n, seed)
    totals = [[0.0, 0.0, 0.0, 0, 0.0, 0.0] for _ in options]
    scores = [0.0] * len(options)

    # 候補は自分の手札から作られ、determinize は自分の手札をそのまま残すので、
    # ここで非合法手が出てきたら呼び出し側のバグ。握りつぶさず落とす。
    for _ in range(n_dets):
        base = determinize(state, me, rng, weights)
        for i, option in enumerate(options):
            t = base.clone()
            _apply_option(t, me, kind, option, agents, rng)
            guard = 0
            while t.phase != PHASE_GAME_END:
                rules.step(t, agents, rng)
                guard += 1
                if guard > 4000:  # 正常なゲームは数百手で終わる。無限ループはバグ
                    raise RuntimeError("ロールアウトが終局しませんでした")
            sc, money_diff, win = terminal_score(t, me, scale)
            scores[i] = sc
            acc = totals[i]
            acc[0] += sc
            acc[1] += money_diff
            acc[2] += win
            acc[3] += 1
        if 0 <= base_index < len(options):
            ref = scores[base_index]
            for i in range(len(options)):
                d = scores[i] - ref
                totals[i][4] += d
                totals[i][5] += d * d
    return totals


class PIMCAgent:
    """Agent プロトコルを満たしつつ、内部で PIMC 探索を回す."""

    def __init__(
        self,
        params: Params | None = None,
        rng: random.Random | None = None,
        name: str = "pimc",
        seed: int = 0,
        budget: float = 0.4,
        rollout: str = "heuristic",
        jobs: int = 1,
        scale: float = 40.0,
        min_dets: int = 8,
        weights=None,
        pool=None,
        prune_after: int = 48,
        prune_sigmas: float = 2.5,
        min_alive: int = 3,
        tie_sigmas: float = 1.0,
    ):
        self.P = params or Params()
        self.rng = rng or random.Random(seed)
        self.name = name
        self.seed = seed
        self.budget = budget
        self.rollout = rollout
        self.jobs = max(1, jobs)
        self.scale = scale
        self.min_dets = min_dets
        self.weights = weights
        self.pool = pool
        #: 枝刈りの条件。全候補がこの回数を超えるまでは誰も落とさない
        self.prune_after = prune_after
        #: 「最良との差が誤差 prune_sigmas 個ぶんより大きい」候補だけ落とす
        self.prune_sigmas = prune_sigmas
        self.min_alive = min_alive
        #: 探索の1位とヒューリスティックの手の差がこの誤差以内なら後者を採る
        self.tie_sigmas = tie_sigmas
        #: 探索が使えない場面のフォールバック兼、候補列挙のための評価器
        self.fallback = HeuristicAgent(self.P, self.rng, name="fallback", seed=seed)
        self.last_stats: list[OptionStat] = []
        #: 直近の探索で、差が誤差の範囲だったので評価関数の手を採ったか
        self.last_prior_promoted = False

    # ------------------------------------------------------------------ 探索

    def search(
        self,
        s: GameState,
        me: int,
        kind: str,
        options: list,
        labels=None,
        budget=None,
        prior=None,
    ):
        """候補を評価して ``list[OptionStat]`` を良い順に返す.

        生き残っている候補には常に同じ回数を割り当てる。こうしないと、たまたま
        少ないサンプルで良い数字が出ただけの候補が上位に来てしまう。
        「勝ち目がないと言い切れる」候補だけを打ち切って時間を回す。

        ``prior`` はヒューリスティックが選ぶ手。最良との差が誤差で説明できる範囲なら
        こちらを採る。9択の平均値をそのまま比べると必ずどれかが上振れて1位になり、
        探索がヒューリスティックより悪くなることがあるため。
        """
        labels = labels or [str(o) for o in options]
        stats = [OptionStat(o, lab) for o, lab in zip(options, labels)]
        self.last_prior_promoted = False
        if len(options) <= 1:
            self.last_stats = stats
            return stats

        prior_index = -1
        if prior is not None:
            for i, o in enumerate(options):
                if o == prior:
                    prior_index = i
                    break

        budget = self.budget if budget is None else budget
        deadline = time.time() + budget
        alive = list(range(len(options)))
        # 1バッチは途中で止められないので、最初は最小限だけ回して速度を測る。
        # 並列時はワーカーを遊ばせないよう jobs 個ぶんは確保する。
        batch = max(1, self.jobs)
        sec_per_det = 0.0

        while len(alive) > 1:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if sec_per_det > 0:
                affordable = int(remaining / sec_per_det)
                if affordable < 1:
                    break
                batch = min(batch, affordable)

            base_slot = alive.index(prior_index) if prior_index in alive else -1
            t0 = time.time()
            results = self._run_batch(
                s, me, kind, [options[i] for i in alive], batch, base_slot
            )
            if results is None:
                break
            spent = time.time() - t0
            for slot, acc in enumerate(results):
                st = stats[alive[slot]]
                st.score += acc[0]
                st.money += acc[1]
                st.wins += acc[2]
                st.n += acc[3]
                st.diff += acc[4]
                st.diff_sq += acc[5]

            done = max(1, max(acc[3] for acc in results))
            sec_per_det = spent / done
            alive = self._prune(stats, alive, prior_index)
            batch = min(batch * 2, 512)

        # 生き残りを先に、それぞれ平均の良い順。打ち切った候補はその下に回す
        alive_set = set(alive)
        order = sorted(
            range(len(stats)),
            key=lambda i: (0 if i in alive_set else 1, -stats[i].mean_score),
        )
        ranked = [stats[i] for i in order]
        self.last_prior_promoted = False
        if prior is not None and ranked[0].option != prior:
            promoted = self._prefer_prior(ranked, prior)
            self.last_prior_promoted = promoted is not ranked
            ranked = promoted
        self.last_stats = ranked
        return ranked

    def _prefer_prior(self, ranked: list[OptionStat], prior) -> list[OptionStat]:
        """差が誤差の範囲なら、ヒューリスティックの手を先頭に持ってくる."""
        best = ranked[0]
        if best.option == prior:
            return ranked
        for i, st in enumerate(ranked):
            if st.option != prior or st.n < 2:
                continue
            # 同じ配牌の上で測った差なので、その差の誤差で判定する
            gain = best.mean_diff - st.mean_diff
            noise = self.tie_sigmas * max(best.diff_stderr, 1e-9)
            if gain <= noise:
                return [st] + ranked[:i] + ranked[i + 1 :]
            break
        return ranked

    def _prune(self, stats, alive: list[int], keep: int = -1) -> list[int]:
        """明らかに劣る候補だけ落とす. 残りは常に同じ回数だけ回す.

        ``keep`` は基準手。差の統計を取り続けるため決して落とさない。
        """
        if len(alive) <= self.min_alive:
            return alive
        if min(stats[i].n for i in alive) < self.prune_after:
            return alive
        best = max(stats[i].mean_score for i in alive)
        kept = [
            i
            for i in alive
            if i == keep or stats[i].mean_score + self.prune_sigmas * stats[i].stderr >= best
        ]
        if len(kept) < self.min_alive:
            ordered = sorted(alive, key=lambda i: (i != keep, -stats[i].mean_score))
            kept = ordered[: self.min_alive]
        return kept

    def _run_batch(self, s, me, kind, options, n_dets, base_slot=-1):
        weights = self.weights
        base_seed = self.rng.randrange(1 << 30)
        if self.pool is not None and self.jobs > 1 and n_dets >= self.jobs:
            per = max(1, n_dets // self.jobs)
            jobs = [
                (s, me, kind, options, self.P, base_seed + 104729 * w, per,
                 self.rollout, weights, self.scale, base_slot)
                for w in range(self.jobs)
            ]
            try:
                chunks = self.pool.map(evaluate_options, jobs)
            except Exception as e:  # ワーカーが落ちたら黙って諦めずに知らせる
                print(f"  ! 並列ワーカーでエラー: {e}", file=sys.stderr)
                return None
            merged = [[0.0, 0.0, 0.0, 0, 0.0, 0.0] for _ in options]
            for chunk in chunks:
                for i, acc in enumerate(chunk):
                    for f in range(6):
                        merged[i][f] += acc[f]
            return merged
        job = (s, me, kind, options, self.P, base_seed, n_dets,
               self.rollout, weights, self.scale, base_slot)
        return evaluate_options(job)

    # -------------------------------------------------- Agent プロトコル

    def choose_play(self, s: GameState, p: int) -> int:
        from ..cards import kind_name

        options = s.legal_plays(p)
        if len(options) == 1:
            return options[0]
        prior = self.fallback.choose_play(s, p)
        stats = self.search(s, p, PLAY, options, [kind_name(k) for k in options], prior=prior)
        return stats[0].option

    def choose_second(self, s: GameState, p: int) -> int | None:
        from ..cards import kind_name

        legal = s.legal_seconds(p)
        if not legal:
            return None
        options = [None] + legal
        labels = ["出さない"] + [kind_name(k) for k in legal]
        prior = self.fallback.choose_second(s, p)
        stats = self.search(s, p, SECOND, options, labels, prior=prior)
        return stats[0].option

    def bid_options(
        self, s: GameState, p: int, lot: list[int], lot_type: int, anchor: int | None = None
    ) -> list[int]:
        """留保価格の候補. ヒューリスティックの答えを必ず含めた上で、その周りに並べる."""
        ref = self.fallback.lot_value(s, lot)
        cap = s.money[p]
        raw = [0.0] + [ref * f for f in (0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15)]
        if anchor is not None:
            raw.append(float(anchor))
        seen, out = set(), []
        for v in sorted(raw):
            q = int(min(max(v, 0), cap))
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def reservation(self, s: GameState, p: int, lot: list[int], lot_type: int) -> int:
        prior = self.fallback.reservation(s, p, lot, lot_type)
        options = self.bid_options(s, p, lot, lot_type, anchor=prior)
        stats = self.search(s, p, BID, options, [f"{q}まで" for q in options], prior=prior)
        return stats[0].option

    def sealed_bid(self, s: GameState, p: int, lot: list[int]) -> int:
        prior = self.fallback.sealed_bid(s, p, lot)
        options = self.bid_options(s, p, lot, SEALED, anchor=prior)
        stats = self.search(s, p, SEALED_BID, options, [f"{q}で入札" for q in options], prior=prior)
        return stats[0].option

    def price_options(self, s: GameState, p: int, lot: list[int], anchor: int | None = None):
        ref = self.fallback.lot_value(s, lot)
        cap = s.money[p]
        raw = [ref * f for f in (0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3)]
        if anchor is not None:
            raw.append(float(anchor))
        seen, out = set(), []
        for v in sorted(raw):
            q = int(min(max(v, 0), cap))
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def fixed_price(self, s: GameState, p: int, lot: list[int]) -> int:
        prior = max(0, min(int(self.fallback.fixed_price(s, p, lot)), s.money[p]))
        options = self.price_options(s, p, lot, anchor=prior)
        stats = self.search(s, p, PRICE, options, [f"{q}で提示" for q in options], prior=prior)
        return stats[0].option

    def fixed_accept(self, s: GameState, p: int, lot: list[int], price: int) -> bool:
        if price > s.money[p]:
            return False
        prior = self.fallback.fixed_accept(s, p, lot, price)
        stats = self.search(s, p, ACCEPT, [True, False], ["買う", "見送る"], prior=prior)
        return bool(stats[0].option)


def default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))
