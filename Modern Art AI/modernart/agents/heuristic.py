"""評価関数ベースのエージェント.

強さそのものより「速くて筋が通っている」ことを狙う。PIMC のロールアウト方策と
相手モデルを兼ねるので、1手あたりの計算量が効いてくる。

競りの値付けは次の考え方で決める。lot の期待価値を W、相手の得を自分の損として
見る重みを γ (1人あたり g = γ/(n-1)) とすると、

  自分が q で落札      -> 自分 +W-q、競売人 +q
  ライバルが q で落札  -> ライバル +W-q、競売人 +q

の比較から、払ってよい上限は

  競売人でない: q_max = (W + g·W_rival) / (1 + g)
  競売人       : q_max = (W + g·W_rival) / (2 + g)

となる。競売人のときに上限がおよそ半分になるのは、自分で落札すると代金が銀行に
消えて売上が入らないため。ここに方式ごとの掛け目 (shade) を掛けたものを留保価格とする。
"""

from __future__ import annotations

import math
import random

from .. import rules
from ..cards import CARDS_TO_END_ROUND, DOUBLE, N_ARTISTS, ONCE, artist_of, type_of
from ..params import Params
from ..state import PHASE_AUCTION, PHASE_ROUND_END, PHASE_SECOND, GameState
from ..valuation import artist_values, final_values

# 決定的なゆらぎを作るための正規乱数テーブル (競りが機械的になるのを防ぐ)
_NOISE_TABLE = tuple(random.Random(20240824).gauss(0.0, 1.0) for _ in range(1024))
_NOISE_MASK = len(_NOISE_TABLE) - 1


class HeuristicAgent:
    __slots__ = ("P", "rng", "name", "seed", "_depth", "_selves", "max_depth")

    def __init__(
        self,
        params: Params | None = None,
        rng: random.Random | None = None,
        name: str = "heuristic",
        seed: int = 0,
    ):
        self.P = params or Params()
        self.rng = rng or random.Random()
        self.name = name
        self.seed = seed
        self._depth = 0
        self.max_depth = 1
        self._selves: list = []

    # ------------------------------------------------------------------ 評価

    def values(self, s: GameState) -> list[float]:
        """今ラウンド末に各画家のカード1枚がいくらになるかの評価額."""
        return artist_values(s, self.P.risk, self.P.cash_weight(s.round_idx))

    def _noise(self, p: int, s: GameState, lot: list[int]) -> float:
        sigma = self.P.value_noise
        if sigma <= 0.0:
            return 1.0
        h = hash((self.seed, p, s.round_idx, tuple(s.round_counts), s.seller, tuple(lot)))
        return math.exp(sigma * _NOISE_TABLE[h & _NOISE_MASK])

    def lot_value(self, s: GameState, lot: list[int], vals: list[float] | None = None) -> float:
        v = vals if vals is not None else self.values(s)
        return sum(v[artist_of(k)] for k in lot)

    def static_eval(self, s: GameState, p: int) -> float:
        """局面の相対的な良さ. 自分の資産 − γ×(他人の平均資産)."""
        vals = final_values(s) if s.phase == PHASE_ROUND_END else self.values(s)
        n = s.n
        g = self.P.gamma / (n - 1)
        # 4ラウンド目の決算後は手札が無価値になる
        hand_w = 0.0 if (s.phase == PHASE_ROUND_END and s.round_idx >= 3) else self.P.future_weight

        def assets(j: int) -> float:
            col = s.collections[j]
            hand = s.hand_artist[j]
            total = float(s.money[j])
            for a in range(N_ARTISTS):
                if col[a]:
                    total += col[a] * vals[a]
                if hand_w and hand[a]:
                    total += hand_w * hand[a] * vals[a]
            return total

        mine = assets(p)
        others = 0.0
        for j in range(n):
            if j != p:
                others += assets(j)
        return mine - g * others

    # -------------------------------------------------------------- 競りの値付け

    def _max_pay(self, s: GameState, p: int, lot: list[int], vals=None) -> float:
        """相手の得も勘定に入れた上での支払い上限 (掛け目を掛ける前)."""
        w = self.lot_value(s, lot, vals)
        g = self.P.gamma / (s.n - 1)
        # 完全情報のロールアウトではライバルの評価額も同じ値になる
        num = w + g * w
        den = (2.0 + g) if p == s.seller else (1.0 + g)
        return num / den

    def reservation(self, s: GameState, p: int, lot: list[int], lot_type: int) -> int:
        q = self._max_pay(s, p, lot) * self.P.shade(lot_type) * self._noise(p, s, lot)
        if lot_type == ONCE and p == s.seller:  # 一声は競売人が最後に声を出せる
            q *= self.P.once_seller_bonus
        return int(min(q, s.money[p]))

    def sealed_bid(self, s: GameState, p: int, lot: list[int]) -> int:
        q = self._max_pay(s, p, lot) * self.P.shade_sealed * self._noise(p, s, lot)
        return int(min(q, s.money[p]))

    def fixed_price(self, s: GameState, p: int, lot: list[int]) -> int:
        """差し値の提示額.

        「売れる最高額で売る」か「誰も買わない額にして自分で抱える」かの2択を比べる。
        """
        vals = self.values(s)
        w = self.lot_value(s, lot, vals)
        g = self.P.gamma / (s.n - 1)

        # 買い手側の留保価格 (競売人でない側の式) を、一番払える相手について見る
        rival_max = w * self.P.shade_fixed
        richest = max((s.money[j] for j in range(s.n) if j != p), default=0)
        rival_max = min(rival_max, richest)

        sell_price = int(rival_max * self.P.fixed_margin)
        sell_price = max(0, min(sell_price, s.money[p]))
        util_sell = sell_price * (1.0 + g) - g * w

        keep_price = min(int(rival_max) + 1, s.money[p])
        util_keep = w - keep_price

        return sell_price if util_sell >= util_keep else keep_price

    def fixed_accept(self, s: GameState, p: int, lot: list[int], price: int) -> bool:
        q = self._max_pay(s, p, lot) * self.P.shade_fixed * self._noise(p, s, lot)
        return price <= q and price <= s.money[p]

    # ------------------------------------------------------------------ 出品

    def _agents_for(self, n: int) -> list:
        if len(self._selves) != n:
            self._selves = [self] * n
        return self._selves

    def _resolve_lot(self, t: GameState) -> None:
        """出品されたばかりの1件を、競りの決着まで進める."""
        agents = self._agents_for(t.n)
        guard = 0
        limit = 2 * t.n + 4
        while t.phase in (PHASE_SECOND, PHASE_AUCTION) and guard < limit:
            rules.step(t, agents, self.rng)
            guard += 1

    def choose_play(self, s: GameState, p: int) -> int:
        options = s.legal_plays(p)
        if len(options) == 1:
            return options[0]
        best_k, best_v = options[0], -1e30
        for k in options:
            t = s.clone()
            t.apply_play(k)
            self._resolve_lot(t)
            v = self.static_eval(t, p)
            if v > best_v:
                best_v, best_k = v, k
        return best_k

    def choose_second(self, s: GameState, p: int) -> int | None:
        options = s.legal_seconds(p)
        if not options:
            return None
        if self._depth >= self.max_depth:
            return self._cheap_second(s, p, options)

        self._depth += 1
        try:
            t = s.clone()
            t.apply_decline_second()
            self._resolve_lot(t)
            best_v, best_k = self.static_eval(t, p), None

            for k in options:
                t = s.clone()
                t.apply_second(k)
                self._resolve_lot(t)
                v = self.static_eval(t, p)
                if v > best_v:
                    best_v, best_k = v, k
            return best_k
        finally:
            self._depth -= 1

    def _cheap_second(self, s: GameState, p: int, options: list[int]) -> int | None:
        """深い所で呼ばれたとき用の簡易版: 売上が見込めるなら出す."""
        vals = self.values(s)
        a = artist_of(options[0])
        revenue = 2.0 * vals[a] * self.P.shade_open
        if revenue < self.P.second_min_revenue:
            return None
        return max(options, key=lambda k: self.P.shade(type_of(k)))


class GreedyAgent(HeuristicAgent):
    """出品を「局面を回さず、式で見積もって」決める軽い版.

    ``HeuristicAgent.choose_play`` は候補ごとに局面を複製して競りまで進めるので重い。
    PIMC のロールアウトはこれを何万回も呼ぶため、同じ勘所を閉じた式で近似する。
    見ているのは3つだけ:

      1. その色が1枚増えると、自分と相手が *すでに持っている絵* の値がどう動くか
      2. 競売人として入る売上（買い手が得るぶんは差し引く）
      3. その色の5枚目なら競りは起きない。ラウンドを終わらせる損得だけを見る
    """

    def choose_play(self, s: GameState, p: int) -> int:
        options = s.legal_plays(p)
        if len(options) == 1:
            return options[0]

        vals = self.values(s)
        g = self.P.gamma / (s.n - 1)
        cw = self.P.cash_weight(s.round_idx)
        n = s.n
        # 同じ画家なら「出したあと」の評価は共通なので、画家ごとに1回だけ計算する
        after: dict[int, list[float]] = {}
        # 場に出ている絵の枚数（自分/相手）を先に集計しておく
        mine_col = s.collections[p]
        others_col = [0] * N_ARTISTS
        for j in range(n):
            if j != p:
                cj = s.collections[j]
                for b in range(N_ARTISTS):
                    others_col[b] += cj[b]

        best_k, best_v = options[0], -1e30
        for k in options:
            a = artist_of(k)
            va = after.get(a)
            if va is None:
                va = after[a] = artist_values(s, self.P.risk, cw, extra_artist=a)

            score = 0.0
            for b in range(N_ARTISTS):
                d = va[b] - vals[b]
                if d:
                    score += d * (mine_col[b] - g * others_col[b])

            if s.round_counts[a] + 1 < CARDS_TO_END_ROUND:  # 5枚目なら競りは起きない
                if type_of(k) == DOUBLE:
                    # 2枚目が出れば2枚を売り、出なければ1枚を無料で取れる
                    w2 = va[a] * 2.0
                    price = w2 * self.P.shade_open
                    sell = price - g * (w2 - price)
                    score += 0.5 * sell + 0.5 * va[a]
                else:
                    w = va[a]
                    price = w * self.P.shade_open
                    score += price - g * (w - price)

            if score > best_v:
                best_v, best_k = score, k
        return best_k

    def choose_second(self, s: GameState, p: int) -> int | None:
        options = s.legal_seconds(p)
        return self._cheap_second(s, p, options) if options else None
