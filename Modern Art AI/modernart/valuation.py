"""ラウンド末に各画家のカードがいくらで売れるかの見積り.

ラウンドはどれか1色が5枚出た時点で終わる。よって「今 何枚出ているか」と
「まだ手札に何枚残っているか」が決まれば、最終的な順位の分布はほぼ決まる。
これを短いモンテカルロで推定し、``lru_cache`` で使い回す。

入力は全て整数タプルなのでキャッシュのキーにそのまま使える。
"""

from __future__ import annotations

import random
from functools import lru_cache

from .cards import CARDS_TO_END_ROUND, N_ARTISTS, RANK_BONUS

#: 1回の見積りに使うサンプル数
SAMPLES = 40


def _rank_payout(c: list[int], board_value, out: list[float]) -> None:
    """出品枚数 ``c`` から各画家の売却額を ``out`` に書く (同数は画家順で上位)."""
    for a in range(N_ARTISTS):
        out[a] = 0.0
    taken = [False] * N_ARTISTS
    for slot in range(3):
        best, best_c = -1, 0
        for a in range(N_ARTISTS):
            if not taken[a] and c[a] > best_c:
                best, best_c = a, c[a]
        if best < 0:
            return  # 出品のある画家が3色未満
        taken[best] = True
        out[best] = board_value[best] + RANK_BONUS[slot]


@lru_cache(maxsize=1 << 19)
def artist_outlook(
    counts: tuple[int, ...],
    in_hands: tuple[int, ...],
    board_value: tuple[int, ...],
    samples: int = SAMPLES,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(平均売却額, 標準偏差)`` を画家ごとに返す.

    counts     : 今ラウンドすでに場に出ている枚数
    in_hands   : 全員の手札にある枚数 (= 今ラウンドまだ出うる枚数)
    board_value: 価格表の累積額
    """
    pay = [0.0] * N_ARTISTS
    # もう5枚出ている、または手札が尽きている = 順位は確定している
    if max(counts) >= CARDS_TO_END_ROUND or sum(in_hands) == 0:
        _rank_payout(list(counts), board_value, pay)
        return tuple(pay), (0.0,) * N_ARTISTS

    total = [0.0] * N_ARTISTS
    total_sq = [0.0] * N_ARTISTS
    # 乱数の種を入力から作る。共有の乱数を使うと、同じ局面でも呼び出し順で
    # 値が変わってしまい、並列ワーカー間でも食い違う
    rand = random.Random(hash((counts, in_hands, board_value)) & 0xFFFFFFFF).random

    for _ in range(samples):
        c = list(counts)
        w = list(in_hands)
        left = w[0] + w[1] + w[2] + w[3] + w[4]
        s = float(left)
        while left > 0:
            r = rand() * s
            a = 0
            acc = w[0]
            while r >= acc and a < N_ARTISTS - 1:
                a += 1
                acc += w[a]
            if w[a] <= 0:  # 数値誤差で空の色を引いた場合
                break
            w[a] -= 1
            s -= 1.0
            left -= 1
            c[a] += 1
            if c[a] >= CARDS_TO_END_ROUND:
                break
        _rank_payout(c, board_value, pay)
        for a in range(N_ARTISTS):
            v = pay[a]
            total[a] += v
            total_sq[a] += v * v

    mean = []
    sd = []
    for a in range(N_ARTISTS):
        m = total[a] / samples
        var = total_sq[a] / samples - m * m
        mean.append(m)
        sd.append(var**0.5 if var > 0 else 0.0)
    return tuple(mean), tuple(sd)


def artist_values(
    state, risk: float, cash_weight: float = 1.0, extra_artist: int = -1
) -> list[float]:
    """状態から、今ラウンド末に各画家のカード1枚がいくらになるかの評価額.

    ``extra_artist`` を渡すと「その色をもう1枚 場に出したあと」の値を返す。
    出品を選ぶときの「この色を押したらどうなるか」に使う。
    """
    counts = state.round_counts
    hands = state.hand_artist_total
    if extra_artist >= 0:
        counts = list(counts)
        hands = list(hands)
        counts[extra_artist] += 1
        if hands[extra_artist] > 0:
            hands[extra_artist] -= 1
    mean, sd = artist_outlook(tuple(counts), tuple(hands), tuple(state.board_value))
    inv = 1.0 / cash_weight
    return [(mean[a] - risk * sd[a]) * inv for a in range(N_ARTISTS)]


def final_values(state) -> list[float]:
    """ラウンド終了が確定した状態での、実際の売却額."""
    out = [0.0] * N_ARTISTS
    _rank_payout(state.round_counts, state.board_value, out)
    return out
