"""相手の手札の推定 (determinization).

自分から見て未確認のカードは

    未確認 = 全70枚 − 自分の手札 − これまでに場に出た全カード

で確定する。これを各相手の既知の手札枚数どおりに配り、余りを山札とすれば、
観測と矛盾しない完全情報の局面が1つできる。PIMC はこれを何度も作って平均する。

一様に配るのが基本だが、「よく出品している色/高値で競り落とした色は多く持って
いそう」という傾向を ``affinity`` として弱く反映できる。
"""

from __future__ import annotations

import math
import random
from heapq import nlargest

from .cards import FULL_DECK, N_ARTISTS, N_KINDS, artist_of
from .state import GameState


def unseen_counts(state: GameState, me: int) -> list[int]:
    """``me`` から見て未確認のカード (= 他プレイヤーの手札 + 未配布の山札)."""
    hand = state.hands[me]
    played = state.played_ever
    return [FULL_DECK[k] - hand[k] - played[k] for k in range(N_KINDS)]


def as_card_list(counts) -> list[int]:
    out: list[int] = []
    for k in range(N_KINDS):
        c = counts[k]
        if c:
            out.extend([k] * c)
    return out


def _draw_weighted(pool: list[int], size: int, weights, rng: random.Random) -> list[int]:
    """重み付きの非復元抽出 (Efraimidis-Spirakis). ``pool`` から抜いた分を返す."""
    # key = U^(1/w) の大きい順に size 個。w が大きいほど選ばれやすい
    rand = rng.random
    keyed = [(rand() ** (1.0 / weights[artist_of(k)]), i, k) for i, k in enumerate(pool)]
    picked = nlargest(size, keyed)
    taken = {i for _, i, _ in picked}
    rest = [k for i, k in enumerate(pool) if i not in taken]
    pool[:] = rest
    return [k for _, _, k in picked]


def sample_hands(
    pool: list[int],
    sizes: list[int],
    rng: random.Random,
    weights: list[list[float]] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """``pool`` を ``sizes`` どおりに配り、``(各人の手札(kindのリスト), 余り)`` を返す.

    ``pool`` は破壊しない。
    """
    rest = pool[:]
    if weights is None:
        hands: list[list[int]] = []
        rng.shuffle(rest)
        pos = 0
        for size in sizes:
            hands.append(rest[pos : pos + size])
            pos += size
        return hands, rest[pos:]

    # 先に引く人ほど希望どおりの札を取れてしまうので、引く順番はその都度シャッフルする
    hands = [[] for _ in sizes]
    order = list(range(len(sizes)))
    rng.shuffle(order)
    for j in order:
        if sizes[j] > 0:
            hands[j] = _draw_weighted(rest, sizes[j], weights[j], rng)
    rng.shuffle(rest)
    return hands, rest


def determinize(
    state: GameState,
    me: int,
    rng: random.Random,
    weights: list[list[float]] | None = None,
) -> GameState:
    """``me`` の観測と矛盾しない完全情報の局面を1つ作る.

    ``state`` の相手の手札の *中身* は無視し、枚数だけを使う。
    """
    t = state.clone()
    pool = as_card_list(unseen_counts(state, me))

    others = [j for j in range(state.n) if j != me]
    sizes = [state.hand_size(j) for j in others]
    if sum(sizes) > len(pool):
        raise ValueError(
            f"手札の合計 {sum(sizes)} 枚に対して未確認カードが {len(pool)} 枚しかありません。"
            "入力に取りこぼしがあるかもしれません"
        )

    w = None
    if weights is not None:
        w = [weights[j] for j in others]
    hands, rest = sample_hands(pool, sizes, rng, w)

    for j, cards in zip(others, hands):
        counts = [0] * N_KINDS
        for k in cards:
            counts[k] += 1
        t.set_hand(j, counts)
    t.deck = rest

    # ダブルの2枚目を聞かれている最中なら、その人が出せる札を持つ確率も反映済み
    return t


class Affinity:
    """「この人はこの色を多く持っていそう」という弱い推定."""

    __slots__ = ("w", "play_weight", "win_weight", "cap")

    def __init__(self, n: int, play_weight: float = 0.18, win_weight: float = 0.12, cap: float = 1.0):
        self.w = [[0.0] * N_ARTISTS for _ in range(n)]
        self.play_weight = play_weight
        self.win_weight = win_weight
        self.cap = cap

    def on_play(self, p: int, artist: int) -> None:
        self._bump(p, artist, self.play_weight)

    def on_win(self, p: int, artist: int, price: int, reference: float) -> None:
        """相場より高く買ったなら、その色を推している可能性が高い."""
        if reference <= 0:
            return
        eager = min(2.0, price / reference)
        self._bump(p, artist, self.win_weight * eager)

    def _bump(self, p: int, a: int, amount: float) -> None:
        v = self.w[p][a] + amount
        self.w[p][a] = max(-self.cap, min(self.cap, v))

    def weights(self) -> list[list[float]]:
        """サンプリング用の相対重み (exp をとって正の値にする)."""
        return [[math.exp(x) for x in row] for row in self.w]

    def clone(self) -> "Affinity":
        c = Affinity(len(self.w), self.play_weight, self.win_weight, self.cap)
        c.w = [row[:] for row in self.w]
        return c
