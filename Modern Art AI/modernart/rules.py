"""配札・競りの解決・1ゲームの進行ドライバ."""

from __future__ import annotations

import random

from .cards import (
    DEAL_TABLE,
    FIXED,
    FULL_DECK,
    N_KINDS,
    ONCE,
    OPEN,
    SEALED,
)
from .state import (
    PHASE_AUCTION,
    PHASE_GAME_END,
    PHASE_PLAY,
    PHASE_ROUND_END,
    PHASE_SECOND,
    GameState,
)


def full_deck_list() -> list[int]:
    """70枚を kind のリストとして返す."""
    out: list[int] = []
    for k in range(N_KINDS):
        out.extend([k] * FULL_DECK[k])
    return out


def new_game(n: int, rng: random.Random, double_any_artist: bool = False) -> GameState:
    if n not in DEAL_TABLE:
        raise ValueError("人数は3〜5です")
    s = GameState(n, double_any_artist)
    s.deck = full_deck_list()
    rng.shuffle(s.deck)
    deal(s)
    return s


def deal(s: GameState) -> int:
    """現在のラウンドの追加配布を行い、配った枚数(1人あたり)を返す."""
    count = DEAL_TABLE[s.n][s.round_idx]
    for _ in range(count):
        for p in range(s.n):
            if s.deck:
                s.add_card(p, s.deck.pop())
    s.start_round()
    return count


def bid_order(s: GameState) -> list[int]:
    """同額時の優先順位: 競売人が最優先、以降 時計回り."""
    return [(s.seller + i) % s.n for i in range(s.n)]


def resolve_auction(s: GameState, agents, rng: random.Random) -> tuple[int, int]:
    """現在の競りを解決し ``(落札者, 価格)`` を返す."""
    lot, lt = s.lot, s.lot_type
    order = bid_order(s)

    if lt == FIXED:
        if s.lot_price >= 0:
            price = s.lot_price  # 実戦で宣言済みの額
        else:
            price = max(0, min(int(agents[s.seller].fixed_price(s, s.seller, lot)), s.money[s.seller]))
        for p in order[1:]:  # 競売人の左隣から。競売人自身は最後の受け皿
            if s.lot_declined[p]:
                continue  # すでに断っている
            if s.money[p] >= price and agents[p].fixed_accept(s, p, lot, price):
                return p, price
        return s.seller, price  # 全員辞退 -> 競売人が自腹 (銀行へ)

    if lt == SEALED:
        bids = [0] * s.n
        for p in range(s.n):
            bids[p] = max(0, min(int(agents[p].sealed_bid(s, p, lot)), s.money[p]))
        winner = max(order, key=lambda p: bids[p])  # max は最初の最大値を返す = 優先順
        return winner, bids[winner]

    if lt not in (OPEN, ONCE):
        raise ValueError(f"競りの方式が不正です: {lt}")

    # 公開競り / 一声: 留保価格の最大が落札し、2番手の額 +1 を払う
    vals = [0] * s.n
    for p in range(s.n):
        vals[p] = max(0, min(int(agents[p].reservation(s, p, lot, lt)), s.money[p]))
    winner = max(order, key=lambda p: vals[p])
    second = max((vals[p] for p in range(s.n) if p != winner), default=0)
    price = min(vals[winner], second + 1)
    return winner, max(0, min(price, s.money[winner]))


def step(s: GameState, agents, rng: random.Random) -> None:
    """状態を1手進める."""
    ph = s.phase
    if ph == PHASE_PLAY:
        s.apply_play(agents[s.turn].choose_play(s, s.turn))
    elif ph == PHASE_SECOND:
        p = s.second_offerer
        k = agents[p].choose_second(s, p)
        if k is None:
            s.apply_decline_second()
        else:
            s.apply_second(k)
    elif ph == PHASE_AUCTION:
        winner, price = resolve_auction(s, agents, rng)
        s.apply_auction_result(winner, price)
    elif ph == PHASE_ROUND_END:
        s.score_round()
        if s.phase != PHASE_GAME_END:
            deal(s)
    else:
        raise RuntimeError("ゲームは終了しています")


def play_out(s: GameState, agents, rng: random.Random) -> GameState:
    """終局まで進める (``s`` を破壊的に更新して返す)."""
    while s.phase != PHASE_GAME_END:
        step(s, agents, rng)
    return s


def play_game(agents, rng: random.Random) -> GameState:
    return play_out(new_game(len(agents), rng), agents, rng)
