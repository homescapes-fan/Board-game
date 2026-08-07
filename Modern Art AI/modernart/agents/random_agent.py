"""ベースラインのランダムエージェント (エンジンの検証と強さの下限測定に使う)."""

from __future__ import annotations

import random

from ..state import GameState


class RandomAgent:
    def __init__(self, rng: random.Random | None = None, name: str = "random"):
        self.rng = rng or random.Random()
        self.name = name

    def choose_play(self, state: GameState, p: int) -> int:
        return self.rng.choice(state.legal_plays(p))

    def choose_second(self, state: GameState, p: int) -> int | None:
        options = state.legal_seconds(p)
        if not options or self.rng.random() < 0.4:
            return None
        return self.rng.choice(options)

    def reservation(self, state: GameState, p: int, lot: list[int], lot_type: int) -> int:
        return self.rng.randint(0, min(30, state.money[p]))

    def sealed_bid(self, state: GameState, p: int, lot: list[int]) -> int:
        return self.rng.randint(0, min(30, state.money[p]))

    def fixed_price(self, state: GameState, p: int, lot: list[int]) -> int:
        return self.rng.randint(0, min(30, state.money[p]))

    def fixed_accept(self, state: GameState, p: int, lot: list[int], price: int) -> bool:
        return price <= self.rng.randint(0, 30)
