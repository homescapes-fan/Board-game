"""エージェントのインタフェース.

競りは1入札ずつ模擬せず、各プレイヤーの「留保価格」(そこまでなら払う額) を訊いてから
方式ごとにまとめて解決する。実際の競りの読み合いは留保価格の側に織り込む。
"""

from __future__ import annotations

from typing import Protocol

from ..state import GameState


class Agent(Protocol):
    name: str

    def choose_play(self, state: GameState, p: int) -> int:
        """手番として場に出すカードの kind を返す."""
        ...

    def choose_second(self, state: GameState, p: int) -> int | None:
        """ダブルの2枚目に出すカード. 出さないなら None."""
        ...

    def reservation(self, state: GameState, p: int, lot: list[int], lot_type: int) -> int:
        """公開競り/一声で、そこまでなら払ってよい上限額 (千円単位)."""
        ...

    def sealed_bid(self, state: GameState, p: int, lot: list[int]) -> int:
        """入札 (一位価格・同時公開) で提出する額."""
        ...

    def fixed_price(self, state: GameState, p: int, lot: list[int]) -> int:
        """差し値の競売人として提示する額. 自分の所持金を超えてはならない."""
        ...

    def fixed_accept(self, state: GameState, p: int, lot: list[int], price: int) -> bool:
        """差し値を提示されて買うかどうか."""
        ...
