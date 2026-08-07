import random
import unittest

from modernart import cards as C
from modernart.advisor import Advisor, Tracker
from modernart.cli import (
    NO_SECOND,
    Console,
    Quit,
    parse_winner_and_price,
    step_auction,
    step_play,
    step_round_end,
    step_second,
)
from modernart.params import Params
from modernart.state import (
    PHASE_AUCTION,
    PHASE_GAME_END,
    PHASE_PLAY,
    PHASE_ROUND_END,
    PHASE_SECOND,
)

K = C.kind


class Script:
    """台本どおりに答える input の代わり."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.asked = []

    def __call__(self, prompt: str) -> str:
        self.asked.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def build(n, hero, hand, lines, budget=0.05):
    t = Tracker(n, hero, random.Random(0))
    t.deal(C.parse_hand_list(hand))
    script = Script(lines)
    con = Console({"t": t}, read=script)
    advisor = Advisor(t, Params(), budget=budget, jobs=1)
    return t, con, advisor, script


class TestStepFlow(unittest.TestCase):
    HAND3 = "黄公 黄声 緑入 緑倍 緑公 桃差 桃公 青入 青倍 肌声"

    def test_play_then_auction(self):
        t, con, adv, _ = build(3, 0, self.HAND3, ["緑公", "2 20"])
        step_play(con, t, adv)
        self.assertEqual(t.state.phase, PHASE_AUCTION)
        self.assertEqual(t.state.lot, [K(C.GREEN, C.OPEN)])
        step_auction(con, t, adv)
        self.assertEqual(t.state.money, [120, 80, 100])
        self.assertEqual(t.state.collections[1][C.GREEN], 1)

    def test_enter_accepts_the_recommendation(self):
        """空入力で推奨がそのまま通る（文字列のまま渡らないこと）."""
        t, con, adv, _ = build(3, 0, self.HAND3, ["", "1 0"])
        step_play(con, t, adv)
        self.assertEqual(t.state.phase, PHASE_SECOND if t.state.pending_double >= 0 else PHASE_AUCTION)
        self.assertEqual(t.state.hand_size(0), 9)  # 何か1枚は出た

    def test_double_second_by_enter_uses_the_recommended_card(self):
        t, con, adv, _ = build(3, 0, self.HAND3, ["緑倍", "", "2 20"])
        step_play(con, t, adv)
        self.assertEqual(t.state.phase, PHASE_SECOND)
        step_second(con, t, adv)
        # 出したにせよ出さなかったにせよ、文字列が渡って落ちないこと
        self.assertIn(t.state.phase, (PHASE_AUCTION, PHASE_SECOND))
        if t.state.phase == PHASE_AUCTION:
            self.assertEqual(len(t.state.lot), 2)
            self.assertEqual(t.state.seller, 0)

    def test_double_second_explicit_card(self):
        t, con, adv, _ = build(3, 0, self.HAND3, ["緑倍", "緑公", "2 20"])
        step_play(con, t, adv)
        step_second(con, t, adv)
        self.assertEqual(t.state.phase, PHASE_AUCTION)
        self.assertEqual(sorted(t.state.lot), sorted([K(C.GREEN, C.DOUBLE), K(C.GREEN, C.OPEN)]))
        self.assertEqual(t.state.lot_type, C.OPEN)

    def test_double_second_declined_by_everyone(self):
        t, con, adv, _ = build(3, 0, self.HAND3, ["緑倍", "なし", "なし", "なし"])
        step_play(con, t, adv)
        for _ in range(3):
            step_second(con, t, adv)
        self.assertEqual(t.state.collections[0][C.GREEN], 1)  # 無料で獲得
        self.assertEqual(t.state.money, [100, 100, 100])
        self.assertEqual(t.state.phase, PHASE_PLAY)

    def test_fixed_price_flow(self):
        t, con, adv, _ = build(3, 1, self.HAND3, ["桃差", "15", "2"])
        # P1(idx0) が差し値を出す
        step_play(con, t, adv)
        self.assertEqual(t.state.lot_type, C.FIXED)
        step_auction(con, t, adv)
        self.assertEqual(t.state.money, [115, 85, 100])  # 自分が15で購入
        self.assertEqual(t.state.collections[1][C.PINK], 1)

    def test_fixed_price_rejects_a_different_amount(self):
        t, con, adv, script = build(3, 1, self.HAND3, ["桃差", "15", "2 12", "2 15"])
        step_play(con, t, adv)
        step_auction(con, t, adv)
        # 12 は弾かれて再入力になり、15 が通る
        self.assertEqual(t.state.money, [115, 85, 100])
        self.assertEqual(script.lines, [])

    def test_seller_buys_when_nobody_takes_the_fixed_price(self):
        t, con, adv, _ = build(3, 1, self.HAND3, ["桃差", "15", "1"])
        step_play(con, t, adv)
        step_auction(con, t, adv)
        self.assertEqual(t.state.money, [85, 100, 100])  # 競売人が銀行へ支払う
        self.assertEqual(t.state.collections[0][C.PINK], 1)

    def test_quit_on_end_of_input(self):
        t, con, adv, _ = build(3, 0, self.HAND3, [])
        with self.assertRaises(Quit):
            step_play(con, t, adv)


class TestMetaCommands(unittest.TestCase):
    HAND = "黄公 黄声 緑入 緑倍 緑公 桃差 桃公 青入 青倍 肌声"

    def test_undo_raises_restart_and_rolls_back(self):
        from modernart.cli import Restart

        t, con, adv, _ = build(3, 0, self.HAND, ["緑公", "2 20", "undo"])
        step_play(con, t, adv)
        step_auction(con, t, adv)
        self.assertEqual(t.state.money, [120, 80, 100])
        with self.assertRaises(Restart):
            con.ask("なにか", lambda r: r)
        self.assertEqual(t.state.money, [100, 100, 100])

    def test_hand_command_replaces_the_hand(self):
        from modernart.cli import Restart

        t, con, adv, _ = build(
            3, 0, self.HAND, ["hand", "黄公 黄声 緑入 緑倍 緑公 桃差 桃公 青入 青倍 肌公"]
        )
        with self.assertRaises(Restart):
            con.ask("なにか", lambda r: r)
        self.assertEqual(t.state.hands[0][K(C.BEIGE, C.OPEN)], 1)
        self.assertEqual(t.state.hands[0][K(C.BEIGE, C.ONCE)], 0)

    def test_bad_input_is_retried(self):
        t, con, adv, script = build(3, 0, self.HAND, ["むらさき公", "緑公", "2 20"])
        step_play(con, t, adv)
        self.assertEqual(t.state.lot, [K(C.GREEN, C.OPEN)])
        self.assertEqual(len(script.asked), 2)  # 1回やり直している


class TestRoundTransition(unittest.TestCase):
    def test_round_end_scores_and_deals(self):
        """3人でラウンドを終わらせ、ラウンド2の追加配札まで通す."""
        t = Tracker(3, 0, random.Random(0))
        t.deal(C.parse_hand_list("黄公 黄声 緑入 緑倍 緑公 桃差 桃公 青入 青倍 肌声"))
        # 黄を5枚出してラウンドを終わらせる
        s = t.state
        s.round_counts[C.YELLOW] = 4
        s.collections[0][C.YELLOW] = 2
        t.play(K(C.YELLOW, C.OPEN))
        self.assertEqual(s.phase, PHASE_ROUND_END)

        script = Script(["黄声 緑入 緑倍 桃差 桃公 青入"])  # ラウンド2は3人で6枚
        con = Console({"t": t}, read=script)
        step_round_end(con, t)
        self.assertEqual(s.round_idx, 1)
        self.assertEqual(s.board_value[C.YELLOW], 30)
        self.assertEqual(s.money[0], 100 + 60)  # 黄2枚 × 30
        # 自分だけ1枚出しているので 9+6、相手は 10+6
        self.assertEqual([s.hand_size(p) for p in range(3)], [15, 16, 16])
        self.assertEqual(s.phase, PHASE_PLAY)

    def test_last_round_needs_no_deal(self):
        t = Tracker(3, 0, random.Random(0))
        t.deal(C.parse_hand_list("黄公 黄声 緑入 緑倍 緑公 桃差 桃公 青入 青倍 肌声"))
        s = t.state
        s.round_idx = 3
        s.round_counts[C.YELLOW] = 4
        s.phase = PHASE_PLAY
        t.play(K(C.YELLOW, C.OPEN))
        con = Console({"t": t}, read=Script([]))
        step_round_end(con, t)  # 配札プロンプトを出さずに終わる
        self.assertEqual(s.phase, PHASE_GAME_END)


class TestFixedPriceParser(unittest.TestCase):
    def test_winner_only_is_allowed_for_fixed_price(self):
        pw = parse_winner_and_price(3, hero=0, money=[100, 100, 100], fixed_price=15)
        self.assertEqual(pw("2"), (1, 15))
        self.assertEqual(pw("2 15"), (1, 15))
        with self.assertRaises(ValueError):
            pw("2 14")

    def test_sentinel_is_distinct_from_none(self):
        self.assertIsNotNone(NO_SECOND)


if __name__ == "__main__":
    unittest.main()
