import random
import unittest

from modernart import cards as C
from modernart import rules
from modernart.advisor import Advisor, Tracker, TrackerError
from modernart.agents.heuristic import HeuristicAgent
from modernart.beliefs import determinize, unseen_counts
from modernart.cli import parse_amount, parse_player, parse_winner_and_price
from modernart.params import Params
from modernart.state import (
    PHASE_AUCTION,
    PHASE_GAME_END,
    PHASE_PLAY,
    PHASE_ROUND_END,
    PHASE_SECOND,
    IllegalAction,
)

K = C.kind
R1_HAND = "黄公 黄声 緑入 緑倍 桃差 桃公 青入 青倍 肌声"


def as_kinds(counts) -> list[int]:
    out = []
    for k in range(C.N_KINDS):
        out.extend([k] * counts[k])
    return out


def fresh(hero=1, hand=R1_HAND) -> Tracker:
    t = Tracker(4, hero, random.Random(0))
    t.deal(C.parse_hand_list(hand))
    return t


class TestTrackerBasics(unittest.TestCase):
    def test_deal_requires_the_right_count(self):
        t = Tracker(4, 1, random.Random(0))
        with self.assertRaises(TrackerError):
            t.deal(C.parse_hand_list("黄公 緑倍"))  # 4人なら9枚

    def test_deal_rejects_impossible_duplicates(self):
        t = Tracker(4, 0, random.Random(0))
        with self.assertRaises(TrackerError):
            t.deal(C.parse_hand_list("黄入 黄入 黄入 緑公 緑公 緑公 桃公 桃公 桃公"))  # 黄入は2枚しかない

    def test_failed_deal_leaves_no_trace(self):
        t = Tracker(4, 0, random.Random(0))
        with self.assertRaises(TrackerError):
            t.deal(C.parse_hand_list("黄入 黄入 黄入 緑公 緑公 緑公 桃公 桃公 桃公"))
        self.assertEqual(t.state.hand_size(0), 0)
        self.assertEqual(sum(t.state.played_ever), 0)

    def test_hand_sizes_after_deal(self):
        t = fresh()
        self.assertEqual([t.state.hand_size(p) for p in range(4)], [9] * 4)
        self.assertEqual(t.state.hands[1], C.parse_hand(R1_HAND))

    def test_unseen_accounting(self):
        t = fresh()
        pool = unseen_counts(t.state, t.hero)
        self.assertEqual(sum(pool), 70 - 9)
        for k in range(C.N_KINDS):
            self.assertGreaterEqual(pool[k], 0)

    def test_opponent_play_reduces_their_hand(self):
        t = fresh()
        before = t.state.hand_size(0)
        t.play(K(C.GREEN, C.OPEN))
        self.assertEqual(t.state.hand_size(0), before - 1)
        self.assertEqual(t.state.played_ever[K(C.GREEN, C.OPEN)], 1)
        self.assertEqual(t.state.round_counts[C.GREEN], 1)

    def test_cannot_play_more_copies_than_exist(self):
        t = fresh(hero=3)
        # 黄入札はデッキに2枚。自分は持っていないので、場に2枚出たら打ち止め
        t.play(K(C.YELLOW, C.SEALED))
        t.auction_result(3, 0)
        t.play(K(C.YELLOW, C.SEALED))
        t.auction_result(3, 0)
        with self.assertRaises(TrackerError):
            t.play(K(C.YELLOW, C.SEALED))

    def test_hero_must_actually_hold_the_card(self):
        t = fresh()
        t.play(K(C.GREEN, C.OPEN))  # P1
        t.auction_result(0, 0)
        self.assertEqual(t.state.turn, 1)  # 自分の番
        with self.assertRaises(TrackerError):
            t.play(K(C.BEIGE, C.FIXED))  # 手札にない

    def test_failed_action_leaves_no_undo_entry(self):
        """失敗した操作が履歴に残ると、「戻す」を2回押さないと戻らなくなる."""
        t = fresh(hero=1)
        t.play(K(C.GREEN, C.OPEN))
        t.auction_result(2, 15)
        money = t.state.money[:]

        with self.assertRaises(TrackerError):
            t.play(K(C.BEIGE, C.FIXED))  # 手札にない
        self.assertEqual(t.state.money, money)  # 盤面は動いていない

        self.assertTrue(t.undo())
        self.assertEqual(t.state.money, [100] * 4)  # 1回で落札前に戻る

    def test_failed_action_does_not_half_apply(self):
        t = fresh(hero=1)
        t.play(K(C.GREEN, C.OPEN))
        before = (t.state.money[:], t.state.round_counts[:], t.state.hand_size(0))
        with self.assertRaises(IllegalAction):
            t.auction_result(2, 999)  # 所持金を超える
        self.assertEqual(
            (t.state.money, t.state.round_counts, t.state.hand_size(0)), before
        )

    def test_undo_restores_everything(self):
        t = fresh()
        snapshot = (t.state.money[:], t.state.round_counts[:], t.state.hand_size(0))
        t.play(K(C.GREEN, C.OPEN))
        t.auction_result(2, 15)
        self.assertNotEqual(t.state.money, snapshot[0])
        self.assertTrue(t.undo())
        self.assertTrue(t.undo())
        self.assertEqual(t.state.money, snapshot[0])
        self.assertEqual(t.state.round_counts, snapshot[1])
        self.assertEqual(t.state.hand_size(0), snapshot[2])

    def test_mark_declines_before_hero(self):
        t = fresh(hero=2)
        t.play(K(C.GREEN, C.FIXED))  # P1 が差し値で出品
        t.announce_price(20)
        t.mark_declines_before_hero()
        self.assertTrue(t.state.lot_declined[1])  # 自分より前の P2 は断った扱い
        self.assertFalse(t.state.lot_declined[2])  # 自分
        self.assertFalse(t.state.lot_declined[3])

    def test_set_hero_hand_fixes_typos(self):
        t = fresh()
        t.set_hero_hand(C.parse_hand("黄公 緑公 桃公 青公 肌公 黄声 緑声 桃声 青声"))
        self.assertEqual(t.state.hand_size(1), 9)
        self.assertEqual(sum(t.state.hand_size(p) for p in range(4)), 36)

    def test_set_hero_hand_rejects_cards_already_played(self):
        t = fresh(hero=1)
        t.play(K(C.YELLOW, C.SEALED))  # P1 が黄入札を出す（デッキに2枚）
        t.auction_result(0, 0)
        with self.assertRaises(TrackerError):
            # 残り1枚しかないのに2枚あると言っている
            t.set_hero_hand(C.parse_hand("黄入 黄入 緑公 緑声 桃公 桃声 青公 青声 肌公"))
        self.assertEqual(t.state.hand_size(1), 9)  # 失敗しても手札は変わらない


class TestDeterminize(unittest.TestCase):
    def test_sample_is_consistent_with_observations(self):
        t = fresh()
        t.play(K(C.GREEN, C.OPEN))
        t.auction_result(2, 15)
        for seed in range(30):
            d = determinize(t.state, t.hero, random.Random(seed))
            with self.subTest(seed=seed):
                # 自分の手札はそのまま
                self.assertEqual(d.hands[t.hero], t.state.hands[t.hero])
                # 枚数も一致
                self.assertEqual(
                    [d.hand_size(j) for j in range(4)],
                    [t.state.hand_size(j) for j in range(4)],
                )
                # 手札 + 山札 + 場に出た分 = ちょうど70枚
                total = [d.played_ever[k] for k in range(C.N_KINDS)]
                for h in d.hands:
                    for k in range(C.N_KINDS):
                        total[k] += h[k]
                for k in d.deck:
                    total[k] += 1
                self.assertEqual(total, list(C.FULL_DECK))

    def test_affinity_biases_toward_observed_colours(self):
        t = fresh()
        for _ in range(6):
            t.affinity.on_play(0, C.BEIGE)
        w = t.affinity.weights()
        counts = 0
        for seed in range(60):
            d = determinize(t.state, t.hero, random.Random(seed), w)
            counts += d.hand_artist[0][C.BEIGE]
        flat = 0
        for seed in range(60):
            d = determinize(t.state, t.hero, random.Random(seed))
            flat += d.hand_artist[0][C.BEIGE]
        self.assertGreater(counts, flat)


class TestTrackerMirrorsRealGame(unittest.TestCase):
    """裏で本物のゲームを回し、その公開情報だけを Tracker に流し込んで一致を見る."""

    def _run(self, n, hero, seed):
        P = Params()
        rng = random.Random(seed)
        agents = [HeuristicAgent(P, random.Random(seed * 31 + i), seed=i) for i in range(n)]
        s = rules.new_game(n, rng)
        t = Tracker(n, hero, random.Random(seed))
        t.deal(as_kinds(s.hands[hero]))

        steps = 0
        while s.phase != PHASE_GAME_END and steps < 2000:
            steps += 1
            self.assertEqual(t.state.money, s.money)
            self.assertEqual(t.state.board_value, s.board_value)
            self.assertEqual(t.state.round_counts, s.round_counts)
            self.assertEqual(t.state.collections, s.collections)
            self.assertEqual(t.state.played_ever, s.played_ever)
            self.assertEqual(t.state.phase, s.phase)
            self.assertEqual(t.state.turn, s.turn)
            self.assertEqual(t.state.round_idx, s.round_idx)
            self.assertEqual(t.state.hands[hero], s.hands[hero])
            self.assertEqual(
                [t.state.hand_size(j) for j in range(n)],
                [s.hand_size(j) for j in range(n)],
            )

            if s.phase == PHASE_PLAY:
                k = agents[s.turn].choose_play(s, s.turn)
                s.apply_play(k)
                t.play(k)
            elif s.phase == PHASE_SECOND:
                p = s.second_offerer
                k = agents[p].choose_second(s, p)
                if k is None:
                    s.apply_decline_second()
                    t.second(None)
                else:
                    s.apply_second(k)
                    t.second(k)
            elif s.phase == PHASE_AUCTION:
                if s.lot_type == C.FIXED:
                    price = agents[s.seller].fixed_price(s, s.seller, s.lot)
                    price = max(0, min(int(price), s.money[s.seller]))
                    s.announce_fixed_price(price)
                    t.announce_price(price)
                w, price = rules.resolve_auction(s, agents, rng)
                s.apply_auction_result(w, price)
                t.auction_result(w, price)
            elif s.phase == PHASE_ROUND_END:
                s.score_round()
                t.score_round()
                if s.phase != PHASE_GAME_END:
                    before = s.hands[hero][:]
                    rules.deal(s)
                    got = []
                    for k in range(C.N_KINDS):
                        got.extend([k] * (s.hands[hero][k] - before[k]))
                    t.deal(got)
        self.assertEqual(s.phase, PHASE_GAME_END)
        self.assertEqual(t.state.money, s.money)

    def test_mirrors(self):
        for n in (3, 4, 5):
            for seed in (1, 2, 3):
                with self.subTest(n=n, seed=seed):
                    self._run(n, hero=seed % n, seed=seed)


class TestAdvisorOutput(unittest.TestCase):
    def test_gives_advice_on_own_turn_only(self):
        t = fresh(hero=1)
        adv = Advisor(t, Params(), budget=0.3, jobs=1)
        self.assertIsNone(adv.advise())  # P1 の手番なので何も言わない
        t.play(K(C.GREEN, C.OPEN))
        a = adv.advise()  # 競り -> いくらまで出すか
        self.assertIsNotNone(a)
        self.assertEqual(a.kind, "bid")
        self.assertTrue(a.rows)
        self.assertTrue(all(0 <= r.option <= t.state.money[1] for r in a.rows))

    def test_play_advice_options_are_in_hand(self):
        t = fresh(hero=0)
        adv = Advisor(t, Params(), budget=0.3, jobs=1)
        a = adv.advise()
        self.assertEqual(a.kind, "play")
        for r in a.rows:
            self.assertGreater(t.state.hands[0][r.option], 0)

    def test_fixed_price_advice_within_money(self):
        t = fresh(hero=0)
        t.play(K(C.PINK, C.FIXED))
        a = Advisor(t, Params(), budget=0.3, jobs=1).advise()
        self.assertEqual(a.kind, "price")
        self.assertTrue(all(0 <= r.option <= t.state.money[0] for r in a.rows))


class TestCliParsers(unittest.TestCase):
    def test_parse_player(self):
        pp = parse_player(4, hero=2)
        self.assertEqual(pp("3"), 2)
        self.assertEqual(pp("P1"), 0)
        self.assertEqual(pp("自分"), 2)
        with self.assertRaises(ValueError):
            pp("5")

    def test_parse_amount(self):
        pa = parse_amount(30)
        self.assertEqual(pa("24"), 24)
        self.assertEqual(pa("24,000円".replace(",000", "")), 24)
        with self.assertRaises(ValueError):
            pa("31")
        with self.assertRaises(ValueError):
            pa("-1")

    def test_parse_winner_and_price(self):
        pw = parse_winner_and_price(4, hero=1, money=[100, 100, 30, 100])
        self.assertEqual(pw("2 24"), (1, 24))
        with self.assertRaises(ValueError):
            pw("3 40")  # P3 は30しか持っていない
        with self.assertRaises(ValueError):
            pw("2")


if __name__ == "__main__":
    unittest.main()
