import random
import unittest

from modernart import cards as C
from modernart import rules
from modernart.agents.heuristic import HeuristicAgent
from modernart.arena import SEATINGS, round_to_seatings
from modernart.params import BOUNDS, Params
from modernart.state import PHASE_AUCTION, PHASE_ROUND_END, GameState
from modernart.valuation import artist_outlook, final_values

K = C.kind


class TestOutlook(unittest.TestCase):
    def test_settled_round_is_exact(self):
        """手札が尽きていれば順位は確定しているので、見積りは実際の額と一致する."""
        mean, sd = artist_outlook((4, 3, 2, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
        self.assertEqual([round(m) for m in mean], [30, 20, 10, 0, 0])
        self.assertEqual([round(x, 6) for x in sd], [0.0] * 5)

    def test_accumulated_board_value_is_included(self):
        mean, _ = artist_outlook((4, 3, 2, 1, 0), (0, 0, 0, 0, 0), (60, 0, 0, 0, 0))
        self.assertEqual(round(mean[C.YELLOW]), 90)  # 累積60 + 1位30

    def test_unranked_artist_is_worthless_despite_high_board_value(self):
        mean, _ = artist_outlook((0, 3, 3, 3, 3), (0, 0, 0, 0, 0), (90, 0, 0, 0, 0))
        self.assertEqual(mean[C.YELLOW], 0.0)  # 出品0枚なら価格表がいくら高くても0円

    def test_tie_break_favours_earlier_artist(self):
        mean, _ = artist_outlook((2, 2, 2, 2, 2), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
        self.assertEqual([round(m) for m in mean], [30, 20, 10, 0, 0])

    def test_more_cards_on_table_means_higher_value(self):
        low, _ = artist_outlook((1, 2, 2, 2, 2), (5, 5, 5, 5, 5), (0, 0, 0, 0, 0))
        high, _ = artist_outlook((4, 2, 2, 2, 2), (5, 5, 5, 5, 5), (0, 0, 0, 0, 0))
        self.assertGreater(high[C.YELLOW], low[C.YELLOW])

    def test_cards_left_in_hands_raise_the_outlook(self):
        few, _ = artist_outlook((2, 2, 2, 2, 2), (0, 4, 4, 4, 4), (0, 0, 0, 0, 0))
        many, _ = artist_outlook((2, 2, 2, 2, 2), (8, 4, 4, 4, 4), (0, 0, 0, 0, 0))
        self.assertGreater(many[C.YELLOW], few[C.YELLOW])

    def test_matches_actual_scoring(self):
        s = GameState(4)
        s.round_counts = [4, 3, 2, 1, 0]
        s.board_value = [10, 0, 20, 0, 0]
        s.phase = PHASE_ROUND_END
        vals = final_values(s)
        s.collections[0] = [1, 1, 1, 1, 0]
        gains = s.score_round()
        self.assertEqual(gains[0], sum(vals))
        self.assertEqual(vals, [40.0, 20.0, 30.0, 0.0, 0.0])


class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.P = Params()
        self.agent = HeuristicAgent(self.P, random.Random(0), seed=1)

    def _auction_state(self, lot_type=C.OPEN, seller=0):
        s = GameState(4)
        for p in range(4):
            for k in (K(C.GREEN, C.OPEN), K(C.BLUE, C.SEALED), K(C.PINK, C.ONCE)):
                s.add_card(p, k)
        s.round_counts = [0, 3, 1, 1, 0]
        s.lot = [K(C.GREEN, lot_type)]
        s.lot_type = lot_type
        s.seller = seller
        s.phase = PHASE_AUCTION
        return s

    def test_seller_bids_less_than_buyers(self):
        """競売人は落札しても代金が銀行に消えるので、支払い意欲が下がる."""
        s = self._auction_state(seller=0)
        seller_res = self.agent.reservation(s, 0, s.lot, C.OPEN)
        buyer_res = self.agent.reservation(s, 1, s.lot, C.OPEN)
        self.assertLess(seller_res, buyer_res)

    def test_reservation_never_exceeds_money(self):
        s = self._auction_state()
        s.money[2] = 3
        self.assertLessEqual(self.agent.reservation(s, 2, s.lot, C.OPEN), 3)
        self.assertLessEqual(self.agent.sealed_bid(s, 2, s.lot), 3)

    def test_two_card_lot_is_worth_more(self):
        s = self._auction_state()
        one = self.agent.lot_value(s, [K(C.GREEN, C.OPEN)])
        two = self.agent.lot_value(s, [K(C.GREEN, C.OPEN), K(C.GREEN, C.DOUBLE)])
        self.assertAlmostEqual(two, 2 * one, places=6)

    def test_fixed_price_within_own_money(self):
        s = self._auction_state(lot_type=C.FIXED)
        s.money[0] = 12
        price = self.agent.fixed_price(s, 0, s.lot)
        self.assertGreaterEqual(price, 0)
        self.assertLessEqual(price, 12)

    def test_prefers_the_valuable_artist_when_bidding(self):
        s = GameState(4)
        for p in range(4):
            s.add_card(p, K(C.GREEN, C.OPEN))
            s.add_card(p, K(C.BEIGE, C.OPEN))
        s.round_counts = [0, 3, 0, 0, 0]  # 緑は3枚出ていて上位が堅い、肌は0枚
        s.seller = 0
        s.phase = PHASE_AUCTION
        green = self.agent.reservation(s, 1, [K(C.GREEN, C.OPEN)], C.OPEN)
        beige = self.agent.reservation(s, 1, [K(C.BEIGE, C.OPEN)], C.OPEN)
        self.assertGreater(green, beige)

    def test_static_eval_prefers_more_money(self):
        s = GameState(4)
        base = self.agent.static_eval(s, 0)
        s.money[0] += 50
        self.assertGreater(self.agent.static_eval(s, 0), base)

    def test_plays_are_legal(self):
        for seed in range(20):
            agents = [HeuristicAgent(self.P, random.Random(seed), seed=seed + i) for i in range(4)]
            s = rules.play_game(agents, random.Random(seed))
            self.assertTrue(all(m >= 0 for m in s.money))


class TestGreedyPlay(unittest.TestCase):
    """出品の判断が「式で近似した軽い版」でも筋が通っているか."""

    def setUp(self):
        from modernart.agents.heuristic import GreedyAgent

        self.agent = GreedyAgent(Params(), random.Random(0), seed=1)

    def _state(self, counts, hands=("黄公", "緑公", "桃公", "青公", "肌公")):
        s = GameState(4)
        for p in range(4):
            for name in hands:
                s.add_card(p, C.parse_kind(name))
        s.round_counts = list(counts)
        for a in range(5):
            for _ in range(counts[a]):
                s.played_ever[K(a, C.ONCE)] += 1  # 帳尻合わせ（枚数だけ使う）
        return s

    def test_avoids_wasting_the_fifth_card(self):
        """5枚目を出すとラウンドが終わり、そのカードは競りにかからず0円になる."""
        s = self._state([4, 0, 0, 0, 0])
        s.collections[0] = [0, 0, 0, 0, 0]
        chosen = self.agent.choose_play(s, 0)
        self.assertNotEqual(C.artist_of(chosen), C.YELLOW)

    def test_ends_the_round_when_holding_the_leader(self):
        """逆に、1位の絵を抱えているなら5枚目を出して決算に持ち込みたい."""
        s = self._state([4, 3, 0, 0, 0])
        s.collections[0] = [4, 0, 0, 0, 0]  # 黄を4枚持っている
        s.board_value = [60, 0, 0, 0, 0]
        chosen = self.agent.choose_play(s, 0)
        self.assertEqual(C.artist_of(chosen), C.YELLOW)

    def test_returns_a_card_in_hand(self):
        for seed in range(30):
            s = rules.new_game(4, random.Random(seed))
            k = self.agent.choose_play(s, s.turn)
            with self.subTest(seed=seed):
                self.assertGreater(s.hands[s.turn][k], 0)


class TestOutlookShortcut(unittest.TestCase):
    def test_settled_round_skips_the_simulation(self):
        """5枚出ていればラウンドは終わっている. ばらつきは0になるはず."""
        mean, sd = artist_outlook((5, 2, 1, 0, 0), (9, 9, 9, 9, 9), (0, 0, 0, 0, 0))
        self.assertEqual([round(m) for m in mean], [30, 20, 10, 0, 0])
        self.assertEqual(list(sd), [0.0] * 5)

    def test_extra_artist_shifts_the_value(self):
        from modernart.valuation import artist_values

        s = GameState(4)
        for p in range(4):
            for name in ("黄公", "緑公", "桃公"):
                s.add_card(p, C.parse_kind(name))
        s.round_counts = [2, 2, 0, 0, 0]
        base = artist_values(s, risk=0.0)
        pushed = artist_values(s, risk=0.0, extra_artist=C.YELLOW)
        self.assertGreater(pushed[C.YELLOW], base[C.YELLOW])


class TestArenaFairness(unittest.TestCase):
    def test_every_entry_visits_every_seat_equally(self):
        for n in (3, 4, 5):
            with self.subTest(n=n):
                perms = SEATINGS[n]
                for entry in range(n):
                    seats = [perm.index(entry) for perm in perms]
                    for seat in range(n):
                        self.assertEqual(seats.count(seat), len(perms) // n)

    def test_every_entry_gets_every_left_neighbour_equally(self):
        """左隣が誰かで有利不利が変わるので、そこも均等でなければ比較にならない."""
        for n in (3, 4, 5):
            with self.subTest(n=n):
                perms = SEATINGS[n]
                for entry in range(n):
                    neighbours = []
                    for perm in perms:
                        seat = perm.index(entry)
                        neighbours.append(perm[(seat + 1) % n])
                    for other in range(n):
                        if other != entry:
                            self.assertEqual(neighbours.count(other), len(perms) // (n - 1))

    def test_round_to_seatings(self):
        self.assertEqual(round_to_seatings(100, 4), 120)
        self.assertEqual(round_to_seatings(1, 3), 6)


class TestParamsPerRule(unittest.TestCase):
    """ルールごとに調整値を使い分けられているか."""

    def test_falls_back_when_the_any_color_file_is_missing(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        from modernart import params as PM

        with tempfile.TemporaryDirectory() as d:
            same = Path(d) / "tuned.json"
            anyc = Path(d) / "any.json"
            Params(gamma=0.11).save(same)
            with mock.patch.object(PM, "TUNED_PATH", same), mock.patch.object(PM, "TUNED_ANY_PATH", anyc):
                # 別色用のファイルがまだ無いので、通常の調整値に落ちる
                self.assertAlmostEqual(Params.load_for_rule(True).gamma, 0.11)
                self.assertAlmostEqual(Params.load_for_rule(False).gamma, 0.11)

                Params(gamma=0.99).save(anyc)
                self.assertAlmostEqual(Params.load_for_rule(True).gamma, 0.99)
                self.assertAlmostEqual(Params.load_for_rule(False).gamma, 0.11)
        del json


class TestParams(unittest.TestCase):
    def test_vector_roundtrip(self):
        p = Params(gamma=0.5, risk=0.1)
        self.assertEqual(Params.from_vector(p.to_vector()), p)

    def test_every_field_has_bounds(self):
        for name in Params.field_names():
            self.assertIn(name, BOUNDS)

    def test_defaults_are_within_bounds(self):
        p = Params()
        for name in Params.field_names():
            lo, hi = BOUNDS[name]
            self.assertTrue(lo <= getattr(p, name) <= hi, name)


if __name__ == "__main__":
    unittest.main()


class TestTuningGuards(unittest.TestCase):
    """自己対戦の崩壊（プールごと弱くなる）を検知できるか."""

    def test_strength_index_is_one_for_equal_players(self):
        """勝ちを人数倍しているので、実力が同じなら期待値は 1.00 になる."""
        from modernart import tune

        P = Params()
        for n in (3, 4, 5):
            with self.subTest(n=n):
                m, se = tune.evaluate(P, [P], [n], 120, seed=7, workers=None)
                self.assertLess(abs(m - 1.0), 4 * se + 0.05)

    def test_a_weak_challenger_scores_below_one(self):
        from modernart import tune

        strong = Params()
        weak = Params(shade_open=0.3, shade_once=0.3, shade_sealed=0.3, shade_fixed=0.3)
        m, _ = tune.evaluate(weak, [strong], [4], 120, seed=11, workers=None)
        self.assertLess(m, 1.0)

    def test_champion_is_checked_against_a_fixed_anchor(self):
        """プールだけでなく、動かない基準にも勝てないと交代しないこと."""
        import inspect

        from modernart import tune

        src = inspect.getsource(tune.cem)
        self.assertIn("anchor", src)
        self.assertIn("[anchor]", src, "固定の基準に対する再測定が無い")


class TestOnlyAdoptWhenStronger(unittest.TestCase):
    """強くなっていないパラメータでファイルを上書きしないこと."""

    def _run(self, extra):
        import tempfile
        from pathlib import Path

        from modernart import tune

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.json"
            Params(gamma=0.123).save(out)  # 上書きされたら分かる目印
            code = tune.main(
                ["--iters", "1", "--pop", "2", "--games", "18", "--out", str(out)] + extra
            )
            return code, Params.load_tuned(out).gamma

    def test_does_not_overwrite_without_improvement(self):
        code, gamma = self._run([])
        self.assertEqual(code, 1, "改善が無いのに成功扱いになっている")
        self.assertAlmostEqual(gamma, 0.123, msg="改善が無いのに上書きされた")

    def test_force_overwrites(self):
        code, gamma = self._run(["--force"])
        self.assertEqual(code, 0)
        self.assertNotAlmostEqual(gamma, 0.123)
