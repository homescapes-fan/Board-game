import random
import unittest

from modernart import cards as C
from modernart import rules
from modernart.agents.heuristic import HeuristicAgent
from modernart.agents.pimc import (
    ACCEPT,
    BID,
    PLAY,
    OptionStat,
    PIMCAgent,
    _Override,
    evaluate_options,
    terminal_score,
)
from modernart.params import Params
from modernart.state import PHASE_AUCTION, PHASE_GAME_END, GameState

K = C.kind


def stocked(n=4, per_player=("黄公", "緑公", "桃公", "青公", "肌公")) -> GameState:
    s = GameState(n)
    for p in range(n):
        for name in per_player:
            s.add_card(p, C.parse_kind(name))
    return s


class TestTerminalScore(unittest.TestCase):
    def test_leader_scores_above_half(self):
        s = GameState(4)
        s.money = [200, 100, 100, 100]
        score, diff, win = terminal_score(s, 0, 40.0)
        self.assertGreater(score, 0.5)
        self.assertEqual(diff, 100)
        self.assertEqual(win, 1.0)

    def test_loser_scores_below_half(self):
        s = GameState(4)
        s.money = [50, 100, 100, 100]
        score, diff, win = terminal_score(s, 0, 40.0)
        self.assertLess(score, 0.5)
        self.assertEqual(diff, -50)
        self.assertEqual(win, 0.0)

    def test_tie_splits_the_win(self):
        s = GameState(4)
        s.money = [100, 100, 50, 50]
        _, diff, win = terminal_score(s, 0, 40.0)
        self.assertEqual(diff, 0)
        self.assertEqual(win, 0.5)


class TestOverride(unittest.TestCase):
    def test_forces_the_chosen_bid_only(self):
        P = Params()
        base = HeuristicAgent(P, random.Random(0), seed=1)
        s = stocked()
        s.lot = [K(C.GREEN, C.OPEN)]
        s.lot_type = C.OPEN
        s.seller = 0
        s.phase = PHASE_AUCTION

        ov = _Override(base, BID, 42)
        self.assertEqual(ov.reservation(s, 1, s.lot, C.OPEN), 42)
        # 差し替えたのは公開競りの留保価格だけ。他は元のまま
        self.assertEqual(ov.sealed_bid(s, 1, s.lot), base.sealed_bid(s, 1, s.lot))
        self.assertEqual(ov.fixed_price(s, 1, s.lot), base.fixed_price(s, 1, s.lot))

        play_state = stocked()  # 出品フェーズの局面で委譲を確認する
        self.assertEqual(ov.choose_play(play_state, 1), base.choose_play(play_state, 1))

    def test_accept_override(self):
        P = Params()
        base = HeuristicAgent(P, random.Random(0), seed=1)
        s = stocked()
        s.lot = [K(C.GREEN, C.FIXED)]
        s.lot_type = C.FIXED
        s.seller = 0
        s.phase = PHASE_AUCTION
        self.assertTrue(_Override(base, ACCEPT, True).fixed_accept(s, 1, s.lot, 99))
        self.assertFalse(_Override(base, ACCEPT, False).fixed_accept(s, 1, s.lot, 0))


class TestFixedPriceState(unittest.TestCase):
    def _fixed_state(self):
        s = stocked()
        s.lot = [K(C.GREEN, C.FIXED)]
        s.lot_type = C.FIXED
        s.seller = 0
        s.phase = PHASE_AUCTION
        return s

    def test_announced_price_is_used(self):
        s = self._fixed_state()
        s.announce_fixed_price(37)

        class Buyer:
            name = "buyer"

            def fixed_price(self, *a):
                raise AssertionError("宣言済みなら競売人に訊いてはいけない")

            def fixed_accept(self, s, p, lot, price):
                return p == 2

        agents = [Buyer() for _ in range(4)]
        self.assertEqual(rules.resolve_auction(s, agents, random.Random(0)), (2, 37))

    def test_declined_players_are_skipped(self):
        s = self._fixed_state()
        s.announce_fixed_price(10)
        s.decline_fixed(1)

        class Eager:
            name = "eager"

            def fixed_accept(self, s, p, lot, price):
                return True

        winner, price = rules.resolve_auction(s, [Eager() for _ in range(4)], random.Random(0))
        self.assertEqual((winner, price), (2, 10))  # P2(idx1)は断り済みなので飛ばす

    def test_price_cannot_exceed_seller_money(self):
        s = self._fixed_state()
        s.money[0] = 20
        with self.assertRaises(Exception):
            s.announce_fixed_price(21)


class TestSearchAllocation(unittest.TestCase):
    def setUp(self):
        self.P = Params()
        self.agent = PIMCAgent(self.P, random.Random(0), seed=3, budget=1.5, rollout="greedy")

    def test_every_option_is_reported(self):
        s = rules.new_game(4, random.Random(2))
        options = s.legal_plays(0)
        rows = self.agent.search(s, 0, PLAY, options, [C.kind_name(k) for k in options])
        self.assertEqual(len(rows), len(options))
        self.assertEqual({r.option for r in rows}, set(options))
        self.assertTrue(all(r.n > 0 for r in rows))

    def test_survivors_share_the_top_sample_count(self):
        """打ち切られていない候補は必ず同じ回数まわっている."""
        s = rules.new_game(4, random.Random(2))
        options = s.legal_plays(0)
        rows = self.agent.search(s, 0, PLAY, options)
        top_n = rows[0].n
        # 1位は必ず最多サンプル群に属する（少ないサンプルの上振れを拾わない）
        self.assertEqual(top_n, max(r.n for r in rows))

    def test_single_option_short_circuits(self):
        s = rules.new_game(4, random.Random(2))
        rows = self.agent.search(s, 0, PLAY, [s.legal_plays(0)[0]])
        self.assertEqual(len(rows), 1)

    def test_respects_the_time_budget(self):
        import time

        s = rules.new_game(4, random.Random(2))
        agent = PIMCAgent(self.P, random.Random(0), seed=3, budget=2.0, rollout="greedy")
        t0 = time.time()
        agent.search(s, 0, PLAY, s.legal_plays(0))
        self.assertLess(time.time() - t0, 6.0)  # 見積り誤差ぶんの余裕は見る


class TestPriorFallback(unittest.TestCase):
    """差が誤差の範囲なら、ヒューリスティックの手を採る."""

    def setUp(self):
        self.agent = PIMCAgent(Params(), random.Random(0), seed=3, budget=0.5, rollout="greedy")

    def _stats(self, rows):
        """rows = [(option, 平均スコア, 基準手との平均差, 差の標準偏差, 回数)]"""
        out = []
        for opt, mean, diff, sd, n in rows:
            st = OptionStat(opt, str(opt))
            st.n = n
            st.score = mean * n
            st.diff = diff * n
            st.diff_sq = (sd * sd + diff * diff) * n
            out.append(st)
        return out

    def test_prior_wins_when_the_gap_is_noise(self):
        # 差 0.005 に対して ばらつき 0.3、400サンプル -> 誤差 0.015。有意でない
        ranked = self._stats(
            [("a", 0.505, 0.005, 0.30, 400), ("b", 0.500, 0.0, 0.0, 400)]
        )
        out = self.agent._prefer_prior(ranked, "b")
        self.assertEqual(out[0].option, "b")
        self.assertEqual(len(out), 2)

    def test_prior_loses_when_the_gap_is_real(self):
        # 差 0.60 に対して 誤差はごくわずか
        ranked = self._stats(
            [("a", 0.80, 0.60, 0.20, 4000), ("b", 0.20, 0.0, 0.0, 4000)]
        )
        out = self.agent._prefer_prior(ranked, "b")
        self.assertEqual(out[0].option, "a")

    def test_prior_already_first_is_untouched(self):
        ranked = self._stats([("a", 0.6, 0.1, 0.2, 100), ("b", 0.5, 0.0, 0.0, 100)])
        self.assertIs(self.agent._prefer_prior(ranked, "a"), ranked)

    def test_prior_not_among_options_is_ignored(self):
        ranked = self._stats([("a", 0.6, 0.1, 0.2, 100), ("b", 0.5, 0.0, 0.0, 100)])
        out = self.agent._prefer_prior(ranked, "zzz")
        self.assertEqual(out[0].option, "a")

    def test_paired_stderr_is_smaller_than_the_marginal_one(self):
        """同じ配牌の上で比べるので、差の誤差は平均そのものの誤差より小さい."""
        st = self._stats([("a", 0.5, 0.02, 0.05, 400)])[0]
        self.assertLess(st.diff_stderr, st.stderr)

    def test_bid_options_always_contain_the_prior(self):
        s = rules.new_game(4, random.Random(2))
        s.lot = [K(C.GREEN, C.OPEN)]
        s.lot_type = C.OPEN
        s.seller = 0
        s.phase = PHASE_AUCTION
        prior = self.agent.fallback.reservation(s, 1, s.lot, C.OPEN)
        self.assertIn(prior, self.agent.bid_options(s, 1, s.lot, C.OPEN, anchor=prior))

    def test_price_options_always_contain_the_prior(self):
        s = rules.new_game(4, random.Random(2))
        s.lot = [K(C.GREEN, C.FIXED)]
        s.lot_type = C.FIXED
        s.seller = 0
        s.phase = PHASE_AUCTION
        prior = max(0, min(int(self.agent.fallback.fixed_price(s, 0, s.lot)), s.money[0]))
        self.assertIn(prior, self.agent.price_options(s, 0, s.lot, anchor=prior))


class TestSearchQuality(unittest.TestCase):
    def test_refuses_to_pay_for_a_worthless_artist(self):
        """他の色で決着がつく直前、まだ1枚も出ていない色の絵に価値はない."""
        s = GameState(4)
        for p in range(4):
            for name in ("黄公", "黄声", "緑公"):
                s.add_card(p, C.parse_kind(name))
        s.round_counts = [4, 3, 0, 0, 0]  # 黄が次で5枚 -> ラウンド終了間近
        s.lot = [K(C.BEIGE, C.OPEN)]  # 肌は0枚で、手札にも1枚もない
        s.lot_type = C.OPEN
        s.seller = 0
        s.phase = PHASE_AUCTION
        agent = PIMCAgent(Params(), random.Random(0), seed=5, budget=2.0, rollout="greedy")
        rows = agent.search(s, 1, BID, agent.bid_options(s, 1, s.lot, C.OPEN))
        self.assertEqual(rows[0].option, 0, "売れない色に金を払ってはいけない")

    def test_pays_up_for_an_artist_that_is_certain_to_win(self):
        """黄が独走していて他の色は1枚も出ていない。この黄の絵は 30 で売れる."""
        s = GameState(4)
        for name in ("黄公", "黄公", "黄声"):  # すでに場に出ている3枚
            k = C.parse_kind(name)
            s.played_ever[k] += 1
            s.round_counts[C.YELLOW] += 1
        for p in range(4):  # 手札はすべて黄なので、他の色は絶対に順位に入らない
            for name in ("黄声", "黄入"):
                s.add_card(p, C.parse_kind(name))
        lot_card = C.parse_kind("黄公")
        s.played_ever[lot_card] += 1
        s.round_counts[C.YELLOW] += 1  # 競りに出ている4枚目
        s.lot = [lot_card]
        s.lot_type = C.OPEN
        s.seller = 0
        s.phase = PHASE_AUCTION

        agent = PIMCAgent(Params(), random.Random(0), seed=5, budget=2.0, rollout="greedy")
        rows = agent.search(s, 1, BID, agent.bid_options(s, 1, s.lot, C.OPEN))
        self.assertGreater(rows[0].option, 10, "1位確定の絵に相応の値を付けるべき")

    def test_evaluate_options_counts_every_rollout(self):
        s = rules.new_game(4, random.Random(9))
        options = s.legal_plays(0)
        job = (s, 0, PLAY, options, Params(), 11, 5, "greedy", None, 40.0, 0)
        totals = evaluate_options(job)
        self.assertEqual(len(totals), len(options))
        self.assertTrue(all(acc[3] == 5 for acc in totals))
        for acc in totals:
            self.assertTrue(0.0 <= acc[0] / acc[3] <= 1.0)
            self.assertTrue(0.0 <= acc[2] / acc[3] <= 1.0)
        # 基準手(index 0)自身との差は必ず0
        self.assertEqual(totals[0][4], 0.0)
        self.assertEqual(totals[0][5], 0.0)

    def test_paired_differences_are_collected(self):
        s = rules.new_game(4, random.Random(9))
        options = s.legal_plays(0)
        job = (s, 0, PLAY, options, Params(), 11, 6, "greedy", None, 40.0, 0)
        totals = evaluate_options(job)
        base = totals[0][0] / totals[0][3]
        for i, acc in enumerate(totals):
            mean_diff = acc[4] / acc[3]
            self.assertAlmostEqual(mean_diff, acc[0] / acc[3] - base, places=9)

    def test_rollouts_reach_the_end_of_the_game(self):
        s = rules.new_game(5, random.Random(4))
        agents = [HeuristicAgent(Params(), random.Random(i), seed=i) for i in range(5)]
        rules.play_out(s, agents, random.Random(4))
        self.assertEqual(s.phase, PHASE_GAME_END)
        self.assertEqual(s.round_idx, 4)


if __name__ == "__main__":
    unittest.main()
