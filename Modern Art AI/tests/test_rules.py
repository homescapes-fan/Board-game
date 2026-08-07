import random
import unittest

from modernart import cards as C
from modernart import rules
from modernart.agents import RandomAgent
from modernart.state import (
    PHASE_AUCTION,
    PHASE_GAME_END,
    PHASE_PLAY,
    PHASE_ROUND_END,
    PHASE_SECOND,
    GameState,
    IllegalAction,
)

K = C.kind


def blank(n=4) -> GameState:
    """手札も山札も空の状態. 遷移を1手ずつ検証するのに使う."""
    return GameState(n)


def give(s, p, *kinds):
    for k in kinds:
        s.add_card(p, k)


def at_round_end(counts, n=4) -> GameState:
    """決算直前の状態を組み立てる."""
    s = GameState(n)
    s.round_counts = list(counts)
    s.phase = PHASE_ROUND_END
    return s


class TestNormalAuction(unittest.TestCase):
    def test_winner_pays_seller(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        give(s, 1, K(C.BLUE, C.OPEN))  # ラウンドが即終了しないようダミー
        s.apply_play(K(C.GREEN, C.OPEN))
        self.assertEqual(s.phase, PHASE_AUCTION)
        self.assertEqual(s.lot, [K(C.GREEN, C.OPEN)])
        self.assertEqual(s.lot_type, C.OPEN)
        self.assertEqual(s.seller, 0)
        self.assertEqual(s.round_counts[C.GREEN], 1)

        s.apply_auction_result(winner=2, price=24)
        self.assertEqual(s.money[2], 76)
        self.assertEqual(s.money[0], 124)
        self.assertEqual(s.collections[2][C.GREEN], 1)
        self.assertEqual(s.phase, PHASE_PLAY)
        self.assertEqual(s.turn, 1)

    def test_seller_wins_pays_bank(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        give(s, 1, K(C.BLUE, C.OPEN))
        s.apply_play(K(C.GREEN, C.OPEN))
        s.apply_auction_result(winner=0, price=15)
        self.assertEqual(s.money[0], 85)
        self.assertEqual(sum(s.money), 400 - 15)  # 銀行へ消える
        self.assertEqual(s.collections[0][C.GREEN], 1)

    def test_cannot_overpay(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        give(s, 1, K(C.BLUE, C.OPEN))
        s.apply_play(K(C.GREEN, C.OPEN))
        with self.assertRaises(IllegalAction):
            s.apply_auction_result(winner=2, price=101)

    def test_cannot_play_card_not_in_hand(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        with self.assertRaises(IllegalAction):
            s.apply_play(K(C.BLUE, C.OPEN))

    def test_turn_skips_players_without_cards(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        give(s, 3, K(C.BLUE, C.OPEN))  # idx1, idx2 は手札なし
        s.apply_play(K(C.GREEN, C.OPEN))
        s.apply_auction_result(0, 0)
        self.assertEqual(s.turn, 3)


class TestDoubleAuction(unittest.TestCase):
    def test_seller_adds_second(self):
        s = blank()
        give(s, 0, K(C.PINK, C.DOUBLE), K(C.PINK, C.SEALED))
        give(s, 1, K(C.BLUE, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        self.assertEqual(s.phase, PHASE_SECOND)
        self.assertEqual(s.second_offerer, 0)
        self.assertEqual(s.legal_seconds(0), [K(C.PINK, C.SEALED)])

        s.apply_second(K(C.PINK, C.SEALED))
        self.assertEqual(s.phase, PHASE_AUCTION)
        self.assertEqual(s.lot_type, C.SEALED)  # 方式は2枚目で決まる
        self.assertEqual(sorted(s.lot), sorted([K(C.PINK, C.DOUBLE), K(C.PINK, C.SEALED)]))
        self.assertEqual(s.seller, 0)
        self.assertEqual(s.round_counts[C.PINK], 2)

        s.apply_auction_result(winner=2, price=30)
        self.assertEqual(s.collections[2][C.PINK], 2)  # 2枚とも落札者へ

    def test_second_card_cannot_be_double(self):
        s = blank()
        give(s, 0, K(C.PINK, C.DOUBLE), K(C.PINK, C.DOUBLE))
        give(s, 1, K(C.BLUE, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        self.assertEqual(s.legal_seconds(0), [])
        with self.assertRaises(IllegalAction):
            s.apply_second(K(C.PINK, C.DOUBLE))

    def test_second_card_must_match_artist(self):
        s = blank()
        give(s, 0, K(C.PINK, C.DOUBLE), K(C.BLUE, C.OPEN))
        give(s, 1, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        self.assertEqual(s.legal_seconds(0), [])

    def test_other_player_adds_second_and_becomes_seller(self):
        s = blank()
        give(s, 0, K(C.PINK, C.DOUBLE))
        give(s, 1, K(C.BLUE, C.OPEN))
        give(s, 2, K(C.PINK, C.FIXED))
        s.apply_play(K(C.PINK, C.DOUBLE))
        s.apply_decline_second()  # P0 が出さない
        self.assertEqual(s.second_offerer, 1)
        s.apply_decline_second()  # P1 も出さない
        self.assertEqual(s.second_offerer, 2)
        s.apply_second(K(C.PINK, C.FIXED))
        self.assertEqual(s.seller, 2)  # 2枚目を出した人が競売人
        self.assertEqual(s.lot_type, C.FIXED)

        s.apply_auction_result(winner=3, price=18)
        self.assertEqual(s.money[2], 118)  # 売上は全額 2枚目を出した人へ
        self.assertEqual(s.money[0], 100)
        self.assertEqual(s.collections[3][C.PINK], 2)
        self.assertEqual(s.turn, 1)  # 手番は元の競売人の次から

    def test_nobody_adds_second_is_free_for_owner(self):
        s = blank()
        give(s, 0, K(C.PINK, C.DOUBLE))
        give(s, 1, K(C.BLUE, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        for _ in range(4):
            s.apply_decline_second()
        self.assertEqual(s.collections[0][C.PINK], 1)
        self.assertEqual(s.money, [100] * 4)
        self.assertEqual(s.round_counts[C.PINK], 1)
        self.assertEqual(s.phase, PHASE_PLAY)
        self.assertEqual(s.turn, 1)


class TestRoundEnd(unittest.TestCase):
    def test_fifth_card_ends_round_without_auction(self):
        s = blank()
        s.round_counts[C.BLUE] = 4
        give(s, 0, K(C.BLUE, C.OPEN))
        give(s, 1, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.BLUE, C.OPEN))
        self.assertEqual(s.phase, PHASE_ROUND_END)
        self.assertEqual(s.round_counts[C.BLUE], 5)  # 枚数にはカウントする
        self.assertTrue(all(c[C.BLUE] == 0 for c in s.collections))  # 誰の所有にもならない

    def test_fifth_card_as_double_first_ends_round(self):
        s = blank()
        s.round_counts[C.PINK] = 4
        give(s, 0, K(C.PINK, C.DOUBLE), K(C.PINK, C.OPEN))
        give(s, 1, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        self.assertEqual(s.phase, PHASE_ROUND_END)  # 2枚目を聞かずに終了
        self.assertEqual(s.round_counts[C.PINK], 5)

    def test_fifth_card_as_double_second_ends_round(self):
        s = blank()
        s.round_counts[C.PINK] = 3
        give(s, 0, K(C.PINK, C.DOUBLE), K(C.PINK, C.OPEN))
        give(s, 1, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.PINK, C.DOUBLE))
        self.assertEqual(s.phase, PHASE_SECOND)
        s.apply_second(K(C.PINK, C.OPEN))
        self.assertEqual(s.phase, PHASE_ROUND_END)
        self.assertEqual(s.round_counts[C.PINK], 5)
        self.assertTrue(all(c[C.PINK] == 0 for c in s.collections))  # 2枚とも無所有

    def test_round_ends_when_hands_are_empty(self):
        s = blank()
        give(s, 0, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.GREEN, C.OPEN))
        s.apply_auction_result(1, 5)
        self.assertEqual(s.phase, PHASE_ROUND_END)

    def test_next_round_starts_left_of_last_seller(self):
        s = blank()
        s.round_counts[C.BLUE] = 4
        s.turn = 2
        give(s, 2, K(C.BLUE, C.OPEN))
        give(s, 3, K(C.GREEN, C.OPEN))
        s.apply_play(K(C.BLUE, C.OPEN))  # P2 がラウンドを終わらせた
        s.score_round()
        self.assertEqual(s.turn, 3)  # 左隣

    def test_start_player_skips_empty_hands(self):
        """ラウンド4は配札が無いので、開始プレイヤーの手札が0枚ということが起こる."""
        s = blank()
        s.round_idx = 3
        s.last_turn_player = 0
        s.phase = PHASE_ROUND_END
        give(s, 3, K(C.GREEN, C.OPEN))  # 手札があるのは P3 だけ
        s.round_counts = [1, 0, 0, 0, 0]
        s.score_round()
        self.assertEqual(s.phase, PHASE_GAME_END)  # 4ラウンド終了

        s = blank()
        s.last_turn_player = 0
        s.phase = PHASE_ROUND_END
        give(s, 3, K(C.GREEN, C.OPEN))
        s.round_counts = [1, 0, 0, 0, 0]
        s.score_round()
        s.start_round()
        self.assertEqual(s.turn, 3)  # P1,P2 は手札0枚なので飛ばす

    def test_round_ends_immediately_when_nobody_has_cards(self):
        s = blank()
        s.last_turn_player = 0
        s.phase = PHASE_ROUND_END
        s.round_counts = [1, 0, 0, 0, 0]
        s.score_round()
        s.start_round()
        self.assertEqual(s.phase, PHASE_ROUND_END)

    def test_hands_carry_over_between_rounds(self):
        s = blank()
        s.round_counts[C.BLUE] = 4
        give(s, 0, K(C.BLUE, C.OPEN), K(C.GREEN, C.SEALED))
        s.apply_play(K(C.BLUE, C.OPEN))
        s.score_round()
        self.assertEqual(s.hands[0][K(C.GREEN, C.SEALED)], 1)  # 持ち越す


class TestScoring(unittest.TestCase):
    def test_ranks_and_payout(self):
        s = at_round_end([4, 3, 2, 1, 0])  # 黄1位 緑2位 桃3位 青4位 肌なし
        s.collections[0] = [2, 1, 1, 3, 0]
        s.collections[1] = [0, 2, 0, 0, 0]
        gains = s.score_round()
        self.assertEqual(s.board_value, [30, 20, 10, 0, 0])
        self.assertEqual(gains[0], 2 * 30 + 1 * 20 + 1 * 10)  # 青(4位)は0円
        self.assertEqual(gains[1], 2 * 20)
        self.assertEqual(s.money[0], 190)
        self.assertEqual(s.round_counts, [0] * 5)
        self.assertEqual(s.collections[0], [0] * 5)

    def test_values_accumulate_across_rounds(self):
        s = at_round_end([3, 0, 0, 0, 0])
        s.score_round()
        self.assertEqual(s.board_value[C.YELLOW], 30)

        s.round_counts = [4, 2, 1, 0, 0]
        s.phase = PHASE_ROUND_END
        s.collections[0][C.YELLOW] = 1
        gains = s.score_round()
        self.assertEqual(s.board_value[C.YELLOW], 60)
        self.assertEqual(gains[0], 60)  # 累積額で売れる

    def test_tie_break_by_artist_order(self):
        s = at_round_end([2, 2, 2, 2, 0])  # 同数 -> 黄 緑 桃 が上位
        s.collections[0] = [1, 1, 1, 1, 0]
        gains = s.score_round()
        self.assertEqual(s.board_value, [30, 20, 10, 0, 0])
        self.assertEqual(gains[0], 60)

    def test_zero_card_artist_never_ranks(self):
        s = at_round_end([1, 1, 0, 0, 0])
        s.score_round()
        self.assertEqual(s.board_value, [30, 20, 0, 0, 0])  # 3位は該当なし

    def test_game_ends_after_four_rounds(self):
        s = blank()
        for _ in range(4):
            s.round_counts = [1, 0, 0, 0, 0]
            s.phase = PHASE_ROUND_END
            s.score_round()
        self.assertEqual(s.phase, PHASE_GAME_END)
        self.assertEqual(s.round_idx, 4)


class _Stub:
    """留保価格などを固定値で返すテスト用エージェント."""

    name = "stub"

    def __init__(self, res=0, sealed=0, price=0, accept=False):
        self.res, self.sealed, self.price, self.accept = res, sealed, price, accept

    def choose_play(self, s, p):
        return s.legal_plays(p)[0]

    def choose_second(self, s, p):
        return None

    def reservation(self, s, p, lot, lot_type):
        return self.res

    def sealed_bid(self, s, p, lot):
        return self.sealed

    def fixed_price(self, s, p, lot):
        return self.price

    def fixed_accept(self, s, p, lot, price):
        return self.accept


def run_auction(n, lot_type, agents, seller=0):
    s = blank(n)
    s.lot = [C.kind(C.GREEN, lot_type)]
    s.lot_type = lot_type
    s.seller = seller
    s.phase = PHASE_AUCTION
    return rules.resolve_auction(s, agents, random.Random(0))


class TestResolveAuction(unittest.TestCase):
    def test_open_is_second_price(self):
        agents = [_Stub(res=10), _Stub(res=40), _Stub(res=25), _Stub(res=5)]
        self.assertEqual(run_auction(4, C.OPEN, agents), (1, 26))  # 2番手(25)+1

    def test_open_price_capped_by_winner_value(self):
        agents = [_Stub(res=20), _Stub(res=20), _Stub(res=0), _Stub(res=0)]
        winner, price = run_auction(4, C.OPEN, agents)
        self.assertEqual((winner, price), (0, 20))  # 競売人優先、自分の留保額まで

    def test_open_nobody_bids_seller_takes_free(self):
        agents = [_Stub(res=0) for _ in range(4)]
        self.assertEqual(run_auction(4, C.OPEN, agents, seller=2), (2, 0))

    def test_sealed_tie_favors_seller_then_clockwise(self):
        agents = [_Stub(sealed=20) for _ in range(4)]
        self.assertEqual(run_auction(4, C.SEALED, agents, seller=2), (2, 20))
        agents[2] = _Stub(sealed=0)
        self.assertEqual(run_auction(4, C.SEALED, agents, seller=2), (3, 20))

    def test_fixed_first_taker_from_sellers_left(self):
        agents = [_Stub(price=20), _Stub(accept=False), _Stub(accept=True), _Stub(accept=True)]
        self.assertEqual(run_auction(4, C.FIXED, agents, seller=0), (2, 20))

    def test_fixed_all_decline_seller_must_buy(self):
        agents = [_Stub(price=20)] + [_Stub(accept=False) for _ in range(3)]
        self.assertEqual(run_auction(4, C.FIXED, agents, seller=0), (0, 20))

    def test_fixed_price_capped_by_seller_money(self):
        agents = [_Stub(price=999)] + [_Stub(accept=False) for _ in range(3)]
        self.assertEqual(run_auction(4, C.FIXED, agents, seller=0), (0, 100))


class TestFullGames(unittest.TestCase):
    def test_deal_counts(self):
        for n in (3, 4, 5):
            with self.subTest(n=n):
                s = rules.new_game(n, random.Random(1))
                expected = C.DEAL_TABLE[n][0]
                self.assertEqual([s.hand_size(p) for p in range(n)], [expected] * n)
                self.assertEqual(len(s.deck), 70 - expected * n)

    def test_random_games_complete(self):
        for n in (3, 4, 5):
            for seed in range(60):
                with self.subTest(n=n, seed=seed):
                    agents = [RandomAgent(random.Random(seed * 10 + i)) for i in range(n)]
                    s = rules.play_game(agents, random.Random(seed))
                    self.assertEqual(s.phase, PHASE_GAME_END)
                    self.assertEqual(s.round_idx, 4)
                    self.assertTrue(all(m >= 0 for m in s.money))
                    self.assertLessEqual(sum(sum(h) for h in s.hands) + len(s.deck), 70)

    def test_money_only_leaves_via_the_bank(self):
        """プレイヤー間の支払いでは総額が変わらない. 減るのは競売人自身の落札のときだけ."""
        n = 4
        rng = random.Random(7)
        agents = [RandomAgent(random.Random(100 + i)) for i in range(n)]
        s = rules.new_game(n, rng)
        while s.phase != PHASE_GAME_END:
            if s.phase == PHASE_AUCTION:
                before, seller = sum(s.money), s.seller
                winner, price = rules.resolve_auction(s, agents, rng)
                s.apply_auction_result(winner, price)
                self.assertEqual(sum(s.money), before - (price if winner == seller else 0))
            else:
                rules.step(s, agents, rng)

    def test_clone_is_independent(self):
        s = rules.new_game(4, random.Random(3))
        c = s.clone()
        c.hands[0][0] += 5
        c.money[1] = 0
        c.collections[2][0] = 9
        c.deck.clear()
        self.assertNotEqual(s.hands[0][0], c.hands[0][0])
        self.assertEqual(s.money[1], 100)
        self.assertEqual(s.collections[2][0], 0)
        self.assertTrue(s.deck)


if __name__ == "__main__":
    unittest.main()
