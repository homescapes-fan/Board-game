"""観測の記録と、そこからの推奨手の生成.

実戦では相手の手札は見えない。そこで ``Tracker`` は

  * 自分の手札と、これまでに場に出た全カード
  * 全員の所持金・コレクション・価格表・出品枚数
  * 相手の手札の *枚数*

だけを正として持つ。相手の手札の中身は仮置きで、考えるたびに ``determinize`` で
引き直すので、仮置きの中身は結果に影響しない。
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .agents.heuristic import HeuristicAgent
from .agents.pimc import ACCEPT, BID, PLAY, PRICE, SEALED_BID, SECOND, OptionStat, PIMCAgent
from .beliefs import Affinity, as_card_list, sample_hands, unseen_counts
from .cards import (
    ARTIST_JA,
    CARDS_TO_END_ROUND,
    DEAL_TABLE,
    FIXED,
    FULL_DECK,
    N_KINDS,
    N_TYPES,
    SEALED,
    artist_of,
    kind,
    kind_name,
)
from .params import Params
from .state import PHASE_AUCTION, PHASE_PLAY, PHASE_SECOND, GameState


class TrackerError(ValueError):
    pass


class Tracker:
    """実戦の進行を記録する."""

    def __init__(self, n: int, hero: int, rng: random.Random | None = None):
        if not 3 <= n <= 5:
            raise TrackerError("人数は3〜5です")
        if not 0 <= hero < n:
            raise TrackerError(f"自分の席は1〜{n}で指定してください")
        self.n = n
        self.hero = hero
        self.rng = rng or random.Random()
        self.state = GameState(n)
        self.affinity = Affinity(n)
        self._undo: list[tuple] = []
        #: 配札済みの最後のラウンド番号。ラウンド2・3の追加配札を促すのに使う
        self.dealt_round = -1
        self._ref = HeuristicAgent(Params.load_tuned(), self.rng, seed=0)

    # ------------------------------------------------------------------ undo

    @contextmanager
    def _change(self):
        """1操作ぶんの変更をまとめる.

        失敗したら盤面を元に戻し、履歴にも残さない。残してしまうと「戻す」を
        2回押さないと実際には戻らなくなる。
        """
        snap = (self.state.clone(), self.affinity.clone(), self.dealt_round)
        try:
            yield
        except Exception:
            self.state, self.affinity, self.dealt_round = snap
            raise
        self._undo.append(snap)
        if len(self._undo) > 400:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.state, self.affinity, self.dealt_round = self._undo.pop()
        return True

    # ------------------------------------------------------------------ 配札

    def remaining_copies(self, k: int) -> int:
        """まだ場に出ておらず、自分も持っていない ``k`` の枚数."""
        return FULL_DECK[k] - self.state.played_ever[k] - self.state.hands[self.hero][k]

    def needs_deal(self) -> bool:
        """このラウンドぶんの配札がまだ入力されていないか."""
        return self.dealt_round < self.state.round_idx

    def deal(self, hero_cards: list[int]) -> int:
        """ラウンド開始の配札. ``hero_cards`` は自分がもらったカード."""
        s = self.state
        count = DEAL_TABLE[self.n][s.round_idx]
        if count and len(hero_cards) != count:
            raise TrackerError(f"ラウンド{s.round_idx + 1}の配札は{count}枚です（{len(hero_cards)}枚 入力されました）")
        # 途中で失敗して半端に反映されないよう、先に全部確かめる
        need = [0] * N_KINDS
        for k in hero_cards:
            need[k] += 1
            if need[k] > self.remaining_copies(k):
                raise TrackerError(f"{kind_name(k)} が多すぎます（デッキに{FULL_DECK[k]}枚しかありません）")
        with self._change():
            for k in hero_cards:
                s.add_card(self.hero, k)
            for j in range(self.n):
                if j != self.hero:
                    for _ in range(count):
                        s.add_card(j, 0)  # 枚数だけ合わせる。中身は次の行で引き直す
            self.restock()
            s.start_round()
            self.dealt_round = s.round_idx
        return count

    def restock(self) -> None:
        """相手の手札の仮置きを、未確認カードから引き直す."""
        s = self.state
        others = [j for j in range(self.n) if j != self.hero]
        sizes = [s.hand_size(j) for j in others]
        pool = as_card_list(unseen_counts(s, self.hero))
        if sum(sizes) > len(pool):
            raise TrackerError(
                f"相手の手札 計{sum(sizes)}枚に対して未確認カードが{len(pool)}枚しかありません。"
                "どこかで入力が抜けている可能性があります"
            )
        hands, _ = sample_hands(pool, sizes, self.rng)
        for j, cards in zip(others, hands):
            counts = [0] * N_KINDS
            for k in cards:
                counts[k] += 1
            s.set_hand(j, counts)

    def set_hero_hand(self, counts: list[int]) -> None:
        """自分の手札を入力し直す (打ち間違えたとき用)."""
        played = self.state.played_ever
        for k in range(N_KINDS):
            if counts[k] + played[k] > FULL_DECK[k]:
                raise TrackerError(
                    f"{kind_name(k)} が多すぎます"
                    f"（デッキに{FULL_DECK[k]}枚、うち{played[k]}枚はもう場に出ています）"
                )
        with self._change():
            self.state.set_hand(self.hero, counts)
            self.restock()

    # ------------------------------------------------------------------ 進行

    def _ensure_card(self, p: int, k: int) -> None:
        """``p`` が ``k`` を持っていることにする (相手の手札の中身は仮置きなので入れ替えてよい)."""
        s = self.state
        if s.hands[p][k] > 0:
            return
        if self.remaining_copies(k) <= 0:
            raise TrackerError(f"{kind_name(k)} はもう場に出きっています（デッキに{FULL_DECK[k]}枚）")
        for other in range(N_KINDS):
            if s.hands[p][other] > 0:
                s.remove_card(p, other)
                break
        else:
            raise TrackerError(f"P{p + 1} の手札は残り0枚のはずです")
        s.add_card(p, k)

    def play(self, k: int) -> None:
        s = self.state
        if s.phase != PHASE_PLAY:
            raise TrackerError("いま出品するタイミングではありません")
        p = s.turn
        with self._change():
            if p != self.hero:
                self._ensure_card(p, k)
            elif s.hands[p][k] <= 0:
                raise TrackerError(f"あなたは {kind_name(k)} を持っていません")
            s.apply_play(k)
            self.affinity.on_play(p, artist_of(k))

    def second(self, k: int | None) -> None:
        s = self.state
        if s.phase != PHASE_SECOND:
            raise TrackerError("いまダブルの2枚目を聞くタイミングではありません")
        p = s.second_offerer
        with self._change():
            if k is None:
                s.apply_decline_second()
                return
            if p != self.hero:
                self._ensure_card(p, k)
            s.apply_second(k)
            self.affinity.on_play(p, artist_of(k))

    def announce_price(self, price: int) -> None:
        with self._change():
            self.state.announce_fixed_price(price)

    def mark_declines_before_hero(self) -> None:
        """差し値で自分の番が来た = 自分より前の人は全員断っている."""
        s = self.state
        if s.phase != PHASE_AUCTION or s.lot_type != FIXED:
            return
        p = s.seller
        for _ in range(self.n):
            p = (p + 1) % self.n
            if p == self.hero:
                break
            s.decline_fixed(p)

    def auction_result(self, winner: int, price: int) -> None:
        s = self.state
        if s.phase != PHASE_AUCTION:
            raise TrackerError("いま競りのタイミングではありません")
        lot = s.lot[:]
        with self._change():
            s.apply_auction_result(winner, price)
            if winner != self.hero and lot:
                per = price / len(lot)
                ref = self._ref.lot_value(s, lot) / len(lot)
                for k in lot:
                    self.affinity.on_win(winner, artist_of(k), per, ref)

    def score_round(self) -> list[int]:
        with self._change():
            return self.state.score_round()

    # ------------------------------------------------------------------ 情報

    def unseen(self) -> list[int]:
        return unseen_counts(self.state, self.hero)

    def opponent_hand_estimate(self) -> list[float]:
        """未確認カードのうち、相手の手札にある割合 (山札に眠っている分を除く)."""
        pool = self.unseen()
        total = sum(pool)
        in_hands = sum(self.state.hand_size(j) for j in range(self.n) if j != self.hero)
        frac = in_hands / total if total else 0.0
        return [c * frac for c in pool]


# --------------------------------------------------------------------- 推奨

@dataclass
class Advice:
    kind: str
    headline: str
    rows: list[OptionStat] = field(default_factory=list)
    note: str = ""
    samples: int = 0
    seconds: float = 0.0
    #: 候補間の差が誤差の範囲で、評価関数の手を採った (表の1位と推奨がずれる)
    tie: bool = False


class Advisor:
    def __init__(
        self,
        tracker: Tracker,
        params: Params | None = None,
        budget: float = 10.0,
        jobs: int = 1,
        rollout: str = "heuristic",
        pool=None,
        use_affinity: bool = True,
    ):
        self.tracker = tracker
        self.P = params or Params.load_tuned()
        self.budget = budget
        self.use_affinity = use_affinity
        self.brain = PIMCAgent(
            self.P,
            random.Random(0xA17),
            name="advisor",
            seed=7,
            budget=budget,
            rollout=rollout,
            jobs=jobs,
            pool=pool,
        )

    def _sync_weights(self) -> None:
        self.brain.weights = self.tracker.affinity.weights() if self.use_affinity else None

    def advise(self) -> Advice | None:
        """いまの局面で自分が決めることがあれば、その推奨を返す."""
        t = self.tracker
        s = t.state
        hero = t.hero
        self._sync_weights()
        t0 = time.time()

        if s.phase == PHASE_PLAY and s.turn == hero:
            adv = self._advise_play(s, hero)
        elif s.phase == PHASE_SECOND and s.second_offerer == hero:
            adv = self._advise_second(s, hero)
        elif s.phase == PHASE_AUCTION:
            adv = self._advise_auction(s, hero)
        else:
            return None
        if adv is not None:
            adv.seconds = time.time() - t0
            adv.samples = sum(r.n for r in adv.rows)
            adv.tie = self.brain.last_prior_promoted
        return adv

    # ------------------------------------------------------------- 各場面

    def _advise_play(self, s: GameState, hero: int) -> Advice:
        options = s.legal_plays(hero)
        prior = self.brain.fallback.choose_play(s, hero)
        rows = self.brain.search(
            s, hero, PLAY, options, [kind_name(k) for k in options], prior=prior
        )
        best = rows[0]
        return Advice(
            kind="play",
            headline=f"{best.label} を出す",
            rows=rows,
            note=self._play_note(s, best.option),
        )

    def _play_note(self, s: GameState, k: int) -> str:
        a = artist_of(k)
        c = s.round_counts[a]
        name = ARTIST_JA[a]
        if c + 1 >= CARDS_TO_END_ROUND:
            return f"{name}の5枚目。出した瞬間にラウンドが終わり、この絵は競りにかからない"
        # 相手の手札の中身は仮置きなので、枚数は信念のほうから見積もる
        est = self.tracker.opponent_hand_estimate()
        others = sum(est[kind(a, t)] for t in range(N_TYPES))
        mine = s.hand_artist[self.tracker.hero][a]
        return (
            f"{name}は場に{c}枚。自分の手札に{mine}枚、相手の手札には{others:.1f}枚ありそう"
        )

    def _advise_second(self, s: GameState, hero: int) -> Advice:
        legal = s.legal_seconds(hero)
        options = [None] + legal
        labels = ["2枚目を出さない"] + [f"{kind_name(k)} を足す" for k in legal]
        prior = self.brain.fallback.choose_second(s, hero)
        rows = self.brain.search(s, hero, SECOND, options, labels, prior=prior)
        best = rows[0]
        note = (
            "出せば自分が競売人になり、売上を全額もらえる"
            if best.option is not None
            else "誰も足さなければ1枚目を無料で取れる"
        )
        return Advice(kind="second", headline=best.label, rows=rows, note=note)

    def _advise_auction(self, s: GameState, hero: int) -> Advice | None:
        lot, lt = s.lot, s.lot_type
        lot_name = " + ".join(kind_name(k) for k in lot)

        if lt == FIXED:
            if s.lot_price < 0:
                if s.seller != hero:
                    return None  # 相手が提示するのを待っている
                prior = max(0, min(int(self.brain.fallback.fixed_price(s, hero, lot)), s.money[hero]))
                options = self.brain.price_options(s, hero, lot, anchor=prior)
                rows = self.brain.search(
                    s, hero, PRICE, options, [f"{q} で提示" for q in options], prior=prior
                )
                return Advice(
                    kind="price",
                    headline=f"{rows[0].option} で提示する",
                    rows=rows,
                    note=f"誰も買わなければ自分が {rows[0].option} で引き取ることになる（{lot_name}）",
                )
            if s.seller == hero:
                return None  # 提示済み。あとは相手の判断待ち
            self.tracker.mark_declines_before_hero()
            prior = self.brain.fallback.fixed_accept(s, hero, lot, s.lot_price)
            rows = self.brain.search(
                s, hero, ACCEPT, [True, False], ["買う", "見送る"], prior=prior
            )
            best = rows[0]
            margin = rows[0].mean_score - rows[1].mean_score
            return Advice(
                kind="accept",
                headline=f"{s.lot_price} で {'買う' if best.option else '見送る'}",
                rows=rows,
                note=f"差は勝率換算で {margin * 100:.1f} ポイント（{lot_name}）",
            )

        kind = SEALED_BID if lt == SEALED else BID
        if lt == SEALED:
            prior = self.brain.fallback.sealed_bid(s, hero, lot)
        else:
            prior = self.brain.fallback.reservation(s, hero, lot, lt)
        options = self.brain.bid_options(s, hero, lot, lt, anchor=prior)
        verb = "で入札" if lt == SEALED else "まで出す"
        rows = self.brain.search(
            s, hero, kind, options, [f"{q}{verb}" for q in options], prior=prior
        )

        best = rows[0]
        walk = next((r for r in rows if r.option == 0), None)
        gain = ""
        if walk is not None and walk is not best:
            gain = f"降りる場合との差は勝率 {(best.mean_score - walk.mean_score) * 100:+.1f} ポイント。"

        if lt == SEALED:
            headline = f"{best.option} で入札"
            note = f"一位価格なのでこの額をそのまま払う。{gain}（{lot_name}）"
        else:
            headline = f"{best.option} まで競る"
            note = f"これを超えて競り落としても損。{gain}（{lot_name}）"
        return Advice(kind="bid", headline=headline, rows=rows, note=note)
