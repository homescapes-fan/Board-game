"""ゲーム状態と、ルールに沿った状態遷移.

``GameState`` は完全情報の状態を持つ (ロールアウト用)。フィールドは list/int のみで
構成し、``clone()`` は手書きの浅いコピーで済ませる (deepcopy は使わない)。

進行はフェーズで表現する::

    PHASE_PLAY    競売人が手札から1枚出す
    PHASE_SECOND  ダブルが出た。2枚目を出すか順に聞いている最中
    PHASE_AUCTION 競りの対象 (lot) が確定し、落札者と価格を待っている
    PHASE_ROUND_END / PHASE_GAME_END
"""

from __future__ import annotations

from .cards import (
    CARDS_TO_END_ROUND,
    DOUBLE,
    FIXED,
    N_ARTISTS,
    N_KINDS,
    N_ROUNDS,
    RANK_BONUS,
    START_MONEY,
    artist_of,
    kinds_of_artist,
    type_of,
)

PHASE_PLAY = 0
PHASE_SECOND = 1
PHASE_AUCTION = 2
PHASE_ROUND_END = 3
PHASE_GAME_END = 4

PHASE_JA = ("出品", "ダブル2枚目", "競り", "ラウンド終了", "ゲーム終了")


class IllegalAction(ValueError):
    pass


class GameState:
    __slots__ = (
        "n",
        "double_any_artist",
        "round_idx",
        "phase",
        "turn",
        "hands",
        "hand_artist",
        "hand_artist_total",
        "money",
        "board_value",
        "round_counts",
        "collections",
        "played_ever",
        "deck",
        "last_turn_player",
        "pending_double",
        "second_asked",
        "second_offerer",
        "lot",
        "lot_type",
        "seller",
        "lot_price",
        "lot_declined",
    )

    def __init__(self, n: int, double_any_artist: bool = False):
        self.n = n
        #: ダブルの2枚目に別の色を出せるルールで遊ぶか（卓によって異なる）
        self.double_any_artist = double_any_artist
        self.round_idx = 0
        self.phase = PHASE_PLAY
        self.turn = 0
        #: 各プレイヤーの手札 (長さ25の枚数ベクタ)
        self.hands = [[0] * N_KINDS for _ in range(n)]
        #: hands の画家別集計 (評価関数が毎回使うので差分維持する)
        self.hand_artist = [[0] * N_ARTISTS for _ in range(n)]
        #: 全員の手札の画家別合計 = 今ラウンドにまだ出うる枚数
        self.hand_artist_total = [0] * N_ARTISTS
        self.money = [START_MONEY] * n
        #: 画家ごとの累積価格 (千円)
        self.board_value = [0] * N_ARTISTS
        #: 今ラウンドの出品枚数 (5枚目=ラウンド終了トリガも含む)
        self.round_counts = [0] * N_ARTISTS
        #: 今ラウンドに各プレイヤーが所有している枚数 [player][artist]
        self.collections = [[0] * N_ARTISTS for _ in range(n)]
        #: 全ラウンド通して場に出た全カード (長さ25). 未確認カードの割り出しに使う
        self.played_ever = [0] * N_KINDS
        #: 未配布の山札 (kind をシャッフル順に並べたリスト。末尾から引く)
        self.deck: list[int] = []
        #: 直近にカードを場に出した手番プレイヤー (次ラウンドの開始プレイヤー決定に使う)
        self.last_turn_player = n - 1

        # --- ダブル進行中の一時状態 ---
        self.pending_double = -1  # 1枚目として出されたダブルカードの kind
        self.second_asked = 0  # 2枚目を聞いた人数
        self.second_offerer = -1  # いま2枚目を聞かれているプレイヤー

        # --- 競り対象 ---
        self.lot: list[int] = []  # 落札者が獲得する kind のリスト (1枚 or 2枚)
        self.lot_type = -1  # 競り方式
        self.seller = -1  # 売上を受け取るプレイヤー
        self.lot_price = -1  # 差し値で宣言済みの価格 (-1 = まだ宣言されていない)
        self.lot_declined = [False] * n  # 差し値ですでに購入を断ったプレイヤー

    # ------------------------------------------------------------------ copy

    def clone(self) -> "GameState":
        s = GameState.__new__(GameState)
        s.n = self.n
        s.double_any_artist = self.double_any_artist
        s.round_idx = self.round_idx
        s.phase = self.phase
        s.turn = self.turn
        s.hands = [h[:] for h in self.hands]
        s.hand_artist = [h[:] for h in self.hand_artist]
        s.hand_artist_total = self.hand_artist_total[:]
        s.money = self.money[:]
        s.board_value = self.board_value[:]
        s.round_counts = self.round_counts[:]
        s.collections = [c[:] for c in self.collections]
        s.played_ever = self.played_ever[:]
        s.deck = self.deck[:]
        s.last_turn_player = self.last_turn_player
        s.pending_double = self.pending_double
        s.second_asked = self.second_asked
        s.second_offerer = self.second_offerer
        s.lot = self.lot[:]
        s.lot_type = self.lot_type
        s.seller = self.seller
        s.lot_price = self.lot_price
        s.lot_declined = self.lot_declined[:]
        return s

    # ------------------------------------------------------------ hand 操作

    def add_card(self, p: int, k: int) -> None:
        a = artist_of(k)
        self.hands[p][k] += 1
        self.hand_artist[p][a] += 1
        self.hand_artist_total[a] += 1

    def remove_card(self, p: int, k: int) -> None:
        if self.hands[p][k] <= 0:
            raise IllegalAction(f"P{p + 1} はそのカードを持っていません")
        a = artist_of(k)
        self.hands[p][k] -= 1
        self.hand_artist[p][a] -= 1
        self.hand_artist_total[a] -= 1

    def set_hand(self, p: int, counts: list[int]) -> None:
        """手札をまるごと置き換える (集計も貼り直す)."""
        for a in range(N_ARTISTS):
            self.hand_artist_total[a] -= self.hand_artist[p][a]
        self.hands[p] = counts[:]
        agg = [0] * N_ARTISTS
        for k in range(N_KINDS):
            if counts[k]:
                agg[artist_of(k)] += counts[k]
        self.hand_artist[p] = agg
        for a in range(N_ARTISTS):
            self.hand_artist_total[a] += agg[a]

    # ----------------------------------------------------------------- query

    def hand_size(self, p: int) -> int:
        return sum(self.hand_artist[p])

    def cards_in_hands(self) -> int:
        return sum(self.hand_artist_total)

    def any_cards_left(self) -> bool:
        return sum(self.hand_artist_total) > 0

    def legal_plays(self, p: int | None = None) -> list[int]:
        """出品できるカード種別 (重複除去済み)."""
        h = self.hands[self.turn if p is None else p]
        return [k for k in range(N_KINDS) if h[k]]

    def legal_seconds(self, p: int) -> list[int]:
        """ダブルの2枚目として出せるカード種別.

        ダブルカードは2枚目にできない。色の制限は卓のルール次第で、
        ``double_any_artist`` が True なら別の色も出せる。
        """
        if self.pending_double < 0:
            return []
        h = self.hands[p]
        if self.double_any_artist:
            return [k for k in range(N_KINDS) if h[k] and type_of(k) != DOUBLE]
        a = artist_of(self.pending_double)
        return [k for k in kinds_of_artist(a) if h[k] and type_of(k) != DOUBLE]

    # ------------------------------------------------------------ transition

    def apply_play(self, k: int) -> None:
        """手番プレイヤーが ``k`` を場に出す."""
        if self.phase != PHASE_PLAY:
            raise IllegalAction(f"出品できるフェーズではありません: {PHASE_JA[self.phase]}")
        p = self.turn
        self.remove_card(p, k)
        self.last_turn_player = p

        if self._place_on_table(k):
            return  # 5枚目でラウンド終了。競りは行われない

        if type_of(k) == DOUBLE:
            self.pending_double = k
            self.second_asked = 0
            self.second_offerer = p
            self.phase = PHASE_SECOND
        else:
            self._open_auction([k], type_of(k), p)

    def apply_second(self, k: int) -> None:
        """いま聞かれているプレイヤーが2枚目を出す. そのプレイヤーが競売人になる."""
        if self.phase != PHASE_SECOND:
            raise IllegalAction("ダブルの2枚目を出すフェーズではありません")
        p = self.second_offerer
        if k not in self.legal_seconds(p):
            raise IllegalAction(f"P{p + 1} はそれを2枚目に出せません")
        self.remove_card(p, k)
        first = self.pending_double
        self.pending_double = -1
        self.second_offerer = -1

        if self._place_on_table(k):
            return  # 5枚目。2枚とも誰の物にもならない

        self._open_auction([first, k], type_of(k), p)

    def apply_decline_second(self) -> None:
        """いま聞かれているプレイヤーが2枚目を出さない."""
        if self.phase != PHASE_SECOND:
            raise IllegalAction("ダブルの2枚目を出すフェーズではありません")
        self.second_asked += 1
        if self.second_asked >= self.n:
            # 誰も出さなかった -> 1枚目を出した本人が無料で獲得
            k = self.pending_double
            self.pending_double = -1
            self.second_offerer = -1
            self.collections[self.turn][artist_of(k)] += 1
            self._finish_lot()
        else:
            self.second_offerer = (self.turn + self.second_asked) % self.n

    def apply_auction_result(self, winner: int, price: int) -> None:
        """競りの結果を適用する. ``winner`` が ``price`` を支払い lot を獲得する."""
        if self.phase != PHASE_AUCTION:
            raise IllegalAction("競りのフェーズではありません")
        if price < 0:
            raise IllegalAction("価格が負です")
        if price > self.money[winner]:
            raise IllegalAction(
                f"P{winner + 1} の所持金 {self.money[winner]} を超える支払い {price} はできません"
            )
        self.money[winner] -= price
        if winner != self.seller:
            self.money[self.seller] += price  # 競売人以外が落札 -> 競売人へ
        # 競売人自身が落札した場合は銀行へ (誰にも入らない)
        col = self.collections[winner]
        for k in self.lot:
            col[artist_of(k)] += 1
        self._finish_lot()

    # ------------------------------------------------------------- internals

    def _place_on_table(self, k: int) -> bool:
        """カードを場に出して枚数を数える. 5枚目ならラウンドを終了させ True を返す."""
        a = artist_of(k)
        self.played_ever[k] += 1
        self.round_counts[a] += 1
        if self.round_counts[a] >= CARDS_TO_END_ROUND:
            self.pending_double = -1
            self.second_offerer = -1
            self.lot = []
            self.lot_type = -1
            self.seller = -1
            self.phase = PHASE_ROUND_END
            return True
        return False

    def _open_auction(self, lot: list[int], lot_type: int, seller: int) -> None:
        self.lot = lot
        self.lot_type = lot_type
        self.seller = seller
        self.lot_price = -1
        self.lot_declined = [False] * self.n
        self.phase = PHASE_AUCTION

    def announce_fixed_price(self, price: int) -> None:
        """差し値の提示額を記録する (実戦では競売人が宣言した額を入れる)."""
        if self.phase != PHASE_AUCTION or self.lot_type != FIXED:
            raise IllegalAction("差し値の競りではありません")
        if not 0 <= price <= self.money[self.seller]:
            raise IllegalAction(f"提示額は 0〜{self.money[self.seller]} の範囲です")
        self.lot_price = price

    def decline_fixed(self, p: int) -> None:
        """差し値で ``p`` が購入を断ったことを記録する."""
        if self.phase != PHASE_AUCTION or self.lot_type != FIXED:
            raise IllegalAction("差し値の競りではありません")
        self.lot_declined[p] = True

    def _finish_lot(self) -> None:
        """1回の出品が片付いた. 次の手番へ進めるか、ラウンドを終える."""
        self.lot = []
        self.lot_type = -1
        self.seller = -1
        self.lot_price = -1
        self.lot_declined = [False] * self.n
        if not self.any_cards_left():
            self.phase = PHASE_ROUND_END
            return
        p = self.turn
        for _ in range(self.n):
            p = (p + 1) % self.n
            if sum(self.hand_artist[p]):
                self.turn = p
                self.phase = PHASE_PLAY
                return
        self.phase = PHASE_ROUND_END  # 到達しないはず

    def start_round(self) -> None:
        """配札が済んだあとに手番を確定させる.

        ラウンド4は配札が無いので、開始プレイヤーの手札が0枚ということが起こる。
        その場合は手札のある次の人まで送る。誰も持っていなければそのラウンドは終わり。
        """
        if self.phase != PHASE_PLAY:
            return
        p = self.turn
        for _ in range(self.n):
            if sum(self.hand_artist[p]):
                self.turn = p
                return
            p = (p + 1) % self.n
        self.phase = PHASE_ROUND_END

    # --------------------------------------------------------------- scoring

    def rank_artists(self) -> list[int]:
        """今ラウンドの上位3画家 (出品0枚の画家は入らない)."""
        candidates = [a for a in range(N_ARTISTS) if self.round_counts[a] > 0]
        candidates.sort(key=lambda a: (-self.round_counts[a], a))
        return candidates[:3]

    def score_round(self) -> list[int]:
        """決算する. 各プレイヤーの獲得額を返し、場をリセットして次ラウンドを開始する."""
        if self.phase != PHASE_ROUND_END:
            raise IllegalAction("ラウンド終了フェーズではありません")

        top = self.rank_artists()
        for i, a in enumerate(top):
            self.board_value[a] += RANK_BONUS[i]

        gains = [0] * self.n
        for p in range(self.n):
            g = 0
            for a in top:
                g += self.collections[p][a] * self.board_value[a]
            gains[p] = g
            self.money[p] += g

        self.round_counts = [0] * N_ARTISTS
        self.collections = [[0] * N_ARTISTS for _ in range(self.n)]
        self.round_idx += 1
        if self.round_idx >= N_ROUNDS:
            self.phase = PHASE_GAME_END
        else:
            # 次ラウンドは「最後に出品したプレイヤーの左隣」から
            self.turn = (self.last_turn_player + 1) % self.n
            self.phase = PHASE_PLAY
        return gains

    # ------------------------------------------------------------------ misc

    def leader_money(self, exclude: int) -> int:
        best = -1 << 30
        for p in range(self.n):
            if p != exclude and self.money[p] > best:
                best = self.money[p]
        return best

    def winners(self) -> list[int]:
        best = max(self.money)
        return [p for p in range(self.n) if self.money[p] == best]

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return (
            f"<GameState R{self.round_idx + 1} {PHASE_JA[self.phase]} "
            f"turn=P{self.turn + 1} counts={self.round_counts} "
            f"value={self.board_value} money={self.money}>"
        )
