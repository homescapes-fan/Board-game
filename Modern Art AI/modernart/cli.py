"""対話型のアドバイザ.

    python3 -m modernart

人数と自分の席を決め、あとは「誰が何を出して、いくらで誰が落札したか」を
入力していくだけ。自分が決める場面では、その前に推奨が出る。
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import unicodedata
from pathlib import Path

from . import cards as C
from .advisor import Advice, Advisor, Tracker, TrackerError
from .agents.pimc import default_jobs
from .params import Params
from .state import (
    PHASE_AUCTION,
    PHASE_GAME_END,
    PHASE_PLAY,
    PHASE_ROUND_END,
    PHASE_SECOND,
    IllegalAction,
)

HELP = """
使えるコマンド (どの入力欄でも打てます)
  ?  help        この説明
  state          盤面をもう一度表示
  undo           直前の入力を取り消す
  hand           自分の手札を入力し直す
  unseen         まだ場に出ていないカードの内訳
  save <file>    途中経過を保存
  load <file>    保存した途中経過を読み込む
  quit / exit    終了

カードの書き方
  色 = 黄 緑 桃 青 肌  (y g p b s でも可)
  方式 = 公(公開競り) 声(一声) 入(入札) 差(差し値) 倍(ダブル)
  例: 黄公  緑倍  桃差  青入  肌声 / YO gd  など
  金額は千円単位で入れてください (24 = 24,000円)
"""


#: 「2枚目を出さない」を表す番兵。None は「既定なし」と紛れるので別物にする
NO_SECOND = object()


class Quit(Exception):
    pass


class Restart(Exception):
    """undo / load で局面が変わったので、表示からやり直す."""


# --------------------------------------------------------------------- 表示


def width(text: str) -> int:
    """端末上の表示幅. 日本語は2桁ぶん占める."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, w: int) -> str:
    return text + " " * max(0, w - width(text))


def fmt_board(t: Tracker) -> str:
    s = t.state
    n = s.n
    out = []
    out.append("")
    out.append(f"━━ ラウンド {min(s.round_idx + 1, 4)} / 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (金額は千円)")

    price = "  ".join(
        f"{C.ARTIST_JA[a]}{s.board_value[a]:>3}" if s.board_value[a] else f"{C.ARTIST_JA[a]}  -"
        for a in range(5)
    )
    counts = "  ".join(f"{C.ARTIST_JA[a]}{s.round_counts[a]:>3}" for a in range(5))
    out.append(f"価格表  {price}")
    out.append(f"出品数  {counts}    ← どれかが5枚でラウンド終了")

    header = ["", *(f"P{p + 1}{'(自分)' if p == t.hero else ''}" for p in range(n))]
    rows = [header, ["所持金", *(str(s.money[p]) for p in range(n))]]
    for a in range(5):
        if any(s.collections[p][a] for p in range(n)):
            rows.append([C.ARTIST_JA[a] + "の絵", *(str(s.collections[p][a] or "・") for p in range(n))])
    rows.append(["手札枚数", *(str(s.hand_size(p)) for p in range(n))])
    widths = [max(width(r[c]) for r in rows) + 2 for c in range(n + 1)]
    out.append("")
    for row in rows:
        out.append("".join(pad(cell, w) for cell, w in zip(row, widths)).rstrip())

    out.append("")
    out.append(f"あなたの手札 ({s.hand_size(t.hero)}枚): {C.format_hand(s.hands[t.hero])}")
    return "\n".join(out)


def fmt_unseen(t: Tracker) -> str:
    pool = t.unseen()
    in_hands = sum(t.state.hand_size(j) for j in range(t.n) if j != t.hero)
    total = sum(pool)
    lines = [f"\nまだ見ていないカード {total}枚 (うち相手の手札 {in_hands}枚、残りは未配布の山札)"]
    for a in range(5):
        parts = []
        for tp in range(5):
            c = pool[C.kind(a, tp)]
            if c:
                parts.append(f"{C.TYPE_JA[tp]}{c}")
        n_a = sum(pool[C.kind(a, tp)] for tp in range(5))
        lines.append(f"  {C.ARTIST_JA[a]} {n_a:>2}枚  " + " ".join(parts))
    return "\n".join(lines)


def fmt_advice(adv: Advice) -> str:
    lines = ["", f"  ▶ 推奨: {adv.headline}"]
    if adv.note:
        lines.append(f"    {adv.note}")
    if adv.tie:
        lines.append("    ※ 上位の差は誤差の範囲だったので、評価関数の手を推奨しています")
    lines.append("    " + "─" * 52)
    label_w = max((width(r.label) for r in adv.rows[:9]), default=0)
    for i, r in enumerate(adv.rows[:9]):
        mark = "★" if i == 0 else "  "
        lines.append(
            f"    {mark} {pad(r.label, label_w)}  勝率 {r.win_rate * 100:5.1f}%   "
            f"2位との金差 {r.mean_money:+7.1f}   (n={r.n})"
        )
    if len(adv.rows) > 9:
        lines.append(f"       … 他 {len(adv.rows) - 9} 案")
    lines.append(f"    ({adv.samples} 回のシミュレーション / {adv.seconds:.1f}秒)")
    return "\n".join(lines)


# --------------------------------------------------------------------- 入力


class Console:
    def __init__(self, tracker_box: dict, read=None):
        self.box = tracker_box  # {"t": Tracker} — load で差し替えるので箱で持つ
        self.read = read or input  # テストから差し替えられるようにしておく

    def ask(self, prompt: str, parse, default=None, default_label: str = ""):
        """メタコマンドを捌きつつ、``parse`` が通るまで訊く.

        空入力は ``default`` をそのまま返す。``default`` は parse を通した後の値
        （カードなら kind、金額なら int）を渡すこと。
        """
        suffix = f" [{default_label}]" if default_label else ""
        while True:
            try:
                raw = self.read(f"{prompt}{suffix} > ").strip()
            except EOFError:
                raise Quit from None
            if not raw:
                if default is not None:
                    return default
                continue
            low = raw.lower()
            if low in ("quit", "exit", "q"):
                raise Quit
            if low in ("?", "help", "h"):
                print(HELP)
                continue
            if low == "state":
                print(fmt_board(self.box["t"]))
                continue
            if low == "unseen":
                print(fmt_unseen(self.box["t"]))
                continue
            if low == "undo":
                if self.box["t"].undo():
                    print("  ← 1つ戻しました")
                    raise Restart
                print("  戻せる操作がありません")
                continue
            if low == "hand":
                self._edit_hand()
                raise Restart
            if low.startswith("save "):
                self._save(raw[5:].strip())
                continue
            if low.startswith("load "):
                self._load(raw[5:].strip())
                raise Restart
            try:
                return parse(raw)
            except (ValueError, TrackerError, IllegalAction) as e:
                print(f"  ! {e}")

    def _edit_hand(self):
        t = self.box["t"]
        print(f"  いまの手札: {C.format_hand(t.state.hands[t.hero])}")
        while True:
            try:
                raw = self.read("  新しい手札を全部入れてください > ").strip()
            except EOFError:
                raise Quit from None
            if not raw:
                return
            try:
                t.set_hero_hand(C.parse_hand(raw))
                print(f"  手札を {C.format_hand(t.state.hands[t.hero])} にしました")
                return
            except (ValueError, TrackerError) as e:
                print(f"  ! {e}")

    def _save(self, path: str):
        try:
            Path(path).write_bytes(pickle.dumps(self.box["t"]))
            print(f"  {path} に保存しました")
        except OSError as e:
            print(f"  ! 保存できません: {e}")

    def _load(self, path: str):
        try:
            self.box["t"] = pickle.loads(Path(path).read_bytes())
            print(f"  {path} を読み込みました")
        except (OSError, pickle.PickleError) as e:
            print(f"  ! 読み込めません: {e}")


def parse_player(n: int, hero: int):
    def parse(raw: str) -> int:
        s = raw.strip().lower().lstrip("pP")
        if s in ("自分", "me", "私"):
            return hero
        v = int(s)
        if not 1 <= v <= n:
            raise ValueError(f"1〜{n} で指定してください")
        return v - 1

    return parse


def parse_amount(cap: int):
    def parse(raw: str) -> int:
        v = int(raw.replace(",", "").replace("円", "").strip())
        if v < 0:
            raise ValueError("0以上で入れてください")
        if v > cap:
            raise ValueError(f"所持金 {cap} を超えています（千円単位で入れてください）")
        return v

    return parse


def parse_winner_and_price(n: int, hero: int, money: list[int], fixed_price: int = -1):
    """'2 24' のように 落札者と金額をまとめて受ける.

    ``fixed_price`` が 0 以上なら差し値。金額は宣言額でなければならないので、
    落札者だけの入力も受け付ける。
    """
    pp = parse_player(n, hero)

    def parse(raw: str) -> tuple[int, int]:
        parts = raw.replace(",", " ").split()
        if len(parts) == 1 and fixed_price >= 0:
            parts.append(str(fixed_price))
        if len(parts) != 2:
            raise ValueError("「落札者 金額」の形で入れてください（例: 2 24）")
        w = pp(parts[0])
        price = int(parts[1].replace("円", ""))
        if price < 0:
            raise ValueError("金額が負です")
        if fixed_price >= 0 and price != fixed_price:
            raise ValueError(f"差し値なので金額は提示額 {fixed_price} のはずです")
        if price > money[w]:
            raise ValueError(f"P{w + 1} の所持金 {money[w]} を超えています")
        return w, price

    return parse


# ----------------------------------------------------------------- メイン進行


def setup(con: Console) -> Tracker:
    print("\n=== モダンアート AI アドバイザー ===")
    n = con.ask("人数 (3〜5)", lambda r: _in_range(int(r), 3, 5, "人数は3〜5です"))
    hero = con.ask(
        f"あなたは何番目スタート？ (1〜{n}、1がスタートプレイヤー)",
        lambda r: _in_range(int(r), 1, n, f"1〜{n} で指定してください") - 1,
    )
    any_colour = con.ask(
        "ダブルの2枚目に別の色も出せる卓ですか？ (y/n)",
        lambda r: r.strip().lower() in ("y", "yes", "はい", "1"),
        default=False,
        default_label="Enterで いいえ",
    )
    t = Tracker(n, hero, random.Random(), double_any_artist=any_colour)
    con.box["t"] = t
    count = C.DEAL_TABLE[n][0]
    con.ask(
        f"あなたの初期手札 {count}枚 (例: 黄公 緑倍 桃差 …)",
        lambda r: t.deal(C.parse_hand_list(r)),
    )
    return t


def _in_range(v: int, lo: int, hi: int, msg: str) -> int:
    if not lo <= v <= hi:
        raise ValueError(msg)
    return v


def show_advice(advisor: Advisor) -> Advice | None:
    adv = advisor.advise()
    if adv is not None:
        print(fmt_advice(adv))
    return adv


def step_play(con: Console, t: Tracker, advisor: Advisor) -> None:
    s = t.state
    p = s.turn
    default = None
    label = ""
    if p == t.hero:
        print("\n── あなたの手番（出品）──")
        adv = show_advice(advisor)
        if adv:
            default = adv.rows[0].option
            label = f"Enterで{C.kind_name(default)}"
        prompt = "出したカード"
    else:
        prompt = f"P{p + 1} が出したカード"
    k = con.ask(prompt, C.parse_kind, default=default, default_label=label)
    t.play(k)


def step_second(con: Console, t: Tracker, advisor: Advisor) -> None:
    s = t.state
    p = s.second_offerer
    legal = s.legal_seconds(p)
    first = C.kind_name(s.pending_double)
    # 空入力 = 出さない。推奨が「足す」ならその kind を既定にする
    default, label = NO_SECOND, "出さない"

    if p == t.hero:
        print(f"\n── {first} のダブル：あなたが2枚目を出せます ──")
        if not legal:
            print("  2枚目に出せるカードが手札にないので、出せません")
            t.second(None)
            return
        adv = show_advice(advisor)
        if adv and adv.rows[0].option is not None:
            default = adv.rows[0].option
            label = f"Enterで{C.kind_name(default)}"

    def parse(raw: str):
        if raw.lower() in ("なし", "no", "-", "パス", "pass"):
            return NO_SECOND
        return C.parse_kind(raw)

    ans = con.ask(
        f"P{p + 1} が足す2枚目 (空欄=出さない)", parse, default=default, default_label=label
    )
    t.second(None if ans is NO_SECOND else ans)


def step_auction(con: Console, t: Tracker, advisor: Advisor) -> None:
    s = t.state
    lot = " + ".join(C.kind_name(k) for k in s.lot)
    seller = s.seller
    print(f"\n── 競り: {lot} / {C.TYPE_NAME[s.lot_type]} / 競売人 P{seller + 1}"
          f"{'(自分)' if seller == t.hero else ''} ──")

    if s.lot_type == C.FIXED:
        if s.lot_price < 0:
            default, label = None, ""
            if seller == t.hero:
                adv = show_advice(advisor)
                if adv:
                    default, label = adv.rows[0].option, f"Enterで{adv.rows[0].option}"
                prompt = "あなたが提示する額"
            else:
                prompt = f"P{seller + 1} の提示額"
            price = con.ask(
                prompt, parse_amount(s.money[seller]), default=default, default_label=label
            )
            t.announce_price(price)
            print(f"  提示額: {price}")
        if seller != t.hero:
            show_advice(advisor)  # この額で買うべきか
        print(f"  誰も買わなければ、競売人 P{seller + 1} が {s.lot_price} で引き取ります")
    else:
        show_advice(advisor)  # いくらまで出すか
        print("  誰も競らなければ、競売人が0円で落札です")

    winner, price = con.ask(
        "落札者と金額 (例: 2 24)",
        parse_winner_and_price(s.n, t.hero, s.money, s.lot_price if s.lot_type == C.FIXED else -1),
    )
    t.auction_result(winner, price)
    who = f"P{winner + 1}{'(自分)' if winner == t.hero else ''}"
    dest = "銀行" if winner == seller else f"P{seller + 1}"
    print(f"  → {who} が {price} で落札（支払い先: {dest}）")


def step_round_end(con: Console, t: Tracker) -> None:
    s = t.state
    ranks = s.rank_artists()
    print("\n══ ラウンド終了 ══")
    print("  出品数: " + "  ".join(f"{C.ARTIST_JA[a]}{s.round_counts[a]}" for a in range(5)))
    if ranks:
        print("  順位: " + "  ".join(
            f"{i + 1}位 {C.ARTIST_JA[a]}(+{C.RANK_BONUS[i]})" for i, a in enumerate(ranks)
        ))
    else:
        print("  出品なし")
    gains = t.score_round()
    print("  価格表: " + "  ".join(
        f"{C.ARTIST_JA[a]}{s.board_value[a]}" for a in range(5) if s.board_value[a]
    ))
    for p in range(s.n):
        tag = "(自分)" if p == t.hero else ""
        print(f"  P{p + 1}{tag}: +{gains[p]} → 所持金 {s.money[p]}")

    if s.phase == PHASE_GAME_END:
        return
    count = C.DEAL_TABLE[s.n][s.round_idx]
    if count:
        con.ask(
            f"\nラウンド{s.round_idx + 1} の追加手札 {count}枚",
            lambda r: t.deal(C.parse_hand_list(r)),
        )
    else:
        t.deal([])  # ラウンド4は配札なし。手番だけ整える


def show_result(t: Tracker) -> None:
    s = t.state
    print("\n══ ゲーム終了 ══")
    order = sorted(range(s.n), key=lambda p: -s.money[p])
    for rank, p in enumerate(order, 1):
        tag = "(自分)" if p == t.hero else ""
        print(f"  {rank}位  P{p + 1}{tag}  {s.money[p]}（{s.money[p] * 1000:,}円）")
    if t.hero in s.winners():
        print("\n  🎉 あなたの勝ちです")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="モダンアート AI アドバイザー")
    ap.add_argument("--time", type=float, default=12.0, help="1手あたりの思考時間（秒）")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="並列数（0で自動）")
    ap.add_argument(
        "--rollout", choices=("heuristic", "greedy"), default="heuristic",
        help="ロールアウト方策。1秒の持ち時間では両者に有意差なし。greedy は少し速く粗い",
    )
    ap.add_argument("--no-affinity", action="store_true", help="相手の色の偏りを推定しない")
    ap.add_argument("--load", help="保存した途中経過から再開する")
    args = ap.parse_args(argv)

    jobs = args.jobs or default_jobs()
    box: dict = {"t": None}
    con = Console(box)

    pool = None
    try:
        if jobs > 1:
            from multiprocessing import Pool

            pool = Pool(jobs)
        if args.load:
            box["t"] = pickle.loads(Path(args.load).read_bytes())
            print(f"{args.load} から再開します")
        else:
            setup(con)
        t = box["t"]
        advisor = Advisor(
            t,
            Params.load_for_rule(t.state.double_any_artist),
            budget=args.time,
            jobs=jobs,
            rollout=args.rollout,
            pool=pool,
            use_affinity=not args.no_affinity,
        )
        print(f"（思考時間 {args.time:g}秒 / 並列 {jobs} / '?' でヘルプ）")

        while True:
            t = box["t"]
            advisor.tracker = t
            s = t.state
            if s.phase == PHASE_GAME_END:
                show_result(t)
                return 0
            print(fmt_board(t))
            try:
                if s.phase == PHASE_PLAY:
                    step_play(con, t, advisor)
                elif s.phase == PHASE_SECOND:
                    step_second(con, t, advisor)
                elif s.phase == PHASE_AUCTION:
                    step_auction(con, t, advisor)
                elif s.phase == PHASE_ROUND_END:
                    step_round_end(con, t)
            except Restart:
                continue
            except (TrackerError, IllegalAction) as e:
                print(f"  ! {e}")
    except (Quit, KeyboardInterrupt):
        print("\n終了します")
        return 0
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    sys.exit(main())
