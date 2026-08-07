"""ブラウザ用のローカルサーバ.

    python3 -m modernart.server

``modernart/web/index.html`` を配り、JSON の API で盤面と推奨をやり取りする。
AI の中身は CLI と同じ ``Tracker`` / ``Advisor`` をそのまま使う。

考えるのに数秒かかるので、推奨の計算は盤面のロックを持たずに走らせる。
その間に入力されても詰まらないように、結果には計算開始時点の ``version`` を
付けて返し、古くなっていたら画面側で捨てる。
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import traceback
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import cards as C
from .advisor import Advisor, Tracker, TrackerError
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

WEB_DIR = Path(__file__).resolve().parent / "web"

PHASE_NAMES = {
    PHASE_PLAY: "play",
    PHASE_SECOND: "second",
    PHASE_AUCTION: "auction",
    PHASE_ROUND_END: "round_end",
    PHASE_GAME_END: "game_end",
}


def card_catalog() -> list[dict]:
    """25種類のカードの一覧. 画面はこれを使って選択肢を描く."""
    return [
        {
            "kind": k,
            "artist": C.artist_of(k),
            "type": C.type_of(k),
            "artistJa": C.ARTIST_JA[C.artist_of(k)],
            "typeJa": C.TYPE_JA[C.type_of(k)],
            "typeName": C.TYPE_NAME[C.type_of(k)],
            "name": C.kind_name(k),
            "total": C.FULL_DECK[k],
        }
        for k in range(C.N_KINDS)
    ]


class Session:
    """1つの対局。盤面の更新と推奨の計算を仲介する."""

    def __init__(self, params: Params, budget: float, jobs: int, pool, use_affinity: bool):
        self.lock = threading.Lock()
        self.params = params
        self.budget = budget
        self.jobs = jobs
        self.pool = pool
        self.use_affinity = use_affinity
        self.tracker: Tracker | None = None
        self.advisor: Advisor | None = None
        self.version = 0
        self.log: list[str] = []
        self._advice_lock = threading.Lock()

    # ------------------------------------------------------------ 盤面の更新

    def start(self, n: int, hero: int, double_any_artist: bool = False) -> None:
        with self.lock:
            self.tracker = Tracker(n, hero, random.Random(), double_any_artist)
            self.advisor = Advisor(
                self.tracker,
                self.params or Params.load_for_rule(double_any_artist),
                budget=self.budget,
                jobs=self.jobs,
                pool=self.pool,
                use_affinity=self.use_affinity,
            )
            self.version += 1
            rule = "ダブルの2枚目はどの色でも可" if double_any_artist else "ダブルの2枚目は同じ色のみ"
            self.log = [f"{n}人でゲーム開始。あなたは P{hero + 1}（{rule}）"]

    def require(self) -> Tracker:
        if self.tracker is None:
            raise TrackerError("まだゲームが始まっていません")
        return self.tracker

    def apply(self, fn) -> None:
        """盤面を1つ進める. ``fn(tracker)`` を排他して呼ぶ."""
        with self.lock:
            fn(self.require())
            self.version += 1

    def note(self, text: str) -> None:
        self.log.append(text)
        del self.log[:-40]

    # ------------------------------------------------------------ 推奨の計算

    def advice(self) -> dict | None:
        with self.lock:
            t = self.require()
            version = self.version
            if self.advisor is None:
                return None
            advisor = self.advisor
        # ここからはロックを持たない。数秒かかるので、その間も入力を受け付ける
        with self._advice_lock:
            adv = advisor.advise()
        if adv is None:
            return None
        return {
            "version": version,
            "kind": adv.kind,
            "headline": adv.headline,
            "note": adv.note,
            "tie": adv.tie,
            "seconds": round(adv.seconds, 1),
            "samples": adv.samples,
            "rows": [
                {
                    "label": r.label,
                    "option": r.option,
                    "win": r.win_rate,
                    "money": round(r.mean_money, 1),
                    "n": r.n,
                }
                for r in adv.rows
            ],
        }

    # ------------------------------------------------------------ 盤面の書き出し

    def snapshot(self) -> dict:
        with self.lock:
            if self.tracker is None:
                return {"started": False, "cards": card_catalog()}
            t = self.tracker
            s = t.state
            return {
                "started": True,
                "version": self.version,
                "cards": card_catalog(),
                "n": s.n,
                "hero": t.hero,
                "round": min(s.round_idx + 1, 4),
                "phase": PHASE_NAMES[s.phase],
                "turn": s.turn,
                "boardValue": list(s.board_value),
                "roundCounts": list(s.round_counts),
                "money": list(s.money),
                "collections": [list(c) for c in s.collections],
                "handSizes": [s.hand_size(p) for p in range(s.n)],
                "hand": [k for k in range(C.N_KINDS) for _ in range(s.hands[t.hero][k])],
                "unseen": t.unseen(),
                "playedEver": list(s.played_ever),
                "lot": list(s.lot),
                "lotType": s.lot_type,
                "seller": s.seller,
                "lotPrice": s.lot_price,
                "secondOfferer": s.second_offerer,
                "pendingDouble": s.pending_double,
                "doubleAnyArtist": s.double_any_artist,
                "log": list(self.log),
                "prompt": self._prompt(t),
            }

    def _playable(self, t: Tracker, p: int) -> list[int]:
        """``p`` が出せる（と観測上ありうる）カード."""
        s = t.state
        if p == t.hero:
            return [k for k in range(C.N_KINDS) if s.hands[t.hero][k]]
        return [k for k in range(C.N_KINDS) if t.remaining_copies(k) > 0]

    def _prompt(self, t: Tracker) -> dict:
        s = t.state
        hero = t.hero
        me = lambda p: "あなた" if p == hero else f"P{p + 1}"  # noqa: E731

        if s.phase == PHASE_GAME_END:
            return {"kind": "game_end", "message": "ゲーム終了"}

        if s.phase == PHASE_ROUND_END:
            return {"kind": "score", "message": f"ラウンド{s.round_idx + 1} の決算をします"}

        if t.needs_deal():
            need = C.DEAL_TABLE[s.n][s.round_idx]
            label = "初期手札" if s.round_idx == 0 else f"ラウンド{s.round_idx + 1}の追加手札"
            return {
                "kind": "deal",
                "count": need,
                "message": f"あなたの{label} {need}枚を選んでください",
            }

        if s.phase == PHASE_PLAY:
            return {
                "kind": "play",
                "player": s.turn,
                "isHero": s.turn == hero,
                "options": self._playable(t, s.turn),
                "message": f"{me(s.turn)}が出したカードを選んでください",
            }

        if s.phase == PHASE_SECOND:
            p = s.second_offerer
            a = C.artist_of(s.pending_double)
            if p == hero:
                options = s.legal_seconds(hero)
            else:
                pool = range(C.N_KINDS) if s.double_any_artist else C.kinds_of_artist(a)
                options = [
                    k for k in pool if C.type_of(k) != C.DOUBLE and t.remaining_copies(k) > 0
                ]
            return {
                "kind": "second",
                "player": p,
                "isHero": p == hero,
                "options": options,
                "artist": a,
                "message": f"{me(p)}は {C.kind_name(s.pending_double)} に2枚目を足しますか？",
            }

        # 競り
        if s.lot_type == C.FIXED and s.lot_price < 0:
            return {
                "kind": "price",
                "player": s.seller,
                "isHero": s.seller == hero,
                "max": s.money[s.seller],
                "message": f"{me(s.seller)}が提示した額を入れてください",
            }
        return {
            "kind": "result",
            "message": "落札者と金額を入れてください",
            "fixedPrice": s.lot_price if s.lot_type == C.FIXED else None,
            "seller": s.seller,
            "maxByPlayer": list(s.money),
        }


# ------------------------------------------------------------------ HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "modernart"

    def __init__(self, session: Session, *args, **kwargs):
        self.session = session
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # サーバのアクセスログは出さない
        pass

    # --------------------------------------------------------------- 送信

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, name: str) -> None:
        path = (WEB_DIR / name).resolve()
        if not path.is_file() or WEB_DIR not in path.parents:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    # --------------------------------------------------------------- 受信

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._file("index.html")
            elif self.path == "/api/state":
                self._json(self.session.snapshot())
            elif self.path == "/api/advice":
                self._json(self.session.advice() or {})
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except Exception as e:
            self._fail(e)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            self._json(self._dispatch(self.path, data))
        except (TrackerError, IllegalAction, ValueError) as e:
            self._json({"error": str(e), "state": self.session.snapshot()}, code=400)
        except Exception as e:
            self._fail(e)

    def _fail(self, e: Exception) -> None:
        traceback.print_exc()
        self._json({"error": f"サーバ側でエラー: {e}"}, code=500)

    def _dispatch(self, path: str, d: dict) -> dict:
        S = self.session
        if path == "/api/new":
            S.start(int(d["n"]), int(d["hero"]), bool(d.get("doubleAnyArtist")))
            return S.snapshot()

        if path == "/api/deal":
            cards = [int(k) for k in d["cards"]]
            S.apply(lambda t: t.deal(cards))
            S.note(f"配札 {len(cards)}枚")
            return S.snapshot()

        if path == "/api/play":
            k = int(d["kind"])
            player = S.require().state.turn
            S.apply(lambda t: t.play(k))
            S.note(f"P{player + 1} が {C.kind_name(k)} を出品")
            return S.snapshot()

        if path == "/api/second":
            k = None if d.get("kind") is None else int(d["kind"])
            player = S.require().state.second_offerer
            S.apply(lambda t: t.second(k))
            S.note(
                f"P{player + 1} が {C.kind_name(k)} を追加" if k is not None
                else f"P{player + 1} は2枚目を出さない"
            )
            return S.snapshot()

        if path == "/api/price":
            price = int(d["price"])
            S.apply(lambda t: t.announce_price(price))
            S.note(f"提示額 {price}")
            return S.snapshot()

        if path == "/api/result":
            winner, price = int(d["winner"]), int(d["price"])
            seller = S.require().state.seller
            S.apply(lambda t: t.auction_result(winner, price))
            dest = "銀行" if winner == seller else f"P{seller + 1}"
            S.note(f"P{winner + 1} が {price} で落札（支払い先 {dest}）")
            return S.snapshot()

        if path == "/api/score":
            box = {}
            S.apply(lambda t: box.update(gains=t.score_round()))
            S.note("決算: " + " / ".join(f"P{i + 1} +{g}" for i, g in enumerate(box["gains"])))
            # ラウンド4は配札がないので、そのまま手番の整理まで済ませる
            t = S.require()
            if t.state.phase != PHASE_GAME_END and C.DEAL_TABLE[t.n][t.state.round_idx] == 0:
                S.apply(lambda t: t.deal([]))
            return S.snapshot()

        if path == "/api/hand":
            counts = [0] * C.N_KINDS
            for k in d["cards"]:
                counts[int(k)] += 1
            S.apply(lambda t: t.set_hero_hand(counts))
            S.note("手札を修正")
            return S.snapshot()

        if path == "/api/undo":
            ok = {}
            S.apply(lambda t: ok.update(done=t.undo()))
            if ok.get("done"):
                S.note("1つ戻した")
            return S.snapshot()

        raise ValueError(f"未知のAPI: {path}")


def serve(host: str, port: int, session: Session, open_browser: bool = False) -> None:
    handler = partial(Handler, session)
    httpd = ThreadingHTTPServer((host, port), handler)  # ここで待ち受けが始まる
    url = f"http://localhost:{port}/"
    print("\nモダンアート AI アドバイザー")
    print(f"  ブラウザで {url} を開いてください")
    print("  終了は Ctrl+C\n")
    if open_browser:
        # 待ち受けが始まってから開く。先に開くと「接続が拒否されました」になる
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="モダンアート AI アドバイザー (ブラウザ版)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time", type=float, default=12.0, help="1手あたりの思考時間（秒）")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="並列数（0で自動）")
    ap.add_argument("--rollout", choices=("heuristic", "greedy"), default="heuristic")
    ap.add_argument("--no-affinity", action="store_true")
    ap.add_argument("--open", action="store_true", help="起動後にブラウザを自動で開く")
    args = ap.parse_args(argv)

    jobs = args.jobs or default_jobs()
    pool = None
    if jobs > 1:
        from multiprocessing import Pool

        pool = Pool(jobs)
    session = Session(None, args.time, jobs, pool, not args.no_affinity)  # ルールを聞いてから選ぶ
    try:
        serve(args.host, args.port, session, open_browser=args.open)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
