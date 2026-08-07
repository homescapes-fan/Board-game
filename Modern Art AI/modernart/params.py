"""ヒューリスティックの調整パラメータ.

既定値は手で置いた出発点。``tune.py`` が自己対戦で上書きした値を
``params/tuned.json`` に書き出し、``Params.load_tuned()`` が読み込む。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

_PARAMS_DIR = Path(__file__).resolve().parent.parent / "params"
#: 「ダブルの2枚目は同じ色のみ」で調整した値
TUNED_PATH = _PARAMS_DIR / "tuned.json"
#: 「ダブルの2枚目はどの色でもよい」で調整した値。最適な打ち方が変わるので分けてある
TUNED_ANY_PATH = _PARAMS_DIR / "tuned_any_color.json"


@dataclass(frozen=True)
class Params:
    #: 相手の利得を自分の損として見る重み. 1.0 で完全ゼロサム視
    gamma: float = 1.0

    #: 留保価格の掛け目 (競り方式ごと). 1.0 = 期待価値いっぱいまで出す
    shade_open: float = 0.78
    shade_once: float = 0.70
    shade_sealed: float = 0.68
    shade_fixed: float = 0.80

    #: 一声は競売人が最後に声を出せる分だけ強気にしてよい
    once_seller_bonus: float = 1.25

    #: 「上位3位に入らないかもしれない」不確実性へのペナルティ
    risk: float = 0.30

    #: 手札に残っているカードの将来価値の重み (1枚あたり、その色の評価額に掛ける)
    future_weight: float = 0.30

    #: 差し値の提示額 = 相手の推定留保価格 × これ
    fixed_margin: float = 0.92

    #: ダブルの2枚目を出す最低ライン (見込み売上)
    second_min_revenue: float = 4.0

    #: ロールアウトでの評価のばらつき (対数正規の σ). 0 だと競りが機械的になる
    value_noise: float = 0.14

    #: 現金の重み (ラウンド1〜4). 大きいほどカードを買い渋る
    cash_weight_r1: float = 1.12
    cash_weight_r2: float = 1.08
    cash_weight_r3: float = 1.04
    cash_weight_r4: float = 1.00

    def cash_weight(self, round_idx: int) -> float:
        return (
            self.cash_weight_r1,
            self.cash_weight_r2,
            self.cash_weight_r3,
            self.cash_weight_r4,
        )[min(round_idx, 3)]

    def shade(self, lot_type: int) -> float:
        from .cards import FIXED, ONCE, OPEN, SEALED

        return {
            OPEN: self.shade_open,
            ONCE: self.shade_once,
            SEALED: self.shade_sealed,
            FIXED: self.shade_fixed,
        }[lot_type]

    # --------------------------------------------------------------- 直列化

    def to_vector(self) -> list[float]:
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def from_vector(cls, vec) -> "Params":
        names = [f.name for f in fields(cls)]
        return cls(**{n: float(v) for n, v in zip(names, vec)})

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def save(self, path: Path | None = None) -> None:
        # 既定値を引数に書くと定義時のパスで固定されてしまうので、呼ばれた時に解決する
        path = path or TUNED_PATH
        # 書き込み中に落ちても壊れたファイルが残らないよう、書いてから差し替える
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load_tuned(cls, path: Path | None = None) -> "Params":
        """調整済みパラメータがあれば読む. なければ既定値.

        自己対戦では1ゲームあたり何度も呼ばれるので、更新時刻つきでキャッシュする。
        """
        path = path or TUNED_PATH
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return cls()
        return _load_cached(str(path), stamp)


    @classmethod
    def load_for_rule(cls, double_any_artist: bool) -> "Params":
        """遊ぶルールに合った調整値を読む. 無ければ既定のものにする."""
        if double_any_artist and TUNED_ANY_PATH.exists():
            return cls.load_tuned(TUNED_ANY_PATH)
        return cls.load_tuned()


@lru_cache(maxsize=8)
def _load_cached(path: str, _stamp: int) -> Params:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(Params)}
        return Params(**{k: v for k, v in data.items() if k in known})
    except (OSError, ValueError, TypeError) as e:
        print(f"! {path} を読めないので既定値を使います: {e}", file=sys.stderr)
        return Params()


#: 探索で動かしてよい範囲 (CEM のクリップに使う)
#:
#: 掛け目の上限を 1.15 にしていたら、調整結果が上限に張り付いた。実際に上限の外を
#: 試すと更に強かったので広げてある。公開競りは2位の額+1で決まるので、留保価格を
#: 高くしても普段はそこまで払わずに済む — つまり期待価値より上まで競って良い。
BOUNDS: dict[str, tuple[float, float]] = {
    "gamma": (0.0, 1.5),
    "shade_open": (0.3, 2.2),
    "shade_once": (0.3, 2.2),
    "shade_sealed": (0.3, 2.2),
    "shade_fixed": (0.3, 2.2),
    "once_seller_bonus": (0.8, 2.5),
    "risk": (-0.3, 0.9),
    "future_weight": (-0.8, 1.2),  # 手札は売上にならないので、負まで許す
    "fixed_margin": (0.5, 1.6),
    "second_min_revenue": (0.0, 20.0),
    "value_noise": (0.0, 0.5),
    "cash_weight_r1": (0.6, 1.6),
    "cash_weight_r2": (0.6, 1.6),
    "cash_weight_r3": (0.6, 1.6),
    "cash_weight_r4": (0.6, 1.6),
}
