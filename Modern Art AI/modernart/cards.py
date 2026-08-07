"""カードの定義と表記の相互変換.

カード種別は ``artist * 5 + auction_type`` の 0..24 の整数 ``kind`` で表す。
手札・山札は長さ 25 の枚数ベクタ (plain list[int]) で持つ。金額は千円単位。
"""

from __future__ import annotations

# --- 画家 (インデックス昇順がそのまま同数時の優先順位) ---
YELLOW, GREEN, PINK, BLUE, BEIGE = range(5)
N_ARTISTS = 5

ARTIST_JA = ("黄", "緑", "桃", "青", "肌")
ARTIST_NAME = ("黄(Lite Metal)", "緑(Yoko)", "桃(Christin P.)", "青(Karl Gitter)", "肌(Krypto)")

# --- 競り方式 ---
OPEN, ONCE, SEALED, FIXED, DOUBLE = range(5)
N_TYPES = 5

TYPE_JA = ("公開", "一声", "入札", "差値", "ダブル")
TYPE_NAME = ("公開競り", "一声", "入札", "差し値", "ダブル")

N_KINDS = N_ARTISTS * N_TYPES  # 25

# --- デッキ構成 ---
# 行=画家, 列=競り方式 (OPEN, ONCE, SEALED, FIXED, DOUBLE) の順。
# 枚数を直すときはこの表だけを編集すればよい (tests/test_cards.py が整合を検証する)。
DECK_TABLE = (
    (3, 3, 2, 2, 2),  # 黄 12枚
    (3, 3, 3, 2, 2),  # 緑 13枚
    (3, 3, 3, 3, 2),  # 桃 14枚
    (3, 3, 3, 3, 3),  # 青 15枚
    (4, 3, 3, 3, 3),  # 肌 16枚
)

ARTIST_TOTALS = tuple(sum(row) for row in DECK_TABLE)  # (12, 13, 14, 15, 16)
DECK_SIZE = sum(ARTIST_TOTALS)  # 70

#: 長さ25の枚数ベクタとしての完全なデッキ
FULL_DECK: tuple[int, ...] = tuple(
    DECK_TABLE[a][t] for a in range(N_ARTISTS) for t in range(N_TYPES)
)

# --- 決算 ---
RANK_BONUS = (30, 20, 10)  # 1位/2位/3位 (千円単位)
CARDS_TO_END_ROUND = 5  # 同一画家がこの枚数出たらラウンド終了
N_ROUNDS = 4
START_MONEY = 100

#: 人数 -> 各ラウンドの追加配布枚数
DEAL_TABLE = {
    3: (10, 6, 6, 0),
    4: (9, 4, 4, 0),
    5: (8, 3, 3, 0),
}


def kind(artist: int, auction_type: int) -> int:
    return artist * N_TYPES + auction_type


def artist_of(k: int) -> int:
    return k // N_TYPES


def type_of(k: int) -> int:
    return k % N_TYPES


def kind_name(k: int) -> str:
    """例: 12 -> '桃入札'"""
    return ARTIST_JA[artist_of(k)] + TYPE_JA[type_of(k)]


def kinds_of_artist(artist: int) -> range:
    base = artist * N_TYPES
    return range(base, base + N_TYPES)


# --- 表記のパース ---

_ARTIST_ALIASES = {
    YELLOW: ("黄", "黄色", "き", "きいろ", "y", "yellow", "lite", "litemetal", "1"),
    GREEN: ("緑", "緑色", "みどり", "g", "green", "yoko", "2"),
    PINK: ("桃", "桃色", "ピンク", "もも", "p", "pink", "christin", "3"),
    BLUE: ("青", "青色", "あお", "b", "blue", "karl", "gitter", "4"),
    BEIGE: ("肌", "肌色", "はだ", "ベージュ", "s", "k", "beige", "krypto", "5"),
}

_TYPE_ALIASES = {
    OPEN: ("公", "公開", "公開競り", "o", "open", "free", "1"),
    ONCE: ("声", "一声", "ひとこえ", "n", "once", "oncearound", "2"),
    SEALED: ("入", "入札", "密", "密札", "h", "sealed", "hidden", "3"),
    FIXED: ("差", "差値", "差し値", "指値", "指し値", "f", "fixed", "4"),
    DOUBLE: ("倍", "W", "ダブル", "d", "double", "5"),
}

_ARTIST_LOOKUP = {a.lower(): i for i, al in _ARTIST_ALIASES.items() for a in al}
_TYPE_LOOKUP = {a.lower(): i for i, al in _TYPE_ALIASES.items() for a in al}

# 1文字トークン（分割に使う）。数字は色と方式で衝突するので単独文字分割からは除く。
_ARTIST_CHARS = {c for c in _ARTIST_LOOKUP if len(c) == 1 and not c.isdigit()}
_TYPE_CHARS = {c for c in _TYPE_LOOKUP if len(c) == 1 and not c.isdigit()}


class CardParseError(ValueError):
    pass


def parse_artist(text: str) -> int:
    a = _ARTIST_LOOKUP.get(text.strip().lower())
    if a is None:
        raise CardParseError(f"色として解釈できません: {text!r}")
    return a


def parse_type(text: str) -> int:
    t = _TYPE_LOOKUP.get(text.strip().lower())
    if t is None:
        raise CardParseError(f"競り方式として解釈できません: {text!r}")
    return t


def parse_kind(text: str) -> int:
    """'黄公' / 'YO' / '黄-公開' / '黄 公開競り' などを kind に変換する."""
    s = text.strip().replace("-", " ").replace("_", " ").replace("/", " ")
    parts = s.split()
    if len(parts) == 2:
        return kind(parse_artist(parts[0]), parse_type(parts[1]))
    if len(parts) != 1:
        raise CardParseError(f"カードとして解釈できません: {text!r}")

    s = parts[0]
    # 先頭から色、残りを方式として最長一致で切る
    lower = s.lower()
    for cut in range(len(lower) - 1, 0, -1):
        a = _ARTIST_LOOKUP.get(lower[:cut])
        t = _TYPE_LOOKUP.get(lower[cut:])
        if a is not None and t is not None:
            return kind(a, t)
    # 1文字ずつの短縮形 (例: 'yo')
    if len(lower) == 2 and lower[0] in _ARTIST_CHARS and lower[1] in _TYPE_CHARS:
        return kind(_ARTIST_LOOKUP[lower[0]], _TYPE_LOOKUP[lower[1]])
    raise CardParseError(f"カードとして解釈できません: {text!r}")


def parse_hand_list(text: str) -> list[int]:
    """スペース/カンマ区切りのカード列を kind のリストにする."""
    return [parse_kind(tok) for tok in text.replace(",", " ").replace("、", " ").split()]


def parse_hand(text: str) -> list[int]:
    """スペース/カンマ区切りのカード列を長さ25の枚数ベクタにする."""
    counts = [0] * N_KINDS
    for k in parse_hand_list(text):
        counts[k] += 1
    return counts


def format_hand(counts: list[int]) -> str:
    """枚数ベクタを人が読める文字列に。画家順・方式順に並べる."""
    out: list[str] = []
    for k in range(N_KINDS):
        for _ in range(counts[k]):
            out.append(kind_name(k))
    return " ".join(out) if out else "(なし)"


def money(v: int) -> str:
    """内部単位(千円)を表示用文字列にする. 例: 24 -> '24' (=24,000円)"""
    return f"{v}"
