import unittest

from modernart import cards as C


class TestDeck(unittest.TestCase):
    def test_totals(self):
        self.assertEqual(C.ARTIST_TOTALS, (12, 13, 14, 15, 16))
        self.assertEqual(C.DECK_SIZE, 70)
        self.assertEqual(sum(C.FULL_DECK), 70)

    def test_breakdown_matches_spec(self):
        # (画家, 公開, 一声, 入札, 差し値, ダブル)
        spec = [
            (C.YELLOW, 3, 3, 2, 2, 2),
            (C.GREEN, 3, 3, 3, 2, 2),
            (C.PINK, 3, 3, 3, 3, 2),
            (C.BLUE, 3, 3, 3, 3, 3),
            (C.BEIGE, 4, 3, 3, 3, 3),
        ]
        for a, o, n, h, f, d in spec:
            with self.subTest(artist=C.ARTIST_JA[a]):
                got = (
                    C.FULL_DECK[C.kind(a, C.OPEN)],
                    C.FULL_DECK[C.kind(a, C.ONCE)],
                    C.FULL_DECK[C.kind(a, C.SEALED)],
                    C.FULL_DECK[C.kind(a, C.FIXED)],
                    C.FULL_DECK[C.kind(a, C.DOUBLE)],
                )
                self.assertEqual(got, (o, n, h, f, d))

    def test_kind_roundtrip(self):
        for a in range(5):
            for t in range(5):
                k = C.kind(a, t)
                self.assertEqual(C.artist_of(k), a)
                self.assertEqual(C.type_of(k), t)


class TestParsing(unittest.TestCase):
    CASES = [
        ("黄公", (C.YELLOW, C.OPEN)),
        ("緑倍", (C.GREEN, C.DOUBLE)),
        ("桃差", (C.PINK, C.FIXED)),
        ("青入", (C.BLUE, C.SEALED)),
        ("肌声", (C.BEIGE, C.ONCE)),
        ("YO", (C.YELLOW, C.OPEN)),
        ("gd", (C.GREEN, C.DOUBLE)),
        ("黄 公開競り", (C.YELLOW, C.OPEN)),
        ("肌-入札", (C.BEIGE, C.SEALED)),
        ("ピンク差し値", (C.PINK, C.FIXED)),
        ("あお一声", (C.BLUE, C.ONCE)),
    ]

    def test_parse_kind(self):
        for text, (a, t) in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(C.parse_kind(text), C.kind(a, t))

    def test_parse_hand(self):
        counts = C.parse_hand("黄公 黄公 緑倍, 肌声")
        self.assertEqual(counts[C.kind(C.YELLOW, C.OPEN)], 2)
        self.assertEqual(counts[C.kind(C.GREEN, C.DOUBLE)], 1)
        self.assertEqual(counts[C.kind(C.BEIGE, C.ONCE)], 1)
        self.assertEqual(sum(counts), 4)

    def test_format_hand_roundtrip(self):
        counts = C.parse_hand("肌声 黄公 緑倍")
        self.assertEqual(C.parse_hand(C.format_hand(counts)), counts)

    def test_parse_errors(self):
        for bad in ("むらさき公", "黄ダンス", "", "xyz"):
            with self.subTest(text=bad), self.assertRaises(C.CardParseError):
                C.parse_kind(bad)


if __name__ == "__main__":
    unittest.main()
