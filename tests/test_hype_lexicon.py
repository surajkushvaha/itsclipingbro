"""Tests for clipbro.hype.lexicon.LexiconWordScorer."""

from __future__ import annotations

import unittest

from clipbro.core import WordScorer
from clipbro.hype.lexicon import DEFAULT_LEXICON, TIERS, LexiconWordScorer, normalise


class TierTests(unittest.TestCase):
    """Known words land in the expected tier."""

    def setUp(self) -> None:
        """Create a default scorer."""
        self.scorer = LexiconWordScorer()

    def test_shock_tier(self) -> None:
        """Shock words score 1.0."""
        for word in ("omg", "wtf", "insane", "unbelievable", "bhagwan", "baap"):
            with self.subTest(word=word):
                self.assertAlmostEqual(self.scorer.score(word), 1.0)

    def test_strong_tier(self) -> None:
        """Strong hype and Hinglish words score 0.8."""
        for word in ("crazy", "letsgo", "pagal", "mast", "zabardast", "jhakaas"):
            with self.subTest(word=word):
                self.assertAlmostEqual(self.scorer.score(word), 0.8)

    def test_emotion_tier(self) -> None:
        """Laughs and emotion words score 0.6."""
        for word in ("haha", "lol", "lmao", "hahaha", "hehe", "lit", "wow", "whoa"):
            with self.subTest(word=word):
                self.assertAlmostEqual(self.scorer.score(word), 0.6)

    def test_weak_tier(self) -> None:
        """Weak fillers score 0.4."""
        for word in ("oh", "yo", "nah", "wait", "bhai", "yaar", "kya", "bro", "arre", "dude"):
            with self.subTest(word=word):
                self.assertAlmostEqual(self.scorer.score(word), 0.4)

    def test_everyday_fillers_stay_below_detector_threshold(self) -> None:
        """Plain Hinglish fillers score under the 0.5 hype-event threshold."""
        for word in ("bhai", "yaar", "kya", "bro", "arre", "यार", "भाई"):
            with self.subTest(word=word):
                self.assertLess(self.scorer.score(word), 0.5)

    def test_removed_conversational_words_score_zero(self) -> None:
        """Pure conversational words carry no excitement and score 0."""
        for word in ("acha", "accha", "haan", "matlab", "dekho", "nahi", "lets"):
            with self.subTest(word=word):
                self.assertEqual(self.scorer.score(word), 0.0)

    def test_tiers_do_not_overlap(self) -> None:
        """No word is listed in two tiers, so each has one definite score."""
        words = [word for _, tier in TIERS for word in tier]
        self.assertEqual(len(words), len(set(words)))
        self.assertEqual(len(DEFAULT_LEXICON), len(words))

    def test_lexicon_size(self) -> None:
        """The built-in lexicon has a useful but bounded size."""
        self.assertGreaterEqual(len(DEFAULT_LEXICON), 80)
        self.assertLessEqual(len(DEFAULT_LEXICON), 150)


class ElongationTests(unittest.TestCase):
    """Repeated letters are collapsed and matched by pattern."""

    def setUp(self) -> None:
        """Create a default scorer."""
        self.scorer = LexiconWordScorer()

    def test_normalise_caps_repeats_at_two(self) -> None:
        """Runs of three or more identical characters shrink to two."""
        self.assertEqual(normalise("nooooo"), "noo")
        self.assertEqual(normalise("Yaaaar,"), "yaar")

    def test_normalise_apostrophe(self) -> None:
        """Apostrophes are dropped so let's and lets match."""
        self.assertEqual(normalise("let's"), "lets")

    def test_elongated_no(self) -> None:
        """Elongated no is a strong reaction."""
        self.assertAlmostEqual(self.scorer.score("nooooo"), 0.8)

    def test_elongated_laugh(self) -> None:
        """Long haha chains match the laugh pattern."""
        self.assertAlmostEqual(self.scorer.score("hahahahaha"), 0.6)
        self.assertAlmostEqual(self.scorer.score("HAHAHAHAHA"), 0.6 + 0.1)

    def test_elongated_yaar(self) -> None:
        """Stretched yaar matches the strong tier."""
        self.assertAlmostEqual(self.scorer.score("yaaaar"), 0.8)

    def test_elongated_emphasis_reaches_strong(self) -> None:
        """Three or more repeats lift a filler to at least 0.8."""
        for word in ("bhaaai", "yaaaar", "kyaaa", "arreee", "brooo"):
            with self.subTest(word=word):
                self.assertGreaterEqual(self.scorer.score(word), 0.8)

    def test_elongation_needs_a_known_base(self) -> None:
        """A stretched word with no lexicon or pattern match still scores 0."""
        self.assertEqual(self.scorer.score("zzzzzq"), 0.0)

    def test_elongated_arre_and_omg(self) -> None:
        """Stretched arre and omg match their patterns."""
        self.assertAlmostEqual(self.scorer.score("arrrre"), 0.8)
        self.assertAlmostEqual(self.scorer.score("omggggg"), 1.0)

    def test_lets_variants(self) -> None:
        """let's and lets are equal and score 0; letsgo is strong."""
        self.assertEqual(self.scorer.score("let's"), 0.0)
        self.assertEqual(self.scorer.score("lets"), 0.0)
        self.assertAlmostEqual(self.scorer.score("letsgo"), 0.8)


class BonusTests(unittest.TestCase):
    """Emphasis bonuses and clamping."""

    def setUp(self) -> None:
        """Create a default scorer."""
        self.scorer = LexiconWordScorer()

    def test_trailing_exclamation_adds_bonus(self) -> None:
        """A trailing ! adds 0.15 to a known word."""
        self.assertAlmostEqual(self.scorer.score("wow!"), 0.6 + 0.15)

    def test_caps_adds_bonus(self) -> None:
        """An ALL-CAPS word of three or more letters adds 0.1."""
        self.assertAlmostEqual(self.scorer.score("WOW"), 0.6 + 0.1)

    def test_short_caps_gets_no_bonus(self) -> None:
        """Caps words shorter than three letters get no bonus."""
        self.assertAlmostEqual(self.scorer.score("OH"), 0.4)

    def test_bonus_clamped_to_one(self) -> None:
        """Bonuses never push a score above 1.0."""
        self.assertLessEqual(self.scorer.score("WTF!"), 1.0)
        self.assertAlmostEqual(self.scorer.score("OMG!!!"), 1.0)
        self.assertAlmostEqual(self.scorer.score("crazy!"), 0.95)

    def test_all_scores_in_range(self) -> None:
        """Every lexicon entry, with and without bonuses, stays in 0..1."""
        for word in DEFAULT_LEXICON:
            for form in (word, word.upper() + "!"):
                with self.subTest(form=form):
                    self.assertTrue(0.0 <= self.scorer.score(form) <= 1.0)


class UnknownTests(unittest.TestCase):
    """Unknown and empty input scores zero."""

    def setUp(self) -> None:
        """Create a default scorer."""
        self.scorer = LexiconWordScorer()

    def test_unknown_word_is_zero(self) -> None:
        """A word outside the lexicon scores 0."""
        self.assertEqual(self.scorer.score("zxqvwk"), 0.0)

    def test_unknown_caps_with_bang_is_zero(self) -> None:
        """Emphasis alone does not make an unknown word hype."""
        self.assertEqual(self.scorer.score("ZXQVWK!"), 0.0)

    def test_empty_and_blank_are_zero(self) -> None:
        """Empty strings, whitespace and bare punctuation score 0."""
        for word in ("", "   ", "!!!", "..."):
            with self.subTest(word=repr(word)):
                self.assertEqual(self.scorer.score(word), 0.0)


class ExtraTests(unittest.TestCase):
    """User-supplied words override or extend the lexicon."""

    def test_extra_adds_new_word(self) -> None:
        """A personal catchphrase scores with its own value."""
        scorer = LexiconWordScorer(extra={"Bhairav": 0.9})
        self.assertAlmostEqual(scorer.score("bhairav"), 0.9)
        self.assertAlmostEqual(scorer.score("BHAIRAV!"), 1.0)

    def test_extra_overrides_builtin(self) -> None:
        """An extra entry replaces the built-in score."""
        scorer = LexiconWordScorer(extra={"lol": 0.2})
        self.assertAlmostEqual(scorer.score("lol"), 0.2)

    def test_extra_zero_mutes_word(self) -> None:
        """An extra score of 0 silences a built-in word."""
        scorer = LexiconWordScorer(extra={"yo": 0.0})
        self.assertEqual(scorer.score("yo"), 0.0)

    def test_extra_normalised_like_input(self) -> None:
        """Extra keys with punctuation or repeats match normalised input."""
        scorer = LexiconWordScorer(extra={"Ghoomo!!": 0.7})
        self.assertAlmostEqual(scorer.score("ghoomo"), 0.7)

    def test_extra_clamped(self) -> None:
        """Out-of-range extra values are clamped into 0..1."""
        scorer = LexiconWordScorer(extra={"boss": 5.0})
        self.assertAlmostEqual(scorer.score("boss"), 1.0)

    def test_custom_lexicon_replaces_default(self) -> None:
        """A supplied lexicon replaces the built-in one."""
        scorer = LexiconWordScorer(lexicon={"x": 0.5})
        self.assertAlmostEqual(scorer.score("x"), 0.5)
        self.assertEqual(scorer.score("wtf"), 0.0)


class DevanagariTests(unittest.TestCase):
    """Devanagari words are scored like Roman-script ones."""

    def setUp(self) -> None:
        """Create a default scorer."""
        self.scorer = LexiconWordScorer()

    def test_devanagari_tiers(self) -> None:
        """Devanagari words hit their tiers."""
        self.assertAlmostEqual(self.scorer.score("भगवान"), 1.0)
        self.assertAlmostEqual(self.scorer.score("यार"), 0.4)
        self.assertAlmostEqual(self.scorer.score("हे"), 0.4)

    def test_devanagari_with_danda(self) -> None:
        """A trailing Devanagari danda is stripped before lookup."""
        self.assertAlmostEqual(self.scorer.score("भगवान।"), 1.0)

    def test_devanagari_laugh_pattern(self) -> None:
        """Repeated हा laughs match the laugh pattern."""
        self.assertAlmostEqual(self.scorer.score("हाहाहा"), 0.6)

    def test_multi_word_phrase_scores_best_token(self) -> None:
        """A phrase scores as its strongest token, so हे भगवान scores 1.0."""
        self.assertAlmostEqual(self.scorer.score("हे भगवान"), 1.0)


class ProtocolTests(unittest.TestCase):
    """The scorer satisfies the WordScorer port."""

    def test_is_word_scorer(self) -> None:
        """isinstance against the port holds."""
        self.assertTrue(isinstance(LexiconWordScorer(), WordScorer))


if __name__ == "__main__":
    unittest.main()
