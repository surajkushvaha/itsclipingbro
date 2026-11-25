"""Original hype lexicon: a content-agnostic WordScorer.

Scores transcript tokens 0..1 from hand-picked tiers (English, Hinglish in
Roman script, Devanagari), with regex families for elongated interjections
and small emphasis bonuses. Nothing here is tied to a game, a show or a
streamer: it measures reaction words only.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping

from clipbro.core import WordScorer

# Tier 1.0: shock and disbelief, the strongest reactions.
_SHOCK: tuple[str, ...] = (
    # English
    "omg", "omfg", "ohmygod", "wtf", "wth", "dafaq", "insane", "unbelievable",
    "holyshit",
    # Hinglish
    "bhagwan", "bhagwaan", "baap", "baapre", "baaprey",
    # Devanagari
    "भगवान", "बाप", "बापरे",
)

# Tier 0.8: strong hype, gaming slang and Hinglish address or exclamation words.
_STRONG: tuple[str, ...] = (
    # English
    "crazy", "unreal", "holy", "noway", "damn", "sheesh", "clutch",
    "goated", "pog", "poggers", "banger", "savage", "legendary", "epic",
    "letsgo", "lfg",
    # Hinglish
    "pagal", "paagal", "pagla", "mast", "zabardast", "jabardast",
    "jhakaas", "bawal", "faadu", "dhamaal", "kamaal", "shaandar",
    "bindaas", "gazab", "wah", "oye", "haye", "hatt",
    # Devanagari
    "पागल", "मस्त", "जबरदस्त", "कमाल", "वाह",
    "गजब", "शानदार", "बिंदास",
)

# Tier 0.6: emotion and laughter.
_EMOTION: tuple[str, ...] = (
    # English
    "haha", "hehe", "heh", "ha", "lol", "lmao", "lmfao", "rofl", "kek", "xd",
    "wow", "whoa", "woah", "lit", "wild", "yikes", "oof", "aww", "fire", "cooked",
    "legend", "massive", "gg", "dang",
    # Hinglish
    "ekdum", "bakwas", "mazaa", "maza", "sahi", "bhaiya", "chalo", "uff", "ufff",
    # Devanagari
    "हाय", "सही", "अबे", "उफ्फ", "मजा",
)

# Tier 0.4: weak fillers and interjections.
_WEAK: tuple[str, ...] = (
    # English
    "oh", "ah", "yo", "nah", "wait", "what", "hm", "hmm", "eh", "huh", "hey",
    "um", "uh", "no", "yes", "yeah", "yep", "sus", "really", "seriously",
    "dude", "bruh", "bro",
    # Hinglish fillers that are frequent but only mildly excited
    "bhai", "yaar", "kya", "arre", "kyun", "kaisa", "bas", "suno",
    # Devanagari
    "हे", "ओह", "नहीं", "सुनो", "भाई", "यार", "क्या", "अरे",
)

TIERS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (1.0, _SHOCK),
    (0.8, _STRONG),
    (0.6, _EMOTION),
    (0.4, _WEAK),
)

# Elongated or variant spellings that the flat tiers do not list.
# Each pattern is matched with fullmatch against the normalised word.
_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"h(a|e)(h?(a|e))+"), 0.6),   # haha, hehe, hahahaha
    (re.compile(r"ha+h+"), 0.6),              # hahh, haah
    (re.compile(r"l(o+l+)+"), 0.6),           # lol, lool, lolol
    (re.compile(r"lmao+"), 0.6),              # lmao, lmaoo
    (re.compile(r"omg+"), 1.0),               # omgg
    (re.compile(r"no{2,}"), 0.8),             # noo, nooo
    (re.compile(r"ya+r+"), 0.8),              # yaar, yaaar
    (re.compile(r"arr+e+y?"), 0.8),           # arre, arrey
    (re.compile(r"bro+"), 0.8),               # broo
    (re.compile(r"bha+i+"), 0.8),             # bhaii
    (re.compile(r"wa+h+"), 0.8),              # waah
    (re.compile(r"wo+w+"), 0.6),              # wooow
    (re.compile(r"o+h+"), 0.6),               # ohh, ooh
    (re.compile(r"a+h+"), 0.6),               # ahh, aahh
    (re.compile(r"ayy+"), 0.6),               # ayyy
    (re.compile(r"yoo+"), 0.6),               # yooo
    (re.compile(r"(हा)+"), 0.6),              # हाहा
)

DEFAULT_LEXICON: dict[str, float] = {
    word: score for score, words in TIERS for word in words
}

_EDGE_CHARS = string.punctuation + "“”‘’…–—¡¿।॥"
_APOSTROPHES = ("'", "’")
_REPEAT_RE = re.compile(r"(.)\1{2,}", re.DOTALL)
_SINGLE_RE = re.compile(r"(.)\1+", re.DOTALL)
_ELONGATED_FLOOR = 0.8
_EXCLAIM_BONUS = 0.15
_CAPS_BONUS = 0.1
_MIN_CAPS_LEN = 3


def clean(word: str) -> str:
    """Lowercase, strip edge punctuation and drop apostrophes; repeats are kept."""
    text = word.strip().lower().strip(_EDGE_CHARS)
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, "")
    return text


def normalise(word: str) -> str:
    """Clean a word and cap character repeats at two.

    "Nooooo!" becomes "noo", "let's" becomes "lets" and "yaaar," becomes "yaar".
    """
    return _REPEAT_RE.sub(r"\1\1", clean(word))


def is_elongated(word: str) -> bool:
    """Return True when the raw token repeats some character three or more times."""
    return _REPEAT_RE.search(clean(word)) is not None


def emphasis_bonus(raw: str) -> float:
    """Return the emphasis bonus for a raw token: trailing '!' and ALL-CAPS words."""
    stripped = raw.strip()
    bonus = 0.0
    if stripped.endswith("!"):
        bonus += _EXCLAIM_BONUS
    core = stripped.strip(_EDGE_CHARS)
    if len(core) >= _MIN_CAPS_LEN and core.isupper():
        bonus += _CAPS_BONUS
    return bonus


def _clamp(value: float) -> float:
    """Clamp a value into 0..1."""
    return min(1.0, max(0.0, float(value)))


class LexiconWordScorer(WordScorer):
    """WordScorer backed by a tiered lexicon, regex families and emphasis bonuses.

    Args:
        extra: Personal words and scores, merged over the lexicon. Keys are
            normalised like input words; a score of 0 mutes a word.
        lexicon: Replacement base lexicon. Defaults to DEFAULT_LEXICON. The regex
            families for elongated words stay active either way.
    """

    def __init__(
        self,
        extra: Mapping[str, float] | None = None,
        lexicon: Mapping[str, float] | None = None,
    ) -> None:
        self._table: dict[str, float] = {}
        self._merge(DEFAULT_LEXICON if lexicon is None else lexicon)
        if extra:
            self._merge(extra)

    def _merge(self, entries: Mapping[str, float]) -> None:
        """Add normalised entries to the lookup table, overriding existing keys."""
        for word, score in entries.items():
            key = normalise(word)
            if key:
                self._table[key] = _clamp(score)

    def score(self, word: str) -> float:
        """Return the hype score of a word in 0..1; unknown or empty input scores 0.

        A string with spaces is scored by its best token, so a multi-word phrase
        such as "हे भगवान" scores as its strongest word.
        """
        tokens = word.split()
        if len(tokens) > 1:
            return max((self.score(token) for token in tokens), default=0.0)
        norm = normalise(word)
        if not norm:
            return 0.0
        base = self._base(norm)
        if is_elongated(word):
            base = self._elongated_base(word, base)
        if base <= 0.0:
            return 0.0
        return min(1.0, base + emphasis_bonus(word))

    def _elongated_base(self, word: str, base: float) -> float:
        """Lift an elongated word to at least 0.8 when its squeezed form is known.

        "bhaaai" and "yaaaar" match a pattern once capped at two repeats; "kyaaa"
        only matches once every run is squeezed to a single character.
        """
        squeezed = _SINGLE_RE.sub(r"\1", clean(word))
        known = max(base, self._base(squeezed))
        return max(known, _ELONGATED_FLOOR) if known > 0.0 else base

    def _base(self, norm: str) -> float:
        """Return the base score: exact table entry first, then the regex families."""
        if norm in self._table:
            return self._table[norm]
        return max((score for pattern, score in _PATTERNS if pattern.fullmatch(norm)), default=0.0)

