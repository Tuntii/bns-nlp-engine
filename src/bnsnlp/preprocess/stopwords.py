"""
Turkish stop words management.
"""

from typing import List, Set

# Common Turkish stop words
# This is a curated list of the most common Turkish stop words
TURKISH_STOP_WORDS = {
    # Articles and determiners
    "bir",
    "bu",
    "şu",
    "o",
    "her",
    # Pronouns
    "ben",
    "sen",
    "biz",
    "siz",
    "onlar",
    "bana",
    "sana",
    "ona",
    "bize",
    "size",
    "onlara",
    "benim",
    "senin",
    "onun",
    "bizim",
    "sizin",
    "onların",
    "beni",
    "seni",
    "onu",
    "bizi",
    "sizi",
    # Conjunctions
    "ve",
    "veya",
    "ya",
    "ile",
    "ama",
    "fakat",
    "ancak",
    "lakin",
    "ki",
    "de",
    "da",
    "mi",
    "mı",
    "mu",
    "mü",
    # Prepositions
    "için",
    "gibi",
    "kadar",
    "göre",
    "üzere",
    "dolayı",
    "ile",
    "den",
    "dan",
    "ten",
    "tan",
    # Common verbs and auxiliaries
    "var",
    "yok",
    "olan",
    "olarak",
    "oldu",
    "olur",
    "etti",
    "eder",
    "etmek",
    "olmak",
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
    # Question words
    "ne",
    "neden",
    "niçin",
    "nasıl",
    "nerede",
    "nereye",
    "nereden",
    "kim",
    "kime",
    "kimi",
    "kimin",
    "kimse",
    "hangi",
    "hangisi",
    # Time and quantity
    "şimdi",
    "sonra",
    "önce",
    "daha",
    "en",
    "çok",
    "az",
    "hiç",
    "hep",
    "her",
    "bazı",
    "birkaç",
    "tüm",
    "bütün",
    # Common adverbs
    "çok",
    "az",
    "daha",
    "en",
    "pek",
    "oldukça",
    "fazla",
    "bile",
    "sadece",
    "yalnız",
    "yalnızca",
    "ancak",
    "belki",
    "mutlaka",
    "kesinlikle",
    # Negation
    "değil",
    "yok",
    "hiç",
    # Other common words
    "şey",
    "yer",
    "zaman",
    "kez",
    "kere",
    "gibi",
    "kadar",
    "beri",
    "itibaren",
    "hem",
    "ya",
    "veya",
    "yahut",
    "ise",
    "eğer",
    "madem",
}


class StopWords:
    """Manager for Turkish stop words.

    Provides functionality to load, manage, and filter stop words.
    """

    def __init__(self, custom_stopwords: Set[str] = None):
        """Initialize stop words manager.

        Args:
            custom_stopwords: Optional set of custom stop words to add
        """
        self.stopwords = TURKISH_STOP_WORDS.copy()

        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def add_stopwords(self, words: Set[str]) -> None:
        """Add custom stop words.

        Args:
            words: Set of words to add to stop words
        """
        self.stopwords.update(words)

    def remove_stopwords(self, words: Set[str]) -> None:
        """Remove words from stop words list.

        Args:
            words: Set of words to remove from stop words
        """
        self.stopwords.difference_update(words)

    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stop word.

        Args:
            word: Word to check

        Returns:
            True if word is a stop word, False otherwise
        """
        # Use Turkish-aware lowercase for proper handling of İ/I
        from .normalizer import TurkishNormalizer

        normalizer = TurkishNormalizer()
        word_lower = normalizer.turkish_lower(word)
        return word_lower in self.stopwords

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from a list of tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of tokens with stop words removed
        """
        return [token for token in tokens if not self.is_stopword(token)]

    def get_stopwords(self) -> Set[str]:
        """Get the current set of stop words.

        Returns:
            Set of stop words
        """
        return self.stopwords.copy()


def load_turkish_stopwords() -> Set[str]:
    """Load default Turkish stop words.

    Returns:
        Set of Turkish stop words
    """
    return TURKISH_STOP_WORDS.copy()
