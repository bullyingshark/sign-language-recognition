import time


class SentenceBuilder:
    """Accumulates words into a sentence and prevents the same sign from being
    added many times in a row (cooldown_sec pause)."""

    def __init__(self, cooldown_sec: float = 2.0) -> None:
        self._cooldown = cooldown_sec
        self._words: list[str] = []
        self._last_time = 0.0

    def update(self, word: str) -> bool:
        """Try to append a word. Returns True if it was appended."""
        now = time.monotonic()
        is_repeat = bool(self._words) and self._words[-1] == word
        if is_repeat and now - self._last_time < self._cooldown:
            return False
        self._words.append(word)
        self._last_time = now
        return True

    def clear(self) -> None:
        self._words.clear()

    @property
    def text(self) -> str:
        return " ".join(self._words)