"""Building a sentence from recognized words."""


class SentenceBuilder:
    """Accumulates recognized words into a sentence.

    Repeat guard: the word that was just added is locked and cannot be added
    again until the hands have been out of the frame for `release_frames`
    consecutive frames. This prevents "hello hello hello" while a sign is
    still being held. A different word is always accepted.
    """

    def __init__(self, release_frames: int = 10) -> None:
        self._release_frames = release_frames
        self._words: list[str] = []
        self._locked: str | None = None        # last added word, blocked until hands go down
        self._frames_without_hands = 0

    def update(self, word: str) -> bool:
        """Try to append a word. Returns True if it was appended."""
        if word == self._locked:
            return False
        self._words.append(word)
        self._locked = word
        return True

    def observe_hands(self, visible: bool) -> None:
        """Call once per frame. Unlocks the last word after the hands were down long enough."""
        if visible:
            self._frames_without_hands = 0
            return
        self._frames_without_hands += 1
        if self._frames_without_hands >= self._release_frames:
            self._locked = None

    def clear(self) -> None:
        """Start a fresh sentence."""
        self._words.clear()
        self._locked = None
        self._frames_without_hands = 0

    @property
    def text(self) -> str:
        return " ".join(self._words)