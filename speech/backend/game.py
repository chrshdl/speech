"""
Superball, a game steered by voice: asked to "play superball", the service
shows a GameCard, whose client lets flowers fall for the unicorn to catch
and steers it with the commands of a COMMANDS capture, "left" and "right".
"""

from speech.backend.messages import GameCard
from speech.intents import Intent, IntentRecognizer

TITLE = "Superball"
# The levels as they are said.
LEVELS = ("one", "two", "three", "four", "five")

INTENT = Intent(
    "PlayGame",
    (
        "play superball",
        "play super ball",
        "let's play superball",
        "let's play super ball",
        "start superball",
        "superball",
        "super ball",
        "play superball level {level}",
        "play super ball level {level}",
        "let's play superball level {level}",
    ),
    {"level": LEVELS},
)

# The commands that steer the unicorn, by the names CommandRecognized
# reports. Young children often say an r as a w, so "right" also as
# "wight": the kid's "right" was heard as "whit", "wegt" or "what".
COMMANDS = [
    Intent("LEFT", ("left", "go left", "move left", "turn left")),
    Intent(
        "RIGHT",
        ("right", "go right", "move right", "turn right", "wight", "go wight"),
    ),
]
# Everyday words said while playing, which compete with the commands, so
# that an utterance more like one of them moves nothing, such as a
# child's "yes", which the model heard as nearly "lest". Words that sound
# like a command as a child says it, such as "what", "we" and "why" for
# "wight", are left out.
OTHERS = (
    "yes", "no", "yeah", "yay", "wow", "oh", "oh no", "okay", "hello", "hi",
    "bye", "he", "it", "is", "look", "stop", "go", "up", "down", "again",
    "play", "game", "fun", "mom", "dad", "oops", "nice", "one", "two",
    "three", "four", "five", "good", "cool", "the", "you", "me", "my", "on",
    "off", "in", "out", "unicorn",
)  # fmt: skip
# The lowest score of a command, see speech/intents.py. Higher takes fewer
# other words and recognizes fewer commands. With the 3,750 hour model
# after three epochs, the first with Speech Commands, -1.5 took 34 of its
# 10,197 test clips of other words for one, and -1.25 took 21 and
# recognized 678 rather than 701 of its 808 clips of "left" and "right".
# Of the kid's, both recognized 20 of 24 commands and took none of 12
# other words, while -1.0 recognized only 16. See
# examples/speech_commands/README.md.
THRESHOLD = -1.25


def commands(model, threshold=THRESHOLD):
    """
    The recognizer of the commands in the output of a streaming CTC model,
    such as an OnnxModel.
    """
    return IntentRecognizer(
        COMMANDS,
        model.char_to_int,
        model.blank,
        model.frame_seconds,
        threshold,
        # Each command after the first is transcribed within the stream.
        spaced=True,
        others=OTHERS,
    )


def level(said):
    """The level of the word said for it, 1 if none."""
    return LEVELS.index(said) + 1 if said in LEVELS else 1


class Game:
    """Answers a request to play with the card that starts the game."""

    async def card(self, card_id, slots):
        return GameCard(id=card_id, title=TITLE, level=level(slots.get("level")))

    async def close(self):
        pass
