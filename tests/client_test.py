import os
import re
from html.parser import HTMLParser

import pytest

ASSISTANT = os.path.join(os.path.dirname(__file__), "..", "web", "assistant")

VOID = {"meta", "link", "input", "br", "img", "hr"}


class Sprites(HTMLParser):
    """
    Collects the landing page's sprites: the box their style places them at,
    --x, --y, --w and --h, and the viewBox of each svg drawn in that box,
    the sprite's own or, for a div, the svgs in it.
    """

    def __init__(self):
        super().__init__()
        self.sprites = []
        # The sprite each open element is in, None outside of one.
        self.open = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        sprite = self.open[-1][1] if self.open else None
        if "sprite" in (attrs.get("class") or "").split():
            style = dict(
                re.findall(r"--([xywh]):\s*(-?[\d.]+)", attrs.get("style", ""))
            )
            box = tuple(float(style[k]) for k in "xywh") if len(style) == 4 else None
            sprite = {"tag": tag, "class": attrs["class"], "box": box, "viewBoxes": []}
            self.sprites.append(sprite)
        if tag == "svg" and sprite is not None and "viewbox" in attrs:
            sprite["viewBoxes"].append(
                tuple(float(v) for v in attrs["viewbox"].split())
            )
        if tag not in VOID:
            self.open.append((tag, sprite))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.open.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        while self.open:
            if self.open.pop()[0] == tag:
                break


def sprites():
    parser = Sprites()
    with open(os.path.join(ASSISTANT, "index.html")) as f:
        parser.feed(f.read())
    return parser.sprites


def test_the_landing_page_has_sprites():
    assert len(sprites()) > 30


@pytest.mark.parametrize("sprite", sprites(), ids=lambda s: s["class"])
def test_sprites_are_drawn_where_they_are_placed(sprite):
    # The style places the sprite and the viewBox draws in it, so they must
    # be the same box, or what it draws is stretched or moved.
    assert sprite["box"] is not None, "style needs --x, --y, --w and --h"
    assert sprite["viewBoxes"], "a sprite draws in an svg with a viewBox"
    for view_box in sprite["viewBoxes"]:
        assert view_box == sprite["box"]
