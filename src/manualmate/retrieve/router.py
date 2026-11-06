"""Decide WHERE to search, from what the parser found.

Result (Route):
  manual_ids  the manuals to search in (never empty: if we know nothing,
              it is all manuals)
  ask         True if more than one manual matches: the app shows the
              user a "Which model do you have?" question with buttons
  options     the buttons: [(manual_id, "Bosch KGN56 fridge-freezer"), ...]
  reason      a short text for logs and the eval ("model", "brand+category", ...)

Rules, in order:
  1. A model name was found -> the manual(s) with the LONGEST model match.
     Brand and category are used to remove manuals that do not fit.
  2. No model, but brand and/or category -> all manuals with that brand
     and/or category.
  3. Nothing -> all manuals.
  One manual -> search it. More than one -> search all of them, and ask.

When the user clicks a button, the app calls route_to(manual_id) and
searches again. The app keeps that manual for the rest of the chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manualmate.retrieve.query_parser import Aliases, ParsedQuery, get_aliases


# More matches than this: the app asks "Which product?" in words, not with buttons.
MAX_OPTIONS = 6


@dataclass
class Route:
    manual_ids: list[str]
    ask: bool
    reason: str
    options: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_single(self) -> bool:
        return len(self.manual_ids) == 1


def _fits(info: dict, parsed: ParsedQuery) -> bool:
    """The manual does not contradict the brand or category in the question."""
    return ((parsed.brand is None or info["brand"] == parsed.brand) and
            (parsed.category is None or info["category"] == parsed.category))


def route(parsed: ParsedQuery, aliases: Aliases | None = None) -> Route:
    aliases = aliases or get_aliases()
    manuals = aliases.manuals

    # 1. Model name.
    fitting = {m: n for m, n in parsed.models.items() if _fits(manuals[m], parsed)}
    if fitting:
        best = max(fitting.values())
        ids = sorted(m for m, n in fitting.items() if n == best)
        reason = "model"
    # 2. Brand and/or category.
    elif parsed.brand or parsed.category:
        ids = sorted(m for m, info in manuals.items() if _fits(info, parsed))
        reason = "+".join(x for x, v in (("brand", parsed.brand),
                                         ("category", parsed.category)) if v)
        if not ids:   # e.g. "Daikin fridge": we have no such manual
            ids, reason = sorted(manuals), reason + " (no manual fits)"
    # 3. Nothing.
    else:
        ids, reason = sorted(manuals), "nothing found"

    ask = len(ids) > 1
    options = ([(m, manuals[m]["name"]) for m in ids]
               if ask and len(ids) <= MAX_OPTIONS else [])
    return Route(manual_ids=ids, ask=ask, reason=reason, options=options)


def route_to(manual_id: str) -> Route:
    """The user picked a manual (a button, or the photo of the label)."""
    return Route(manual_ids=[manual_id], ask=False, reason="user choice")
