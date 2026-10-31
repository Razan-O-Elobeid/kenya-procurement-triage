"""Domain constants shared across modules."""

# three competition levels for procurement method
METHOD_LEVEL = {
    "open": 0,
    "selective": 1,
    "AlternativeSelection": 1,
    "communityparticipartion": 1,  # spelling as in the release
    "direct": 2,
    "SpeciallyPermitted": 2,
}
LEVEL_NAMES = {0: "open", 1: "restricted", 2: "non-competitive"}
