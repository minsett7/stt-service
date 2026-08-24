from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

MYANMAR_DIGITS = str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789")
PROTECTED_VALUE_PATTERN = re.compile(r"[0-9၀-၉]+(?:[,:./-][0-9၀-၉]+)*")


def extract_protected_values(text: str) -> list[str]:
    """Extract numeric values in source order, including Myanmar numeral forms."""
    return [match.group(0).translate(MYANMAR_DIGITS).replace(",", "") for match in PROTECTED_VALUE_PATTERN.finditer(text)]


def protected_values_match(raw: str, corrected: str) -> bool:
    return extract_protected_values(raw) == extract_protected_values(corrected)


def correction_change_ratio(raw: str, corrected: str) -> float:
    if not raw and not corrected:
        return 0.0
    return 1 - SequenceMatcher(a=raw, b=corrected, autojunk=False).ratio()

