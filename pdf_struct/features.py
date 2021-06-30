# Features that are universal to text and PDF
from typing import Optional

import regex as re


def _gt(tb) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


def whereas(text1: Optional[str], text2: Optional[str]):
    if text2 is None:
        return False
    return text2.strip().lower().startswith('whereas')


def colon_ish(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return False
    return text1.strip()[-1] in {'-', ';', ':', ','}


def punctuated(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return True
    return text1.strip().endswith('.')


_PAT_LIST_ISH = re.compile('(?<!^)(?:,|;|and|or) *$', flags=re.IGNORECASE)


def list_ish(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return False
    return _PAT_LIST_ISH.search(text1) is not None


_PAT_THEREFORE = re.compile('now[ ,]+ therefore', flags=re.IGNORECASE)


def therefore(text1: Optional[str], text2: Optional[str]):
    if text2 is None:
        return False
    m = _PAT_THEREFORE.match(text2.strip())
    return m is not None


def all_capital(text: Optional[str]):
    if text is None:
        return -1
    # allow non letters to be there
    return text.upper() == text


def mask_continuation(text1: Optional[str], text2: Optional[str]):
    if text1 is None or text2 is None:
        return False
    return text1.strip().endswith('__') and text2.strip().startswith('__')


def space_separated(text: Optional[str]):
    if text is None:
        return False
    n_spaces = text.strip().count(' ')
    return n_spaces / len(text)
