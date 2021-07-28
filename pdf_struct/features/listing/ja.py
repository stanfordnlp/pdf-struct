from pdf_struct.features.listing import en
from pdf_struct.features.listing.base import BaseSectionNumber, section_pattern, \
    register_section_pattern


# FIXME: Add Japanese specific patterns

def get_text_body_indent_ja(text: str):
    m = en._PAT_ALL_LISTING.match(text)
    return len(m.group(0))


@register_section_pattern('arabic', en.PATS_NUM, int)
@register_section_pattern('roman_upper', en.PATS_ROMAN_UPPER, en.roman_to_int)
@register_section_pattern('roman_lower', en.PATS_ROMAN_LOWER, en.roman_to_int)
@register_section_pattern('alph_upper', en.PATS_ALPH_UPPER, en.alphabet_to_int)
@register_section_pattern('alph_lower', en.PATS_ALPH_LOWER, en.alphabet_to_int)
@register_section_pattern('arabic_multilevel', en.PATS_NUM_MULTILEVEL, int)
class SectionNumberJa(BaseSectionNumber):

    @section_pattern()
    def bullet_point(text: str):
        m = en.PAT_BULLET_POINTS.match(text)
        return None if m is None else m.group(0)
