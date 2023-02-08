from __future__ import unicode_literals
from . import tone, utils
Tone = tone.Tone


class Mark:
    HAT = 4
    HORN = 3
    BREVE = 2
    BAR = 1
    NONE = 0


FAMILY_A = "aăâ"
FAMILY_E = "eê"
FAMILY_O = "oơô"
FAMILY_U = "uư"
FAMILY_D = "dđ"


def get_mark_char(char):
    """
    Get the mark of a single char, if any.
    """
    char = tone.remove_tone_char(char.lower())
    if char == "":
        return Mark.NONE
    if char == "đ":
        return Mark.BAR
    if char in "ă":
        return Mark.BREVE
    if char in "ơư":
        return Mark.HORN
    if char in "âêô":
        return Mark.HAT
    return Mark.NONE


# TODO: Monstrous code. Needs refactoring.
def add_mark(components, mark):
    comp = list(components)
    if mark == Mark.BAR and comp[0] and comp[0][-1].lower() in FAMILY_D:
        comp[0] = add_mark_at(comp[0], len(comp[0])-1, Mark.BAR)
    else:
        # remove all marks and tones in vowel part
        raw_vowel = tone.add_tone(comp, Tone.NONE)[1].lower()
        raw_vowel = "".join([add_mark_char(c, Mark.NONE) for c in raw_vowel])
        if mark == Mark.HAT:
            pos = max(raw_vowel.find("a"), raw_vowel.find("o"), raw_vowel.find("e"))
            comp[1] = add_mark_at(comp[1], pos, Mark.HAT)
        elif mark == Mark.BREVE:
            if raw_vowel != "ua":
                comp[1] = add_mark_at(comp[1], raw_vowel.find("a"), Mark.BREVE)
        elif mark == Mark.HORN:
            if raw_vowel in ("uo", "uoi", "uou"):
                comp[1] = "".join([add_mark_char(c, Mark.HORN) for c in comp[1][:2]]) + comp[1][2:]
            elif raw_vowel == "oa":
                comp[1] = add_mark_at(comp[1], 1, Mark.HORN)
            else:
                pos = max(raw_vowel.find(""), raw_vowel.find("o"))
                comp[1] = add_mark_at(comp[1], pos, Mark.HORN)
    if mark == Mark.NONE:
        if not raw_vowel == comp[1].lower():
            comp[1] = raw_vowel
        elif comp[0] and comp[0][-1] == "đ":
            comp[0] = comp[0][:-1] + "d"
    return comp


def add_mark_at(string, index, mark):
    """
    Add mark to the index-th character of the given string. Return the new string after applying change.
    Notice: index > 0
    """
    if index == -1: return string
    # `eeq`=>`ê` để sửa sai nhanh khi gõ theo thói quen cũ `ee`=>`ê`, tương tự cho oo, aa
    if mark == Mark.HAT and string[index:index+2] in ('oo', 'ee', 'aa'):
        return string[:index] + add_mark_char(string[index], mark) + string[index+2:]
    else:
        return string[:index] + add_mark_char(string[index], mark) + string[index+1:]


def add_mark_char(char, mark):
    """
    Add mark to a single char.
    """
    if char == "":
        return ""
    case = char.isupper()
    ac = tone.get_tone_char(char)
    char = tone.add_tone_char(char.lower(), Tone.NONE)
    new_char = char
    if mark == Mark.HAT:
        if char in FAMILY_A:
            new_char = "â"
        elif char in FAMILY_O:
            new_char = "ô"
        elif char in FAMILY_E:
            new_char = "ê"
    elif mark == Mark.HORN:
        if char in FAMILY_O:
            new_char = "ơ"
        elif char in FAMILY_U:
            new_char = "ư"
    elif mark == Mark.BREVE:
        if char in FAMILY_A:
            new_char = "ă"
    elif mark == Mark.BAR:
        if char in FAMILY_D:
            new_char = "đ"
    elif mark == Mark.NONE:
        if char in FAMILY_A:
            new_char = "a"
        elif char in FAMILY_E:
            new_char = "e"
        elif char in FAMILY_O:
            new_char = "o"
        elif char in FAMILY_U:
            new_char = "u"
        elif char in FAMILY_D:
            new_char = "d"

    new_char = tone.add_tone_char(new_char, ac)
    return utils.change_case(new_char, case)


def is_valid_mark(comps, mark_trans):
    """
    Check whether the mark given by mark_trans is valid to add to the components
    """
    if mark_trans == "*_":
        return True
    components = list(comps)

    if mark_trans[0] == 'd' and components[0] \
            and components[0][-1].lower() in ("d", "đ"):
        return True
    elif components[1] != "" and \
            strip(components[1]).lower().find(mark_trans[0]) != -1:
        return True
    else:
        return False


def remove_mark_char(char):
    """Remove mark from a single character, if any."""
    return add_mark_char(char, Mark.NONE)


def remove_mark_string(string):
    return "".join([remove_mark_char(c) for c in string])


def strip(string):
    """
    Strip a string of all marks and tones.
    """
    return remove_mark_string(tone.remove_tone_string(string))