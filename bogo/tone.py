from __future__ import unicode_literals
from . import utils


class Tone:
    GRAVE = 5 # huyền
    ACUTE = 4 # sắc (nhọn = acute :)
    HOOK = 3  # hỏi
    TIDLE = 2 # ngã
    DOT = 1   # nặng
    NONE = 0


def get_tone_char(char):
    index = utils.VOWELS.find(char.lower())
    if (index != -1):
        return 5 - index % 6
    else:
        return Tone.NONE


def get_tone_string(string):
    tones = list(filter(lambda tone: tone != Tone.NONE,
                          map(get_tone_char, string)))
    return tones[-1] if tones else Tone.NONE


def add_tone(components, tone):
    vowel = components[1]
    last_consonant = components[2]
    if tone == Tone.NONE:
        vowel = remove_tone_string(vowel)
        return [components[0], vowel, last_consonant]

    if vowel == "":
        return components

    # raw_string is a list, not a str object
    raw_string = remove_tone_string(vowel).lower()
    new_vowel = ""

    # Ưu tiên `ê` và `ơ`
    index = max(raw_string.find("ê"), raw_string.find("ơ"))
    if index != -1:
        new_vowel = vowel[:index] + add_tone_char(vowel[index], tone) + vowel[index+1:]

    elif len(vowel) == 1 or (len(vowel) == 2 and last_consonant == ""):
        new_vowel = add_tone_char(vowel[0], tone) + vowel[1:]

    else:
        new_vowel = vowel[:1] + add_tone_char(vowel[1], tone) + vowel[2:]

    return [components[0], new_vowel, components[2]]


def add_tone_char(char, tone):
    if char == "": return ""
    case = char.isupper()
    char = char.lower()
    index = utils.VOWELS.find(char)
    if (index != -1):
        index = index - index % 6 + 5
        char = utils.VOWELS[index - tone]
    return utils.change_case(char, case)


def remove_tone_char(char):
    # Remove tone from a single char, if any.
    return add_tone_char(char, Tone.NONE)


def remove_tone_string(string):
    # Remove all tone from a whole string.
    return "".join([add_tone_char(c, Tone.NONE) for c in string])