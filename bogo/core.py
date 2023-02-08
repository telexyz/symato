from __future__ import unicode_literals
from .validation import is_valid_combination
from . import utils, tone, mark
import logging, sys, string


Mark = mark.Mark
Tone = tone.Tone


class _Action:
    ADD_MARK = 2
    ADD_TONE = 1
    ADD_CHAR = 0


TELEX_DEFINITION = {
    "z": ["a^", "o^", "e^"],
    "w": ["u*", "o*", "a+"],
    "d": "d-",
    "f": "\\",
    "s": "/",
    "r": "?",
    "x": "~",
    "j": ".",
}


def _accepted_chars(rules):
    if sys.version_info[0] > 2:
        ascii_letters = \
            string.ascii_letters
    else:
        ascii_letters = \
            string.lowercase + \
            string.uppercase

    return set(ascii_letters + ''.join(rules.keys()) + utils.VOWELS + "đ")


def process_sequence(sequence,
                     rules=None,
                     skip_non_vietnamese=True):
    """\
    Convert a key sequence into a Vietnamese string with diacritical marks.

    Args:
        rules (optional): see docstring for process_key().
        skip_non_vietnamese (optional): see docstring for process_key().

    It even supports continous key sequences connected by separators.
    i.e. process_sequence('con meof.ddieen') should work.
    """
    result = ""
    raw = result
    result_parts = []
    if rules is None:
        rules = TELEX_DEFINITION

    accepted_chars = _accepted_chars(rules)

    for key in sequence:
        if key not in accepted_chars:
            result_parts.append(result)
            result_parts.append(key)
            result = ""
            raw = ""
        else:
            result, raw = process_key(
                string=result,
                key=key,
                fallback_sequence=raw,
                rules=rules,
                skip_non_vietnamese=skip_non_vietnamese)

    result_parts.append(result)
    return ''.join(result_parts)


def process_key(string, key,
                fallback_sequence="", rules=None,
                skip_non_vietnamese=True):
    """Process a keystroke.

    Args:
        string: The previously processed string or "".
        key: The keystroke.
        fallback_sequence: The previous keystrokes.
        rules (optional): A dictionary listing
            transformation rules. Defaults to TELEX_DEFINITION.
        skip_non_vietnamese (optional): Whether to skip results that
            doesn't seem like Vietnamese. Defaults to True.

    Returns a tuple. The first item of which is the processed
    Vietnamese string, the second item is the next fallback sequence.
    The two items are to be fed back into the next call of process_key()
    as `string` and `fallback_sequence`. If `skip_non_vietnamese` is
    True and the resulting string doesn't look like Vietnamese,
    both items contain the `fallback_sequence`.

    >>> process_key('a', 'a', 'a')
    (â, aa)

    `rules` is a dictionary that maps keystrokes to
    their effect string. The effects can be one of the following:

    'a^': a with circumflex (â), only affect an existing 'a family'
    'a+': a with breve (ă), only affect an existing 'a family'
    'e^': e with circumflex (ê), only affect an existing 'e family'
    'o^': o with circumflex (ô), only affect an existing 'o family'
    'o*': o with horn (ơ), only affect an existing 'o family'
    'd-': d with bar (đ), only affect an existing 'd'
    '/': acute (sắc), affect an existing vowel
    '\': grave (huyền), affect an existing vowel
    '?': hook (hỏi), affect an existing vowel
    '~': tilde (ngã), affect an existing vowel
    '.': dot (nặng), affect an existing vowel

    A keystroke entry can have multiple effects, in which case the
    dictionary entry's value should be a list of the possible
    effect strings. Although you should try to avoid this if
    you are defining a custom input method rule.
    """
    # TODO Figure out a way to remove the `string` argument. Perhaps only the
    #      key sequence is needed?
    def default_return():
        return string + key, fallback_sequence + key

    if rules is None:
        rules = TELEX_DEFINITION

    comps = utils.separate(string)

    # Find all possible transformations this keypress can generate
    trans_list = _get_transformation_list(
        key, rules, fallback_sequence)

    # Then apply them one by one
    new_comps = list(comps)
    for trans in trans_list:
        new_comps = _transform(new_comps, trans)

    if new_comps == comps:
        tmp = list(new_comps)
        if tmp == new_comps: fallback_sequence += key
        new_comps = utils.append_comps(new_comps, key)
    else:
        fallback_sequence += key

    if skip_non_vietnamese is True and key.isalpha() and \
            not is_valid_combination(new_comps, final_form=False):
        result = fallback_sequence, fallback_sequence
    else:
        result = "".join(new_comps), fallback_sequence

    return result


def _get_transformation_list(key, im, fallback_sequence):
    """
    Return the list of transformations inferred from the entered key. The
    map between transform types and keys is given TELEX_DEFINITION
    If entered key is not in im, return "+key", add key to current text
    """
    lkey = key.lower()
    if lkey in im:
        if isinstance(im[lkey], list): trans_list = im[lkey]
        else: trans_list = [im[lkey]]
        return trans_list
    else:
        return ['+' + key]


def _get_action(trans):
    """
    Return the action inferred from the transformation `trans`.
    and the parameter going with this action
    An _Action.ADD_MARK goes with a Mark
    while an _Action.ADD_TONE goes with an Tone
    """
    # TODO: VIQR-like convention
    mark_action = {
        '^': (_Action.ADD_MARK, Mark.HAT),
        '+': (_Action.ADD_MARK, Mark.BREVE),
        '*': (_Action.ADD_MARK, Mark.HORN),
        '-': (_Action.ADD_MARK, Mark.BAR),
    }

    tone_action = {
        '\\': (_Action.ADD_TONE, Tone.GRAVE),
        '/': (_Action.ADD_TONE, Tone.ACUTE),
        '?': (_Action.ADD_TONE, Tone.HOOK),
        '~': (_Action.ADD_TONE, Tone.TIDLE),
        '.': (_Action.ADD_TONE, Tone.DOT),
    }

    if trans[0] == '+':
        return _Action.ADD_CHAR, trans[1]
    if len(trans) == 2:
        return mark_action[trans[1]]
    else:
        return tone_action[trans[0]]


def _transform(comps, trans):
    """
    Transform the given string with transform type trans
    """
    # logging.debug("== In _transform(%s, %s) ==", comps, trans)
    components = list(comps)

    action, parameter = _get_action(trans)
    if action == _Action.ADD_MARK and \
            components[2] == "" and \
            mark.strip(components[1]).lower() in ['oe', 'oa'] and trans == "o^":
        action, parameter = _Action.ADD_CHAR, trans[0] # ko bỏ dấu `^` cho `oe, oa`

    if action == _Action.ADD_TONE:
        # logging.debug("add_tone(%s, %s)", components, parameter)
        components = tone.add_tone(components, parameter)
        # 
    elif action == _Action.ADD_MARK and mark.is_valid_mark(components, trans):
        # logging.debug("add_mark(%s, %s)", components, parameter)
        components = mark.add_mark(components, parameter)

        # Handle uơ in "huơ", "thuở", "quở"
        # If the current word has no last consonant and the first consonant
        # is one of "h", "th" and the vowel is "ươ" then change the vowel into
        # "uơ", keeping case and tone. If an alphabet character is then added
        # into the word then change back to "ươ".
        #
        # NOTE: In the dictionary, these are the only words having this strange
        # vowel so we don't need to worry about other cases.
        if tone.remove_tone_string(components[1]).lower() == "ươ" and \
                not components[2] and components[0].lower() in ["", "h", "th", "kh"]:
            # Backup tones
            ac = tone.get_tone_string(components[1])
            components[1] = ("u", "U")[components[1][0].isupper()] + components[1][1]
            components = tone.add_tone(components, ac)

    elif action == _Action.ADD_CHAR:
        if trans[0] == "<":
            if not components[2]:
                # Only allow ư, ơ or ươ sitting alone in the middle part
                # and ['g', 'i', '']. If we want to type giowf = 'giờ', separate()
                # will create ['g', 'i', '']. Therefore we have to allow
                # components[1] == 'i'.
                if (components[0].lower(), components[1].lower()) == ('g', 'i'):
                    components[0] += components[1]
                    components[1] = ''
                if not components[1] or \
                        (components[1].lower(), trans[1].lower()) == ('ư', 'ơ'):
                    components[1] += trans[1]
        else:
            components = utils.append_comps(components, parameter)
            if parameter.isalpha() and \
                    tone.remove_tone_string(components[1]).lower().startswith("uơ"):
                ac = tone.get_tone_string(components[1])
                components[1] = ('ư',  'Ư')[components[1][0].isupper()] + \
                    ('ơ', 'Ơ')[components[1][1].isupper()] + components[1][2:]
                components = tone.add_tone(components, ac)

    if action == _Action.ADD_MARK or (action == _Action.ADD_CHAR and parameter.isalpha()):
        # If there is any tone, remove and reapply it
        # because it is likely to be misplaced in previous transformations
        ac = tone.get_tone_string(components[1])

        if ac != tone.Tone.NONE:
            components = tone.add_tone(components, Tone.NONE)
            components = tone.add_tone(components, ac)

    # logging.debug("After transform: %s", components)
    return components