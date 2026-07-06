from utils.HTR_CRNN.data.globalvalue.text_comon_values import SOS_STR_TOKEN, EOS_STR_TOKEN


def convert_int_to_chars(indices, char_list, break_on_eos=True, filter_sos=False):
    """
    This function applies a int to character dictionary to every int in a sequence

    Parameters
    ----------
    indices: int array
        A sequence of integers to decode
    char_list: char array
        A list of characters
    break_on_eos: bool
        Whether or not (default is set to true) to stop conversion when encountering a <eos> token

    Returns
    -------
    char_sequence: char array
        The decoded sequence
    """

    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]
            if c == SOS_STR_TOKEN and filter_sos:
                continue
            if c == EOS_STR_TOKEN and break_on_eos:
                break
            chars_sequence += c
        except Exception as e:
            chars_sequence += "Error char index"

    return chars_sequence
