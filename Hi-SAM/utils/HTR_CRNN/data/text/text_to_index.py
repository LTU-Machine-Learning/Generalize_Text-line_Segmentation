from utils.HTR_CRNN.data.globalvalue.text_comon_values import SOS_STR_TOKEN, EOS_STR_TOKEN


def transcript_text_to_index(dictionary, str_to_transform, use_delimiter_tokens=False):
    """
    """

    labels = []

    if use_delimiter_tokens:
        labels.append(dictionary.get(SOS_STR_TOKEN))

    for c in str_to_transform:
        if c not in dictionary:
            print(str_to_transform)
            print("Text unknow char in dictionnary: " + str(c))
            print("Ignore")
            continue
            #return -1
        else:
            labels.append(dictionary.get(c))

    if use_delimiter_tokens:
        labels.append(dictionary.get(EOS_STR_TOKEN))

    return labels
