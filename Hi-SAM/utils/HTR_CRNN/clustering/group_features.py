from HTR_CRNN.data.globalvalue.text_comon_values import CTC_PAD


def groupe_features_per_class(features, gt_seq_frames, index_class_to_filter):
    dict_feature_per_class = {}

    for features_one_item, y_one_item in zip(features, gt_seq_frames):

        if y_one_item is None:
            continue
        # y_one_item: tensor
        for f, y in zip(features_one_item, y_one_item):
            if y.item() in index_class_to_filter:
                continue
            else:
                if y.item() == CTC_PAD:
                    print("Error Pad class is used")
                    # print(one_id)
                else:
                    if y.item() in dict_feature_per_class:
                        dict_feature_per_class[y.item()].append(f)
                    else:
                        dict_feature_per_class[y.item()] = [f]

    return dict_feature_per_class
