def compute_loss_reg_k1(dict_features_per_class, clusters, loss_fct):
    loss_center_all_class = 0

    nb_class = 0
    for index_class, features in dict_features_per_class.items():
        nb_frames_used_class = 0
        loss_one_class = 0
        for one_feature in features:
            index_class_loss = index_class

            loss_reg = loss_fct(one_feature, clusters[index_class_loss])

            loss_one_class += loss_reg
            nb_frames_used_class += 1

        # Norm per class, not all item because classes are unbalanced
        if nb_frames_used_class != 0:
            loss_one_class /= nb_frames_used_class
            nb_class += 1

        loss_center_all_class += loss_one_class

    if nb_class != 0:
        loss_center_all_class /= nb_class

    return loss_center_all_class
