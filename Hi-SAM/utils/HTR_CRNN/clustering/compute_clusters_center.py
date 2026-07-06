import editdistance
import torch

from HTR_CRNN.data.text.best_path import ctc_best_path_one
from HTR_CRNN.evaluate.metrics.evaluation_recognition import nb_chars_from_list
from HTR_CRNN.evaluate.metrics.metrics_counter import MetricLossCER


def compute_means_features(features_list):
    """one element of the list in the features related to this element"""
    if len(features_list) <= 0:
        print("compute_stat_features_v2: len(features_list) <= 0")
        print("return -1")
        return -1

    # N, nb features
    features_tensor = torch.stack(features_list)

    mean_tensor = torch.mean(features_tensor, 0)
    mean_tensor = mean_tensor.detach()

    return mean_tensor


def compute_cluster_center_crnn_k_1(data_loader,
                                    model,
                                    device,
                                    char_list,
                                    token_blank,
                                    index_class_to_filter):
    model.eval()

    nb_cer_ok = 0
    nb_cer_ko = 0

    # Size of features from CRNN
    prototypes_after = torch.zeros([len(char_list), 512]).to(device)

    dict_feature_per_class_after = {}

    metrics_main = MetricLossCER("Main CRNN compute center train with space padding")

    # Get prediction features
    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]
            y_gt_txt = batch_data["label_str"]

            # Remove text padding -> no need
            # y_gt_txt = [t.strip() for t in y_gt_txt]

            nb_item_batch = x.shape[0]

            y_pred, _, after_blstm = model(x)
            output, aux_output = y_pred

            after_blstm = torch.permute(after_blstm, (1, 0, 2))
            after_blstm = torch.sigmoid(after_blstm)

            output = torch.nn.functional.log_softmax(output, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output = output.transpose(0, 1)

            top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(output)]
            predictions_text_main_enc = [ctc_best_path_one(p, char_list, token_blank) if p is not None else "" for p in
                                         top_main_enc]
            # No need to remove text padding here

            cers_enc = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]
            metrics_main.add_cer(sum(cers_enc), nb_chars_from_list(y_gt_txt))

            # CER with space padding is less precise but do the trick
            for i in range(nb_item_batch):
                if cers_enc[i] != 0:
                    nb_cer_ko += 1
                else:
                    nb_cer_ok += 1
                    # Group features by character
                    for f, y in zip(after_blstm[i], top_main_enc[i]):
                        if y in index_class_to_filter:
                            continue
                        if y in dict_feature_per_class_after:
                            dict_feature_per_class_after[y].append(f)
                        else:
                            dict_feature_per_class_after[y] = [f]

    print("nb_cer_ko: " + str(nb_cer_ko))
    print("nb_cer_ok: " + str(nb_cer_ok))

    metrics_main.print_cer()

    # Compute stats features
    for key in dict_feature_per_class_after:
        if len(dict_feature_per_class_after[key]) > 0:
            mean_value = compute_means_features(dict_feature_per_class_after[key])

            prototypes_after[key] = mean_value

    return prototypes_after
