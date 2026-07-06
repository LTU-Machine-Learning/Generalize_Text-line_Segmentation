import time

import editdistance
import torch

from HTR_CRNN.clustering.center_loss import compute_loss_reg_k1
from HTR_CRNN.clustering.group_features import groupe_features_per_class
from HTR_CRNN.data.text.best_path import ctc_best_path_one


def train_crnn_one_epoch(training_loader,
                         optimizer,
                         model,
                         device,
                         ctc_loss,
                         char_list,
                         token_blank,
                         weight_loss_shortcut=0.1,
                         weight_loss_ok=1.0,
                         weight_loss_ko=1.0,
                         use_grad_clip=0):
    loss_main_ok_epoch = 0
    loss_main_ko_epoch = 0

    loss_shortcut_epoch = 0

    model.train()

    for index_batch, batch_data in enumerate(training_loader):
        optimizer.zero_grad()

        x = batch_data["imgs"].to(device)
        x_reduced_len = batch_data["w_reduce"]

        y_enc = batch_data["label_ind"].to(device)
        y_len_enc = batch_data["label_ind_length"]

        y_gt_txt = batch_data["label_str"]

        y_pred, _, _ = model(x)
        output, aux_output = y_pred

        output = torch.nn.functional.log_softmax(output, dim=-1)
        aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)

        # loss = ctc_loss(output.cpu(), y_enc, x_reduced_len, y_len_enc)
        # loss_main_epoch += loss.item()

        loss = 0
        # Separate loss for cer ok ko
        # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
        output_cer = output.transpose(0, 1)
        top_frames_main_pred = [torch.argmax(lp, dim=1)[:x_reduced_len[j]] for j, lp in enumerate(output_cer)]

        top_main_enc_cer = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(output_cer)]

        predictions_text_main_enc = [ctc_best_path_one(p, char_list, token_blank) if p is not None else "" for
                                     p in top_main_enc_cer]
        # Do not remove space padding
        # predictions_text_main_enc = [t.strip() for t in predictions_text_main_enc]  # Remove text padding
        cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

        loss_ok = 0
        loss_ko = 0
        for i_item in range(len(top_frames_main_pred)):
            if cers[i_item] == 0:
                log_prob = output[:, i_item, :].cpu()
                log_prob = log_prob.unsqueeze(1)

                targets = y_enc[i_item]
                targets = targets.unsqueeze(0)

                intput_length = [x_reduced_len[i_item]]
                target_length = [y_len_enc[i_item]]

                loss_one = ctc_loss(log_prob, targets, intput_length, target_length)
                loss_ok += loss_one
                loss_main_ok_epoch += loss_one.item()
            else:
                log_prob = output[:, i_item, :].cpu()
                log_prob = log_prob.unsqueeze(1)

                targets = y_enc[i_item]
                targets = targets.unsqueeze(0)

                intput_length = [x_reduced_len[i_item]]
                target_length = [y_len_enc[i_item]]

                loss_one = ctc_loss(log_prob, targets, intput_length, target_length)
                loss_ko += loss_one
                loss_main_ko_epoch += loss_one.item()

        loss += weight_loss_ok * loss_ok + weight_loss_ko * loss_ko

        loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc, x_reduced_len, y_len_enc)
        loss_shortcut_epoch += loss_shortcut.item()

        loss += loss_shortcut

        # if loss.item() > 80.0:
        #     print("loss > 80")
        #     print(batch_data["ids"])

        loss.backward()

        if use_grad_clip == 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

    # loss_main_epoch /= (index_batch + 1)
    loss_main_ok_epoch /= (index_batch + 1)
    loss_main_ko_epoch /= (index_batch + 1)
    loss_shortcut_epoch /= (index_batch + 1)

    losses = {
        "loss_main_ok_epoch": loss_main_ok_epoch,
        "loss_main_ko_epoch": loss_main_ko_epoch,
        "loss_shortcut": loss_shortcut_epoch,
    }

    return losses


def train_crnn_reg_one_epoch(training_loader,
                             optimizer,
                             model,
                             device,
                             ctc_loss,
                             conf_reg,
                             clusters,
                             char_list,
                             token_blank,
                             weight_loss_shortcut=0.1,
                             weight_loss_ok=1.0,
                             weight_loss_ko=1.0,
                             use_grad_clip=0):
    # loss_enc_main_epoch = 0
    loss_main_ok_epoch = 0
    loss_main_ko_epoch = 0

    loss_enc_shortcut_epoch = 0
    loss_reg_epoch = 0

    model.train()

    for index_batch, batch_data in enumerate(training_loader):
        optimizer.zero_grad()

        x = batch_data["imgs"].to(device)
        x_reduced_len = batch_data["w_reduce"]

        y_enc = batch_data["label_ind"].to(device)
        y_len_enc = batch_data["label_ind_length"]

        y_gt_txt = batch_data["label_str"]
        # y_gt_txt = [t.strip() for t in y_gt_txt] # Do not remove space padding

        y_pred, _, after_blstm = model(x)
        output, aux_output = y_pred

        # Recognition loss
        output = torch.nn.functional.log_softmax(output, dim=-1)

        # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
        output_cer = output.transpose(0, 1)

        # old top_main_enc
        top_frames_main_pred = [torch.argmax(lp, dim=1)[:x_reduced_len[j]] for j, lp in enumerate(output_cer)]

        top_main_enc_cer = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(output_cer)]

        predictions_text_main_enc = [ctc_best_path_one(p, char_list, token_blank) if p is not None else "" for
                                     p in top_main_enc_cer]
        # Do not remove space padding
        # predictions_text_main_enc = [t.strip() for t in predictions_text_main_enc]  # Remove text padding
        cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

        # Get sequence frames labels
        gt_frame_format_ok = []

        loss = 0
        loss_ok = 0
        loss_ko = 0
        for i_item in range(len(top_frames_main_pred)):
            if cers[i_item] == 0:
                gt_frame_format_ok.append(top_frames_main_pred[i_item])

                log_prob = output[:, i_item, :].cpu()
                log_prob = log_prob.unsqueeze(1)

                targets = y_enc[i_item]
                targets = targets.unsqueeze(0)

                intput_length = [x_reduced_len[i_item]]
                target_length = [y_len_enc[i_item]]

                loss_one = ctc_loss(log_prob, targets, intput_length, target_length)
                loss_ok += loss_one
                loss_main_ok_epoch += loss_one.item()
            else:
                gt_frame_format_ok.append(None)

                log_prob = output[:, i_item, :].cpu()
                log_prob = log_prob.unsqueeze(1)

                targets = y_enc[i_item]
                targets = targets.unsqueeze(0)

                intput_length = [x_reduced_len[i_item]]
                target_length = [y_len_enc[i_item]]

                loss_one = ctc_loss(log_prob, targets, intput_length, target_length)
                loss_ko += loss_one
                loss_main_ko_epoch += loss_one.item()

        # Recognition losses
        # loss = ctc_loss(output.cpu(), y_enc, x_reduced_len, y_len_enc)
        # loss_enc_main_epoch += loss.detach().item()

        loss += weight_loss_ok * loss_ok + weight_loss_ko * loss_ko

        aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
        loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc, x_reduced_len, y_len_enc)
        loss_enc_shortcut_epoch += loss_shortcut.detach().item()

        loss += loss_shortcut

        # Regularization Loss
        after_blstm = torch.permute(after_blstm, (1, 0, 2))
        # Apply activation function
        after_blstm = torch.sigmoid(after_blstm)

        dict_feature_per_class_ok = groupe_features_per_class(after_blstm, gt_frame_format_ok,
                                                              conf_reg["index_class_to_filter"])

        loss_reg_ok = compute_loss_reg_k1(dict_feature_per_class_ok, clusters, conf_reg["loss_reg"])

        loss_reg = conf_reg["weight_loss_regularization_ok"] * loss_reg_ok

        # Cas all predictions are classes filtered or cer != 0
        if not isinstance(loss_reg, float) and not isinstance(loss_reg, int):
            loss_reg = loss_reg.to(loss.device)
            loss += loss_reg
            loss_reg_epoch += loss_reg.detach().item()

        loss.backward()

        if use_grad_clip == 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

    # loss_enc_main_epoch /= (index_batch + 1)
    loss_main_ok_epoch /= (index_batch + 1)
    loss_main_ko_epoch /= (index_batch + 1)
    loss_enc_shortcut_epoch /= (index_batch + 1)
    loss_reg_epoch /= (index_batch + 1)

    losses = {
        # "loss_main": loss_enc_main_epoch,
        "loss_main_ok_epoch": loss_main_ok_epoch,
        "loss_main_ko_epoch": loss_main_ko_epoch,
        "loss_shortcut": loss_enc_shortcut_epoch,
        "loss_reg_epoch": loss_reg_epoch,
    }

    return losses
