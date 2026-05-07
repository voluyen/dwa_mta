import math
import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence
from utils import log_rank
import re
import spacy
from spacy.matcher import Matcher
from .span_utils import get_spans_offsets, compute_overall_span_loss

class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        VERB_PHRASE_PATTERN = [
            {"POS": "AUX", "OP": "*"},
            {"POS": "ADV", "OP": "*"},
            {"POS": "VERB", "OP": "+"},
            {"POS": "ADV", "OP": "*"},
        ]

        self.matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])

    def forward(
        self,
        distiller,
        input_data,
        output_data,
        logging_output,
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_model.eval()
        teacher_model_type = distiller.teacher_model_type

        prefix = f"teacher_{teacher_model_type}_"
        teacher_input = {k[len(prefix):]: v for k, v in input_data.items() if k.startswith(prefix)}
        teacher_label = {k[len(prefix):]: v for k, v in output_data.items() if k.startswith(prefix)}

        batch = {
            "input_batch": input_data,
            "label_batch": {"label": output_data["label"], "loss_denom": batch_denom},
            "teacher_input_batch": teacher_input,
            "teacher_label_batch": teacher_label,
        }

        batch_input = input_data

        self.distiller = distiller
        student_kwargs = {
            "input_ids": input_data["input_ids"],
            "attention_mask": input_data["attention_mask"],
        }
        if "position_ids" in input_data:
            student_kwargs["position_ids"] = input_data["position_ids"]
        outputs = model(**student_kwargs, output_hidden_states=True)
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"]
        )[0] / batch_denom
        log["nll_loss"] = ce_loss

        with torch.no_grad():
            teacher_kwargs = {
                "input_ids": teacher_input["input_ids"],
                "attention_mask": teacher_input["attention_mask"],
            }
            if "position_ids" in teacher_input:
                teacher_kwargs["position_ids"] = teacher_input["position_ids"]
            teacher_outputs = teacher_model(**teacher_kwargs, output_hidden_states=True)
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, batch, distiller, log
        )

        span_loss = 0.0
        if self.args.MTA_mode:
            tokenizer = distiller.student_tokenizer
            input_texts = tokenizer.batch_decode(batch_input['input_ids'], skip_special_tokens=False)
            offsets_mapping = tokenizer(input_texts, return_offsets_mapping=True, padding=True,
                                        add_special_tokens=False, return_tensors='pt')['offset_mapping']
            prases_offsets, spans_offsets, words_offsets = get_spans_offsets(input_texts, self.nlp, self.matcher)

            span_loss = compute_overall_span_loss(distiller.mta_projector_list, batch_input['attention_mask'],
                                                outputs.hidden_states, teacher_outputs.hidden_states,
                                                offsets_mapping, prases_offsets, spans_offsets, words_offsets, self.args)
            span_loss = self.args.w_span_loss * span_loss
            log["span_loss"] = span_loss

        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * (kd_loss + span_loss)
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        return loss, logging_output

    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["label_batch"]["label"]
        teacher_target = batch["teacher_label_batch"]["label"]
          
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_input = torch.where(pad_mask, batch["input_batch"]["input_ids"], torch.zeros_like(target))
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_input = torch.where(teacher_pad_mask, batch["teacher_input_batch"][f"input_ids"], torch.zeros_like(teacher_target))
        formal_teacher_target_for_index = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        t_preds = teacher_outputs.logits.argmax(-1)
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()
        tea_preds_embeds = tea_embed_tokens(t_preds).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_preds_embeds = tea_preds_embeds / tea_preds_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.query_projector(stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        if distiller.part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.s2t_projectors(hiddens).float()  # n x d x d x D -> n x D

        tea_v_hiddens = distiller.t2s_projectors(norm_teacher_hiddens + norm_tea_preds_embeds)  # m x D x D x d -> m x d

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        # student space
        t2s_weight = torch.softmax(align, -1).to(hiddens)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)  # n x m x m x d -> n x d
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]
  
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum()
        t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum()
        log["t2s_acc"] = t2s_acc
        log["t2s_acc_ratio"] = t2s_acc_ratio

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, target, reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / batch["label_batch"]["loss_denom"]

        log["t2s_ce_loss"] = t2s_ce_loss

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log

    def compute_on_policy_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["op_label_batch"]["label"]
        teacher_target = batch["op_teacher_label_batch"]["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, batch["op_input_batch"]["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target_for_index = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        t_preds = teacher_outputs.logits.argmax(-1)
        formal_teacher_input = torch.where(teacher_pad_mask, batch["op_teacher_input_batch"][f"input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()
        tea_preds_embeds = tea_embed_tokens(t_preds).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_preds_embeds = tea_preds_embeds / tea_preds_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.query_projector(stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        if distiller.part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.s2t_projectors(hiddens).float()  # n x d x d x D -> n x D

        tea_v_hiddens = distiller.t2s_projectors(norm_teacher_hiddens + norm_tea_preds_embeds)  # m x D x D x d -> m x d

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
       
        align = align + (1.0 - align_mask) * (-100000)

        # student space
        t2s_weight = torch.softmax(align, -1).to(hiddens)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)  # n x m x m x d -> n x d
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]
        
        # CMA for on-policy distillation is a little different, since there is no gold label for projected teacher distribution
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
        assert t_preds.shape == t2s_logits.shape[:2]
        t_preds_as_label = []
        align_ratio = []
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=t2s_logits.device))
                align_ratio.append(1.0)
                continue
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]
            
            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens,
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                try:
                    tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                    if len(tmp) == 1:
                        cur_t_preds_as_label_2[_s_idx] = tmp[0]
                    else:
                        cur_t_preds_as_label_2[_s_idx] = -100
                except:
                    cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:t2s_logits.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=t2s_logits.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, t2s_logits.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=t2s_logits.device).mean()

        # calculate t2s_ce_loss only on aligned tokens from both sequences
        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label, reduction="sum"
        )[0] / batch["op_label_batch"]["loss_denom"]

        t2s_acc_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        t2s_acc = (t2s_acc_mask * t_preds_as_label.ne(-100)).sum() / max(1e-3, t_preds_as_label.ne(-100).sum())
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_agreement"] = t2s_acc

        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / max(1e-3, t_preds_as_label.ne(-100).sum())

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)

        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["op_label_batch"]["loss_denom"]

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log


