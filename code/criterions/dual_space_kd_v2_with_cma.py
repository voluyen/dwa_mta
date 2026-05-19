import math
import torch
import torch.nn.functional as F
import spacy
from spacy.matcher import Matcher

from .various_divergence import VariousDivergence
from .span_utils import get_spans_offsets, compute_overall_span_loss


class DualSpaceKDV2WithCMA(VariousDivergence):
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
        if getattr(self.args, "MTA_mode", False):
            s_tokenizer = distiller.student_tokenizer
            t_tokenizer = distiller.teacher_tokenizers[teacher_model_type] \
                if hasattr(distiller, "teacher_tokenizers") else distiller.teacher_tokenizer

            device = input_data["input_ids"].device
            s_seq_len = input_data["input_ids"].size(1)
            t_seq_len = teacher_input["input_ids"].size(1)

            input_texts = s_tokenizer.batch_decode(input_data["input_ids"], skip_special_tokens=True)
            s_offsets_mapping = s_tokenizer(
                input_texts, return_offsets_mapping=True,
                padding="max_length", max_length=s_seq_len, truncation=True,
                add_special_tokens=False, return_tensors="pt",
            )["offset_mapping"].to(device)
            t_offsets_mapping = t_tokenizer(
                input_texts, return_offsets_mapping=True,
                padding="max_length", max_length=t_seq_len, truncation=True,
                add_special_tokens=False, return_tensors="pt",
            )["offset_mapping"].to(device)

            spans_offsets, words_offsets = get_spans_offsets(input_texts, self.nlp, self.matcher)

            span_loss = compute_overall_span_loss(
                distiller.mta_projector_list,
                input_data["attention_mask"], teacher_input["attention_mask"],
                outputs.logits, teacher_outputs.logits,
                outputs.hidden_states, teacher_outputs.hidden_states,
                s_offsets_mapping, t_offsets_mapping,
                spans_offsets, words_offsets, self.args,
            )
            span_loss = self.args.w_span_loss * span_loss
            log["span_loss"] = span_loss

        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * (kd_loss + span_loss)
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

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
            and hasattr(distiller.teacher_model.transformer, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_input = torch.where(pad_mask, batch["input_batch"]["input_ids"], torch.zeros_like(target))
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_input = torch.where(
            teacher_pad_mask, batch["teacher_input_batch"]["input_ids"], torch.zeros_like(teacher_target)
        )
        formal_teacher_target_for_index = torch.where(
            teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target)
        )
        t_preds = teacher_outputs.logits.argmax(-1)
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()
        tea_preds_embeds = tea_embed_tokens(t_preds).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_preds_embeds = tea_preds_embeds / tea_preds_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        part_teacher_head_pinv = getattr(distiller, "part_teacher_head_pinv", None)
        if part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.projectors["s2t"](hiddens).to(hiddens)

        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens + norm_tea_preds_embeds).to(hiddens)

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        # student space
        t2s_weight = torch.softmax(align, -1).to(hiddens)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )

        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum()
        t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum()
        log["t2s_acc"] = t2s_acc
        log["t2s_acc_ratio"] = t2s_acc_ratio

        loss_denom = batch["label_batch"]["loss_denom"]
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0] / loss_denom
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / loss_denom

        log["t2s_ce_loss"] = t2s_ce_loss

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / loss_denom

        if getattr(self.args, "only_stu_kd", False):
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif getattr(self.args, "only_tea_kd", False):
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
