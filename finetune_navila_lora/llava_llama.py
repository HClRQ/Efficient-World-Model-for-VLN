#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IGNORE_INDEX
from llava.model.loss import soft_cross_entropy

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def _build_kv_prefix(self, episode_ids, steps, B, device, dtype):
        """state -> 每层KV前缀 (TSSM_KV_PREFIX=1)。懒建生成器; 冻结缓存取raw state。"""
        ad = getattr(self, "rssm", None) or getattr(self.get_llm(), "rssm", None)
        if ad is None:
            return None
        # 取当前步的 raw state [B, state_dim]
        if getattr(ad, "_online_state_buf", None) is not None:
            st = ad._online_state_buf.to(device).float()[:, -1, : ad.cache_state_dim]
        elif episode_ids is not None:
            old = ad.history_states
            ad.history_states = 1
            try:
                st = ad.lookup_state_sequence(list(episode_ids), list(steps), device)[:, -1]
            finally:
                ad.history_states = old
        else:
            return None
        # 生成器在 llava_arch 构造期建好(挂在 rssm 下), 这里只取用 —— 懒建会
        # 晚于优化器创建, 参数永不更新(静默失效)。
        gen = getattr(ad, "kv_prefix_gen", None)
        if gen is None:
            return None
        return gen(st.to(dtype))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
        rssm_actions: Optional[torch.LongTensor] = None,
        rssm_collisions: Optional[torch.FloatTensor] = None,
        tssm_episode_ids: Optional[List[str]] = None,
        tssm_steps: Optional[List[int]] = None,
        rssm_episode_ids: Optional[List[str]] = None,
        rssm_steps: Optional[List[int]] = None,
        tssm_imagined: Optional[List[bool]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()
        # B-course routing: adapter reads the flag off itself (no signature
        # churn through prepare_inputs); None keeps legacy behaviour.
        if getattr(self.get_llm(), "rssm", None) is not None or getattr(self, "rssm", None) is not None:
            _ad = getattr(self, "rssm", None)
            if _ad is not None:
                _ad._imagined_flags = tssm_imagined

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images,
                rssm_actions=rssm_actions, rssm_collisions=rssm_collisions,
                rssm_episode_ids=tssm_episode_ids or rssm_episode_ids,
                rssm_steps=tssm_steps or rssm_steps,
            )

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters
        # 读出检索: 打包会重排/合并行, _align_pos的行内位置失效 → 强制不打包
        if self.training and os.environ.get("TSSM_READNCE", "0") == "1":
            support_packing = False

        # ---- state条件化逐层KV前缀 (07-26 用户提议) ----
        # 记忆不占输入token位置, 而是作为每层的 past_key_values 注入。
        # 打包路径会丢弃 past_key_values 且逐样本前缀无法对应 → 门控开时强制不打包。
        _kvp = os.environ.get("TSSM_KV_PREFIX", "0") == "1"
        _prefix_kv = None
        if _kvp:
            support_packing = False
            _prefix_kv = self._build_kv_prefix(
                tssm_episode_ids or rssm_episode_ids, tssm_steps or rssm_steps,
                inputs_embeds.shape[0], inputs_embeds.device, inputs_embeds.dtype)

        if self.training and support_packing and not dpo_forward:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids
            if _prefix_kv is not None:
                # 前缀作为 past_key_values 注入; attention_mask 左侧补 m 个1,
                # 使所有真实token都能注意到前缀。position_ids 保持不变 —— 前缀是
                # 位置无关的记忆(不经RoPE), 真实token位置不因past长度而偏移。
                past_key_values = _prefix_kv
                _m = _prefix_kv[0][0].shape[2]
                _ones = new_attention_mask.new_ones(new_attention_mask.shape[0], _m)
                new_attention_mask = torch.cat([_ones, new_attention_mask], dim=1)

        # 次态对齐: 懒注册钩子截取最后层隐状态(仅训练+门控; 避免
        # output_hidden_states=True 的全层显存开销)
        if (self.training and (os.environ.get("TSSM_ALIGN_AUX", "0") == "1"
                               or os.environ.get("TSSM_READNCE", "0") == "1")
                and not hasattr(self, "_align_hook")):
            def _align_cap(_m, _i, out):
                self._align_h = out
            self._align_hook = self.llm.model.norm.register_forward_hook(_align_cap)
            # 08-11 深监督移植(9B配方): retro/dyn 查询各自锚中间层(如11/23),
            # 环境变量不设则回落末层老口径。decoder layer 输出是tuple, 取[0]。
            def _mk_cap(attr):
                def _cap(_m, _i, out):
                    setattr(self, attr, out[0] if isinstance(out, tuple) else out)
                return _cap
            _rl = os.environ.get("TSSM_READNCE_ANCHOR_LAYER", "")
            _dl = os.environ.get("TSSM_READNCE_DYN_ANCHOR_LAYER", "")
            if _rl:
                self._align_hook_r = self.llm.model.layers[int(_rl)].register_forward_hook(
                    _mk_cap("_align_h_retro"))
            if _dl:
                self._align_hook_d = self.llm.model.layers[int(_dl)].register_forward_hook(
                    _mk_cap("_align_h_dyn"))

        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # print(outputs)

        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        # Loss rescale for SP & DP loss match (LM loss only)
        loss_weight = calculate_loss_weight(new_labels)
        if outputs.loss.requires_grad:
            # LM loss is active: scale it, then add temporal-state loss unscaled
            outputs.loss = outputs.loss * loss_weight
        elif not self.training:
            # EVALUATION: loss legitimately carries no grad — keep the REAL value.
            # (Previously hard-zeroed, which nuked eval_loss and probe losses.)
            outputs.loss = outputs.loss * loss_weight
        else:
            # TRAINING with frozen LM (no active label tokens): loss_weight would
            # be 0/nan, so skip multiplication; temporal-state loss is set below
            outputs.loss = torch.zeros(1, device=outputs.loss.device, dtype=outputs.loss.dtype)

        # ---- 次态对齐 (07-20, TSSM_ALIGN_AUX门控): LLM在记忆位置的隐状态
        # 经小头预测 冻结投影的s_{t+1} token; 教LLM预判记忆演化(WoG式)。----
        if (self.training and os.environ.get("TSSM_ALIGN_AUX", "0") == "1"
                and self.rssm is not None):
            h = getattr(self, "_align_h", None)
            pos = getattr(self, "_align_pos", None)
            ad = self.rssm
            if h is not None and pos and hasattr(ad, "x_aux_align_head"):
                tgt, ok = ad.align_targets(h.device)
                pad_left = getattr(self.llm.config, "tokenizer_padding_side",
                                   "right") == "left"
                rows = []
                for i, (s0, n) in pos.items():
                    if i >= h.shape[0] or not bool(ok[i]):
                        continue
                    shift = (h.shape[1] - int(attention_mask[i].sum())) if (
                        pad_left and attention_mask is not None) else 0
                    a, b = s0 + shift, s0 + shift + n
                    if b > h.shape[1]:
                        continue                     # 截断行, 放弃
                    pred = ad.x_aux_align_head(h[i, a:b, :]).float()
                    rows.append(1 - torch.nn.functional.cosine_similarity(
                        pred, tgt[i], dim=-1).mean())
                if rows:
                    align_loss = ad.align_w * torch.stack(rows).mean()
                    base = getattr(self, "_rssm_loss", None)
                    self._rssm_loss = (align_loss if base is None
                                       else base.to(align_loss.device) + align_loss)

        # ---- 读出检索 (07-29, TSSM_READNCE门控): 逼LLM把state里的信息读出来。
        # retro头: 记忆位置隐状态 → 在冻结ncev5钥匙空间检索t-k帧(同集难负样本);
        # dyn头:  隐状态+动作窗嵌入 → 检索t+k帧(FutureNav式动作条件动力学)。
        # 直接加在outputs.loss上(不吃rssm_weight折扣), 权重由环境变量独立控制。
        if (self.training and os.environ.get("TSSM_READNCE", "0") == "1"
                and self.rssm is not None):
            h = getattr(self, "_align_h", None)
            pos = getattr(self, "_align_pos", None)
            ad = self.rssm
            eps_l = tssm_episode_ids or rssm_episode_ids
            steps_l = tssm_steps or rssm_steps
            if hasattr(ad, "readnce_q_head") and outputs.loss.requires_grad:
                # 集体操作对齐(冒烟9009705尸检): 各rank该步有效样本数可以是0,
                # 某头不进计算图 → ZeRO梯度归约桶两边不齐 → NCCL REDUCE挂死。
                # 恒加 0*Σp 保证全部readnce参数每步在图里(梯度为零但参与归约)。
                _zero = None
                for _mod in (ad.readnce_q_head, ad.readnce_dyn_head,
                             ad.readnce_lag_emb, ad.readnce_dyn_lag_emb,
                             ad.readnce_act_emb):
                    for _p in _mod.parameters():
                        _s = _p.sum()
                        _zero = _s if _zero is None else _zero + _s
                outputs.loss = outputs.loss + 0.0 * _zero.to(outputs.loss.dtype)
            if (h is not None and pos and eps_l is not None and steps_l is not None
                    and hasattr(ad, "readnce_q_head")):
                pad_left = getattr(self.llm.config, "tokenizer_padding_side",
                                   "right") == "left"
                # 中间层锚(未设环境变量时两者都回落末层h, 行为与旧口径逐位一致)
                h_retro = getattr(self, "_align_h_retro", None)
                h_dyn = getattr(self, "_align_h_dyn", None)
                h_retro = h if h_retro is None else h_retro
                h_dyn = h if h_dyn is None else h_dyn
                r_terms, d_terms, r_hit, d_hit = [], [], [], []
                # 锚点开关(07-30): state=记忆token位置(证"信息活到最后层");
                # answer=出动作的位置(焊死"最后一跳": 答案隐状态里必须查得出历史)。
                _anchor = os.environ.get("TSSM_READNCE_ANCHOR", "state")
                for i, (s0, n) in pos.items():
                    if i >= h.shape[0] or i >= len(eps_l):
                        continue
                    K = ad._readnce_keys.get(str(eps_l[i]))
                    if K is None:
                        continue
                    t = int(steps_l[i]); T = K.shape[0]
                    if t >= T:
                        continue
                    if _anchor == "answer" and new_labels is not None:
                        # 预测第j个答案token的隐状态在位置j-1: 取全部答案预测位
                        _lab = new_labels[i]
                        _ans = (_lab != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                        if _ans.numel() == 0:
                            continue
                        # 查询锚点 (07-31 GPT审计): 第2+答案位看得见真实动作前缀
                        # (teacher forcing泄漏) → 默认first只取首答案预测位(零泄漏)
                        if os.environ.get("TSSM_READNCE_QANCHOR", "first") == "first":
                            _pred = (_ans[:1] - 1).clamp(min=0)
                        else:
                            _pred = (_ans - 1).clamp(min=0)
                        _pred = _pred[_pred < h.shape[1]]
                        if _pred.numel() == 0:
                            continue
                        if os.environ.get("TSSM_READNCE_POOL", "mean") == "none":
                            # 不池化(07-30): 每个答案位各自为查询, CE按位置取平均
                            # — 逼每一个出词位置都握着可检索历史(约束更严)
                            pooled_r = h_retro[i, _pred, :]           # [P, d]
                            pooled_d = h_dyn[i, _pred, :]
                        else:
                            pooled_r = h_retro[i, _pred, :].mean(0)   # [d]
                            pooled_d = h_dyn[i, _pred, :].mean(0)
                    else:
                        shift = (h.shape[1] - int(attention_mask[i].sum())) if (
                            pad_left and attention_mask is not None) else 0
                        a, b = s0 + shift, s0 + shift + n
                        if b > h.shape[1]:
                            continue                 # 截断行, 放弃
                        pooled_r = h_retro[i, a:b, :].mean(0)
                        pooled_d = h_dyn[i, a:b, :].mean(0)
                    Kd = K.to(device=h.device, dtype=torch.float32)

                    def _rn_ce(head, extras, tgt, pooled):
                        # pooled 一维=池化查询; 二维[P,d]=逐位置查询(POOL=none),
                        # CE按位置平均, 命中按多数位置计
                        if pooled.dim() == 1:
                            q = head(torch.cat([pooled] + [e.to(pooled.dtype) for e in extras]))
                            q = torch.nn.functional.normalize(q.float(), dim=-1)
                            lg = (Kd @ q) / ad.readnce_tau
                            ce = torch.nn.functional.cross_entropy(
                                lg.unsqueeze(0), torch.tensor([tgt], device=h.device))
                            hit = float(int(lg.argmax()) == tgt)
                        else:
                            P = pooled.shape[0]
                            ex = [e.to(pooled.dtype).unsqueeze(0).expand(P, -1) for e in extras]
                            q = head(torch.cat([pooled] + ex, dim=-1))
                            q = torch.nn.functional.normalize(q.float(), dim=-1)
                            lg = (q @ Kd.T) / ad.readnce_tau              # [P,T]
                            ce = torch.nn.functional.cross_entropy(
                                lg, torch.full((P,), tgt, device=h.device, dtype=torch.long))
                            hit = float((lg.argmax(-1) == tgt).float().mean() >= 0.5)
                        return ce, hit

                    for li, k in enumerate(ad.readnce_retro_lags):
                        if t - k < 0:
                            continue
                        le = ad.readnce_lag_emb(
                            torch.tensor(li, device=h.device))
                        ce, hit = _rn_ce(ad.readnce_q_head, [le], t - k, pooled_r)
                        r_terms.append(ce)
                        r_hit.append(hit)
                    acts = ad._act_seqs.get(str(eps_l[i]))
                    if acts is None:
                        continue
                    for li, k in enumerate(ad.readnce_dyn_lags):
                        # incoming对齐(biraw教训): acts[j]=进入第j帧的动作,
                        # t→t+k 的动作窗 = acts[t+1 : t+k+1]
                        if t + k >= T or t + k + 1 > len(acts):
                            continue
                        win = acts[t + 1: t + k + 1].long().to(h.device)
                        if win.numel() == 0:
                            continue
                        if getattr(ad, "readnce_act_enc", "mean") == "concat":
                            # 保序: 固定k_max槽位, 不足补pad0 — mean会把动作顺序抹掉
                            _slots = torch.zeros(ad.readnce_act_kmax,
                                                 dtype=torch.long, device=h.device)
                            _slots[: win.numel()] = win
                            ae = ad.readnce_act_emb(_slots).flatten()
                        else:
                            ae = ad.readnce_act_emb(win).mean(0)
                        dle = ad.readnce_dyn_lag_emb(
                            torch.tensor(li, device=h.device))
                        ce, hit = _rn_ce(ad.readnce_dyn_head, [dle, ae], t + k, pooled_d)
                        d_terms.append(ce)
                        d_hit.append(hit)
                rn_add = None
                if r_terms:
                    rn_add = ad.readnce_w * torch.stack(r_terms).mean()
                if d_terms:
                    d_loss = ad.readnce_dyn_w * torch.stack(d_terms).mean()
                    rn_add = d_loss if rn_add is None else rn_add + d_loss
                if rn_add is not None:
                    outputs.loss = outputs.loss + rn_add.to(outputs.loss.dtype)
                    self._readnce_n = getattr(self, "_readnce_n", 0) + 1
                    if self._readnce_n % 50 == 1:
                        _ra = sum(r_hit) / max(1, len(r_hit))
                        _da = sum(d_hit) / max(1, len(d_hit))
                        print(f"[readnce] n={self._readnce_n} "
                              f"retro_ce={float(torch.stack(r_terms).mean()) if r_terms else -1:.3f} "
                              f"acc={_ra:.2f}({len(r_hit)}) "
                              f"dyn_ce={float(torch.stack(d_terms).mean()) if d_terms else -1:.3f} "
                              f"acc={_da:.2f}({len(d_hit)})", flush=True)

        # v12 联合训练 (07-27): TSSM 在线前向的自监督锚损失(recon/KL/retro),
        # anchor_w 已在适配器内乘过, 不再吃 rssm_weight(0.1) 折扣 —— 它是防遗忘主锚。
        if self.training and self.rssm is not None:
            _jl = getattr(self.rssm, "_tssm_joint_loss", None)
            if _jl is not None:
                outputs.loss = outputs.loss + _jl
                self.rssm._tssm_joint_loss = None

        # Add temporal-state loss if enabled (after loss_weight to avoid being zeroed out)
        if self.training and self.rssm is not None and hasattr(self, "_rssm_loss"):
            rssm_weight = getattr(self.config, "tssm_loss_weight", getattr(self.config, "rssm_loss_weight", 0.1))
            outputs.loss = outputs.loss + rssm_weight * self._rssm_loss

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
