import logging
import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger(__name__)

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import BertConfig, BertModel
from transformers import RobertaModel, RobertaTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector, CrossAttention
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT
from spamo.callbacks import dump_test_outputs


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        fusion_dropout: float = 0.0,
        fusion_lr: Optional[float] = None,
        new_stream_lr: Optional[float] = None,
        queue_size: int = 0,
        keypoint_dim: int = 0,
        aux_input_size: int = 0,
        motion_input_size: int = 0,
        use_frozen_text_encoder: bool = False,
        lr_patience: int = 5,
        lr_scheduler_mode: str = 'max',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fusion_dropout = fusion_dropout
        self.fusion_lr = fusion_lr
        self.new_stream_lr = new_stream_lr
        self.queue_size = queue_size
        self.keypoint_dim = keypoint_dim
        self.aux_input_size = aux_input_size
        self.motion_input_size = motion_input_size if motion_input_size > 0 else input_size
        self.use_frozen_text_encoder = use_frozen_text_encoder
        self.lr_patience = lr_patience
        self.lr_scheduler_mode = lr_scheduler_mode

        self.prepare_models(model_name)

        # Aux stream projector: e.g. hand-crop ViT [T, aux_input_size] → [T, inter_hidden]
        if self.aux_input_size > 0:
            self.aux_proj = build_vision_projector('linear', aux_input_size, self.inter_hidden)

        # Keypoint projector: MediaPipe [T, keypoint_dim] → [T, inter_hidden]
        if self.keypoint_dim > 0:
            self.kp_proj = nn.Sequential(
                nn.Linear(keypoint_dim, inter_hidden),
                nn.LayerNorm(inter_hidden),
                nn.GELU(),
                nn.Linear(inter_hidden, inter_hidden),
            )
        # Cross-attention enrichment for quad streams.
        # aux/kp enrich V tokens in-place (no prefix length change) — dual ckpt compatible.
        # Zero-init proj → identity forward on step 0 → init BLEU = dual baseline.
        if self.aux_input_size > 0 or self.keypoint_dim > 0:
            self.aux_xattn = CrossAttention(self.inter_hidden, num_heads=8, qkv_bias=True)
            nn.init.zeros_(self.aux_xattn.proj.weight)
            nn.init.zeros_(self.aux_xattn.proj.bias)

        # Memory bank for contrastive learning (queue of past visual embeddings)
        if self.queue_size > 0:
            d = self.inter_hidden if self.use_frozen_text_encoder else self.t5_model.config.d_model
            self.register_buffer('_vis_queue', F.normalize(torch.randn(queue_size, d), dim=-1))
            self._queue_ptr = 0

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        if self.use_frozen_text_encoder:
            self.frozen_text_encoder = RobertaModel.from_pretrained('roberta-large')
            self.frozen_text_encoder.eval()
            for p in self.frozen_text_encoder.parameters():
                p.requires_grad = False
            # projection RobBERTa hidden 1024 to our inter hidden (768)
            self.text_align_proj = nn.Linear(1024, self.inter_hidden)
            self.image_align_proj = nn.Linear(self.t5_model.config.d_model, self.inter_hidden)
        self.set_container()
        self._reset_train_speaker_losses()

    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        ckpt_sd = checkpoint['state_dict']
        model_sd = self.state_dict()
        compatible = {
            k: v for k, v in ckpt_sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        self.load_state_dict(compatible, strict=False)
        skipped = [k for k in ckpt_sd if k not in compatible]
        log.info(f'Checkpoint loaded from {checkpoint_path}: {len(compatible)}/{len(ckpt_sd)} params.')
        if skipped:
            log.warning(f'Skipped {len(skipped)} incompatible keys: {skipped}')

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []
        self.speaker_ids_val = []
        self.visual_embeds_val = []
        self.diag_batch: dict | None = None  # first val batch snapshot for XAI forward pass

    def _reset_train_speaker_losses(self) -> None:
        from collections import defaultdict
        self.train_speaker_losses: dict = defaultdict(list)

    def prepare_models(self, t5_model: str) -> None:
        """
        Prepare the textual and visual models.
        
        Args:
            t5_model: Name or path of the T5 model to use
        """
        
        # Load the textual model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', self.motion_input_size, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)

        # Load the temporal encoder
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # if self.cross_modal_align:
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.
        
        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt with language information
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get lengths for visual and prompt sequences
        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths
        
        # Convert tokens to embeddings
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        # Concatenate visual and text embeddings
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Prepare target labels (replace pad tokens with -100)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Project aux (hand_ViT) and keypoint (MediaPipe) streams to inter_hidden space.
        # Both remain None when the stream is disabled (dual mode).
        aux_outputs = kp_outputs = None
        if self.aux_input_size > 0 and samples.get('aux_values'):
            aux_padded = pad_sequence(samples['aux_values'], batch_first=True).to(self.device)
            aux_outputs = self.aux_proj(aux_padded)  # [B, T_A, inter_hidden]

        if self.keypoint_dim > 0 and samples.get('keypoint_values'):
            kp_padded = pad_sequence(samples['keypoint_values'], batch_first=True).to(self.device)
            kp_outputs = self.kp_proj(kp_padded)  # [B, T_K, inter_hidden]

        # Enrich spatial tokens with aux/kp context via cross-attention.
        # Prefix length stays [V+M] — same as dual — so T5 sees its trained prefix shape.
        if hasattr(self, 'aux_xattn'):
            context_parts = []
            if aux_outputs is not None:
                context_parts.append(aux_outputs)
            if kp_outputs is not None:
                context_parts.append(kp_outputs)
            if context_parts:
                context = torch.cat(context_parts, dim=1)  # [B, T_A+T_K, inter_hidden]
                spatial_outputs = spatial_outputs + self.aux_xattn(spatial_outputs, context)

        # Combine spatial (optionally enriched) + motion — dual layout preserved
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length

            # Concatenate spatial + spatiotemporal (aux/kp already fused into spatial above)
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                parts = [valid_spatial_output, valid_spatiotemporal_output]

                concat_sample = torch.cat(parts, dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            # Apply temporal encoder
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            # Use single feature type
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, ids = [], [], []
        texts, speaker_ids = [], []
        num_frames, glor_lengths, langs = [], [], []
        keypoint_values = []
        aux_values, aux_lengths = [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                # Calculate number of frames after sampling
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                # Collect metadata
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                langs.append(sample['lang'])
                speaker_ids.append(sample.get('speaker_id', 'unknown'))
                
                if self.use_in_context:
                    # Use pl_text if present (PJM EN target — avoids degenerate en==text identity);
                    # otherwise fall back to en_text (Phoenix DE→DE — paper original).
                    first_lang = sample.get('pl_text') or sample.get('en_text', sample['text'])
                    _ex_lang_trans = [
                        f"{first_lang}={sample['text']}",
                        f"{sample['fr_text']}={sample['text']}",
                        f"{sample['es_text']}={sample['text']}"
                    ]
                    _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                    ex_lang_translations.append(' '.join(_ex_lang_trans))
                
                # Handle too long sequences with random cropping
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                # Store processed visual features
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                # Process glor values if available
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))

                # Collect keypoints if available
                kp = sample.get('keypoint_value')
                if kp is not None:
                    keypoint_values.append(kp[::self.frame_sample_rate])

                # Collect aux stream if available
                aux = sample.get('aux_value')
                if aux is not None:
                    aux_sampled = aux[::self.frame_sample_rate]
                    aux_values.append(aux_sampled)
                    aux_lengths.append(len(aux_sampled))
        
        ex_lang_translations = derangement(ex_lang_translations)
        
        # Return structured dictionary
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'ids': ids,
            'text': texts,
            'speaker_ids': speaker_ids,
            'ex_lang_trans': ex_lang_translations,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
            'keypoint_values': keypoint_values,
            'aux_values': aux_values,
            'aux_lengths': aux_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss.
        
        Args:
            visual_outputs: Visual features
            visual_masks: Mask for visual features
            samples: Input samples
            
        Returns:
            Contrastive loss
        """
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Get text embeddings via T5 encoder (frozen, no grad) with masked mean pooling
        mask = output_tokens.attention_mask.unsqueeze(-1).float()
        with torch.no_grad():
            if self.use_frozen_text_encoder:
                roberta_out = self.frozen_text_encoder(
                    input_ids=output_tokens.input_ids,
                    attention_mask=output_tokens.attention_mask
                    ).last_hidden_state.float()
                pooled = (roberta_out * mask).sum(1) / mask.sum(1).clamp(min=1)
                text_embeds = self.text_align_proj(pooled)  
            else:
                tok_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids).float()
                text_embeds = (tok_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Mean pooling for visual embeddings; kp already fused into visual_outputs via aux_xattn
        image_embeds = visual_outputs.mean(1)  # [B, d_model]

        if self.use_frozen_text_encoder:
            image_embeds = self.image_align_proj(image_embeds)
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Calculate cosine similarities with temperature scaling (clamped like CLIP, max=100)
        logit_scale = self.logit_scale.clamp(max=4.6052).exp()

        if self.queue_size > 0:
            # Asymmetric loss: text queries against current batch + memory bank
            all_image = torch.cat([image_embeds, self._vis_queue.clone().detach()], dim=0)  # [B+Q, D]
            logits = torch.matmul(text_embeds, all_image.t()) * logit_scale  # [B, B+Q]
            labels = torch.arange(len(text_embeds), device=self.device)
            loss = F.cross_entropy(logits, labels)
            # Update circular queue with current batch
            with torch.no_grad():
                B = image_embeds.shape[0]
                ptr = self._queue_ptr
                slots = min(B, self.queue_size - ptr)
                self._vis_queue[ptr:ptr + slots] = image_embeds.detach()[:slots]
                if slots < B:
                    self._vis_queue[:B - slots] = image_embeds.detach()[slots:]
                self._queue_ptr = (ptr + B) % self.queue_size
        else:
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            logits_per_image = logits_per_text.T
            loss = clip_loss(logits_per_text)
        
        return loss

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps.
        
        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (loss, log_dict)
        """
        # Prepare visual inputs and project to match text embedding dimensions
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        if self.fusion_dropout > 0.0:
            visual_outputs = F.dropout(visual_outputs, p=self.fusion_dropout, training=self.training)

        # Initialize logging dictionary
        log_dict = {}

        # Diagnostic: diversity of visual features in this batch (signal richness proxy)
        log_dict[f"{split}/visual_feat_std"] = visual_outputs.mean(1).std(0).mean().item()
        # Diagnostic: contrastive temperature (should grow stably, not explode or collapse)
        log_dict[f"{split}/logit_scale"] = self.logit_scale.exp().item()
        
        # STEP 1: Determine training mode and prepare inputs accordingly
        if self.cross_modal_align:
            # For pure contrastive learning or warm-up phase
            if self.warm_up_steps is None and not self.combined_loss:
                # Pure contrastive learning mode
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                # Warm-up phase with contrastive learning
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            else:
                # Combined loss mode (regular training + contrastive)
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                
                # Forward pass through T5 model
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                t5_loss = outputs.loss
                log_dict[f"{split}/loss"] = t5_loss
                
                # Add contrastive component if using combined loss
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        else:
            # Standard training without contrastive learning
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Forward pass through T5 model
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        # STEP 2: Handle evaluation phase (validation/testing)
        if split != "train":
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]
            self.references.extend(reference_strings)
            self.speaker_ids_val.extend(inputs['speaker_ids'])
            self.visual_embeds_val.append(visual_outputs.mean(1).detach().float())

            if not (not self.combined_loss and self.warm_up_steps is None):
                # Full pipeline: generate translations
                input_embeds, input_masks, _, _ = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                generated = self.t5_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    num_beams=self.beam_size,
                    max_length=self.max_txt_len,
                    top_p=0.9,
                    do_sample=True,
                )
                generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
                generated_strings = [gen.lower() for gen in generated_strings]
                self.generated.extend(generated_strings)

                if batch_idx == 0 and self.diag_batch is None:
                    n_diag = min(8, input_embeds.shape[0])
                    self.diag_batch = {
                        'input_embeds': input_embeds[:n_diag].detach(),
                        'input_masks': input_masks[:n_diag].detach(),
                        'output_ids': output_tokens.input_ids[:n_diag].detach(),
                        'output_mask': output_tokens.attention_mask[:n_diag].detach(),
                        'ref_strings': reference_strings[:n_diag],
                        'gen_strings': generated_strings[:n_diag],
                    }
            
            # Calculate evaluation metrics
            # eval_res = evaluate_results(
            #     predictions=generated_strings,
            #     references=reference_strings,
            #     split=split,
            #     tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',
            #     device=self.device
            # )
            
            # Add evaluation results to logging
            # log_dict.update(eval_res)

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        loss, log_dict = self.shared_step(inputs, "train", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True, on_step=True, on_epoch=True)
        loss_val = loss.item()
        for sid in inputs['speaker_ids']:
            self.train_speaker_losses[sid].append(loss_val)
        return loss

    def on_train_epoch_end(self) -> None:
        if not self.train_speaker_losses:
            return
        import statistics
        rows = []
        for sid in sorted(self.train_speaker_losses.keys()):
            vals = self.train_speaker_losses[sid]
            rows.append([sid, len(vals), round(sum(vals) / len(vals), 4)])
        loss_means = [r[2] for r in rows]
        self.log("train/speaker_loss_std", statistics.stdev(loss_means) if len(loss_means) > 1 else 0.0, sync_dist=True)
        if self.logger and hasattr(self.logger, 'experiment'):
            import wandb
            self.logger.experiment.log({
                "train/per_speaker_loss": wandb.Table(
                    columns=["speaker_id", "n_batches", "mean_loss"],
                    data=rows,
                ),
                "trainer/global_step": self.global_step,
            })
        self._reset_train_speaker_losses()

    def _aggregate_cross_attention(self, cross_attentions) -> 'torch.Tensor':
        """
        Combines: max over heads (keeps sharpest attention per position),
        then mean over top-6 decoder layers (most semantic signal).
        Returns float32 [dec_seq_len, enc_seq_len].
        """
        top_layers = cross_attentions[-6:]                          # (6,) × [1, heads, dec, enc]
        stacked = torch.stack([l.squeeze(0) for l in top_layers])  # [6, heads, dec, enc]
        sharpened = stacked.max(dim=1).values                       # [6, dec, enc] — max over heads
        return sharpened.mean(dim=0).float()                        # [dec, enc]

    def _log_xai_diagnostics(self, split: str, cos_sims: 'torch.Tensor | None') -> None:
        pass

    def _compute_retrieval_metrics(self, split: str) -> None:
        """Alignment gap + R@1/5/10 using T5 encoder as text tower."""
        if not self.visual_embeds_val or not self.references:
            return
        vis_embeds = torch.cat(self.visual_embeds_val, dim=0).to(self.device)  # [N, hidden]
        # Build text embeddings via T5 encoder (masked mean pool, chunked)
        chunk_size = 128
        txt_list = []
        with torch.no_grad():
            for i in range(0, len(self.references), chunk_size):
                chunk = self.references[i:i + chunk_size]
                tokens = self.t5_tokenizer(
                    chunk, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt"
                ).to(self.device)
                enc = self.t5_model.encoder(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                ).last_hidden_state.float()
                mask = tokens.attention_mask.unsqueeze(-1).float()
                txt_list.append((enc * mask).sum(1) / mask.sum(1).clamp(min=1))
        txt_embeds = torch.cat(txt_list, dim=0)  # [N, hidden]

        vis_norm = F.normalize(vis_embeds.float(), dim=-1)
        txt_norm = F.normalize(txt_embeds, dim=-1)
        sim = vis_norm @ txt_norm.T  # [N, N]

        N = sim.shape[0]
        diag = sim.diagonal()
        off_diag_sum = sim.sum() - diag.sum()
        alignment_gap = diag.mean().item() - (off_diag_sum / (N * (N - 1))).item()
        self.log(f"{split}/alignment_gap", alignment_gap, sync_dist=True)

        ranks = (sim > diag.unsqueeze(1)).sum(dim=1)  # how many scores beat the correct one
        for k in [1, 5, 10]:
            recall_at_k = (ranks < k).float().mean().item()
            self.log(f"{split}/R@{k}", recall_at_k, sync_dist=True)
        self.log(f"{split}/median_rank", ranks.float().median().item() + 1, sync_dist=True)

    def _log_per_speaker_metrics(self, split: str, speaker_ids: list, preds: list, refs: list) -> None:
        from collections import defaultdict
        import statistics
        from sacrebleu.metrics import BLEU as SacreBLEU, CHRF as SacreCHRF
        spk_preds: dict = defaultdict(list)
        spk_refs: dict = defaultdict(list)
        for sid, pred, ref in zip(speaker_ids, preds, refs):
            spk_preds[sid].append(pred)
            spk_refs[sid].append(ref)
        rows, bleu4s = [], []
        for sid in sorted(spk_preds.keys()):
            b4 = SacreBLEU(max_ngram_order=4, tokenize='13a').corpus_score(spk_preds[sid], [spk_refs[sid]]).score
            chrf = SacreCHRF().corpus_score(spk_preds[sid], [spk_refs[sid]]).score
            rows.append([sid, len(spk_preds[sid]), round(b4, 3), round(chrf, 3)])
            bleu4s.append(b4)
        if len(bleu4s) > 1:
            self.log(f"{split}/speaker_bleu4_std", statistics.stdev(bleu4s), sync_dist=True)
        if self.logger and hasattr(self.logger, 'experiment'):
            import wandb
            self.logger.experiment.log({
                f"{split}/per_speaker_metrics": wandb.Table(
                    columns=["speaker_id", "n_samples", "bleu4", "chrf"],
                    data=rows,
                ),
                "trainer/global_step": self.global_step,
            })

    def on_validation_epoch_end(self) -> None:
        is_contrastive_only = not self.combined_loss and self.warm_up_steps is None

        if is_contrastive_only:
            # Fast contrastive-only mode: skip beam search metrics, compute retrieval metrics only
            self._compute_retrieval_metrics('val')
            self.set_container()
            return

        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)

        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            device=self.device
        )
        self.log_dict(eval_res, sync_dist=True)

        # Degenerate output ratio (empty or ≤2 words)
        degenerate = sum(1 for g in self.generated if len(g.split()) <= 2)
        self.log("val/degenerate_ratio", degenerate / max(len(self.generated), 1), sync_dist=True)

        cos_sims = None
        # Cosine similarity: mean-pooled visual embeddings vs T5 encoder output
        if self.visual_embeds_val:
            vis_embeds = torch.cat(self.visual_embeds_val, dim=0)  # [N, hidden]
            chunk_size = 128
            text_embeds_list = []
            with torch.no_grad():
                for i in range(0, len(self.references), chunk_size):
                    chunk = self.references[i:i + chunk_size]
                    tokens = self.t5_tokenizer(
                        chunk, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt"
                    ).to(self.device)
                    enc_out = self.t5_model.encoder(
                        input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                    ).last_hidden_state  # [B, seq, hidden]
                    # Masked mean pooling — exclude PAD tokens
                    mask = tokens.attention_mask.unsqueeze(-1).float()
                    pooled = (enc_out.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                    text_embeds_list.append(pooled)
            text_embeds = torch.cat(text_embeds_list, dim=0)  # [N, hidden]
            vis_norm = F.normalize(vis_embeds, dim=-1)
            txt_norm = F.normalize(text_embeds, dim=-1)
            cos_sims = (vis_norm * txt_norm).sum(-1)  # [N]
            self.log("val/cosine_sim_mean", cos_sims.mean().item(), sync_dist=True)
            self.log("val/cosine_sim_std", cos_sims.std().item(), sync_dist=True)

        self._log_xai_diagnostics('val', cos_sims)
        self._log_per_speaker_metrics('val', self.speaker_ids_val, self.generated, self.references)

        if self.logger and hasattr(self.logger, 'experiment'):
            import wandb
            n = min(50, len(self.generated))
            table = wandb.Table(
                columns=["reference", "generated"],
                data=[[self.references[i], self.generated[i]] for i in range(n)]
            )
            gen_lengths = [len(g.split()) for g in self.generated]
            ref_lengths = [len(r.split()) for r in self.references]
            self.logger.experiment.log({
                "val/translation_examples": table,
                "val/gen_length_hist": wandb.Histogram(gen_lengths),
                "val/ref_length_hist": wandb.Histogram(ref_lengths),
                "trainer/global_step": self.global_step,
            })

        self.set_container()

    def on_test_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)

        save_dir = self.logger.save_dir if self.logger and hasattr(self.logger, 'save_dir') else '.'
        dump_test_outputs(save_dir, self.references, self.generated, logger=self.logger, split='test')

        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)
        self._log_per_speaker_metrics('test', self.speaker_ids_val, self.generated, self.references)
        self.set_container()

    def configure_optimizers(self):
        if self.fusion_lr is not None:
            # Already-trained fusion modules: temporal encoder + projectors that exist
            # in the upstream/v22 base ckpt. Soft-freezed via fusion_lr (typically 10× smaller).
            fusion_modules = [self.temporal_encoder, self.fusion_proj,
                              self.spatio_proj, self.spatiotemp_proj]
            # New stream projectors added on top of a base ckpt (random init when continuing).
            # Need higher LR than fusion_lr to learn from scratch quickly. If `new_stream_lr`
            # is set, they get their own group; otherwise fall back to fusion_lr (legacy
            # behavior — preserves v9-v23 training dynamics).
            new_stream_modules = []
            if self.aux_input_size > 0:
                new_stream_modules.append(self.aux_proj)
            if self.keypoint_dim > 0:
                new_stream_modules.append(self.kp_proj)
            if hasattr(self, 'aux_xattn'):
                new_stream_modules.append(self.aux_xattn)

            if self.new_stream_lr is None:
                fusion_modules.extend(new_stream_modules)
                new_stream_modules = []

            fusion_ids = {id(p) for m in fusion_modules for p in m.parameters()}
            new_stream_ids = {id(p) for m in new_stream_modules for p in m.parameters()}
            specialized_ids = fusion_ids | new_stream_ids

            param_groups = [
                {'params': [p for p in self.parameters() if id(p) not in specialized_ids], 'lr': self.lr},
                {'params': [p for p in self.parameters() if id(p) in fusion_ids], 'lr': self.fusion_lr},
            ]
            if new_stream_modules:
                param_groups.append({
                    'params': [p for p in self.parameters() if id(p) in new_stream_ids],
                    'lr': self.new_stream_lr,
                })
                log.info(f"Optimizer: 3 groups (T5+LoRA lr={self.lr}, trained fusion lr={self.fusion_lr}, new streams lr={self.new_stream_lr})")
        else:
            param_groups = self.parameters()
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        if self.scheduler_config is not None:
            from torch.optim.lr_scheduler import LambdaLR
            from utils.helpers import instantiate_from_config
            sched = instantiate_from_config(self.scheduler_config)
            log.info("Using LambdaLR (per-step) scheduler from scheduler_config — paper-style cosine+warmup")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": LambdaLR(optimizer, lr_lambda=sched.schedule),
                    "interval": "step",
                    "frequency": 1,
                },
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.lr_scheduler_mode,
            factor=0.5,
            patience=self.lr_patience,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.monitor,
            },
        }