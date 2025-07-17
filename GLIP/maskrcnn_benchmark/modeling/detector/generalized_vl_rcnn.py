# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""

from cgitb import text
from regex import R
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
from diffusers import AutoencoderKL
from maskrcnn_benchmark.modeling.dit import load_dit
from maskrcnn_benchmark.modeling.dit.diffusion import create_diffusion
import clip

from maskrcnn_benchmark.modeling.prompt import VPT_Prompt, DiT_Prompt, MaPLe_Prompt, CLIP_Adapter, FedTPGPromptGenerator


def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j,i] == -1:
                output_label[j,i] = -100
                continue

            if (not input_ids[j,i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j,i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j,i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j,i] = -100
            
            if greenlight_map is not None and greenlight_map[j,i] != 1:
                output_label[j,i] = -100 # If this location should not be masked
    return input_ids, output_label


class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_backbone(cfg) # [96, 192, 384, 768]

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg) # 768

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False
        
        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS 
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
            self.class_name_to_knowledge = load_from_yaml_file(self.cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])
        from torch import nn

        # self.vision_dim = 96 # 768
        self.vision_dim = 96
        self.language_dim = 768

        self.visual_prompts = None
        self.textual_prompts = None
        self.shared = False
        if cfg.PROMPT.ENABLE:
            self.prompt_depth = cfg.PROMPT.PROMPT_DEPTH
            self.prompt_length = cfg.PROMPT.PROMPT_LENGTH
            if cfg.PROMPT.TYPE == 'Mask-VPT':
                # load vae
                self.vae = AutoencoderKL.from_pretrained('oaaoaa/mask_vae')
                # load dit
                self.dit = load_dit(model_type='DiT-My/2', path='/home1/yanweicai/workspace/prompt/PromptDiffusion/DiT-main/results/002-DiT-My-2-timestep100/checkpoints/0094200.pt', latent_size=28)
                # load diffusion

                diffusion_steps = 100
                self.diffusion = create_diffusion(timestep_respacing="ddim25", diffusion_steps=diffusion_steps, noise_schedule='squaredcos_cap_v2') 
                # FIXME change to dit dim and vision dim
                self.dit_latent_size = 4
                self.dit_latent_dim = 28 
                self.dit_dim = self.dit_latent_dim * self.dit_latent_dim

                self.dit2vision = DiT_Prompt(8, self.prompt_length, self.vision_dim, self.prompt_depth)

                self.clip_model, preprocess = clip.load("ViT-B/32")

            elif cfg.PROMPT.TYPE == 'VPT':
                self.visual_prompts = nn.ModuleList([
                    VPT_Prompt(self.prompt_length, self.vision_dim)
                    for i in range(self.prompt_depth)
                ])

            elif cfg.PROMPT.TYPE == 'MaPLe':
                # self.visual_prompts = nn.Parameter(torch.randn(self.prompt_depth, self.prompt_length, self.vision_dim))
                # nn.init.kaiming_normal_(self.visual_prompt, a=0, mode='fan_out')
                # self.v2t = nn.Linear(self.vision_dim, self.language_dim)
                # self.textual_prompts = self.v2t(self.visual_prompt)

                self.prompts = MaPLe_Prompt(self.prompt_length, self.vision_dim, self.language_dim, self.prompt_depth, self.shared)

                    
            elif cfg.PROMPT.TYPE == 'Sprompts':
                self.visual_prompts = nn.ModuleList([
                    VPT_Prompt(self.prompt_length, self.vision_dim)
                    for i in range(self.prompt_depth)
                ])
                self.textual_prompts = nn.ModuleList([
                    VPT_Prompt(self.prompt_length, self.language_dim)
                    for i in range(self.prompt_depth)
                ])
            elif cfg.PROMPT.TYPE == 'CoOp':
                self.textual_prompts = nn.ModuleList([
                    VPT_Prompt(self.prompt_length, self.language_dim)
                    for i in range(self.prompt_depth)
                ])
            elif cfg.PROMPT.TYPE == 'Mask-VPT-V2':
                # load vae
                self.vae = AutoencoderKL.from_pretrained('oaaoaa/mask_vae')
                # load dit
                self.dit = load_dit(model_type='DiT-My/2', path='/home1/yanweicai/workspace/prompt/PromptDiffusion/DiT-main/results/002-DiT-My-2-timestep100/checkpoints/0094200.pt', latent_size=28)
                # load diffusion

                diffusion_steps = 100
                self.diffusion = create_diffusion(timestep_respacing="ddim25", diffusion_steps=diffusion_steps, noise_schedule='squaredcos_cap_v2') 
                # FIXME change to dit dim and vision dim
                self.dit_latent_size = 4
                self.dit_latent_dim = 28 
                self.dit_dim = self.dit_latent_dim * self.dit_latent_dim

                self.dit2vision = DiT_Prompt(8, self.prompt_length, self.vision_dim, self.prompt_depth)
                self.clip_model, preprocess = clip.load("ViT-B/32")
                
                self.textual_prompts = nn.ModuleList([
                    VPT_Prompt(self.prompt_length, self.language_dim)
                    for i in range(self.prompt_depth)
                ])
            elif cfg.PROMPT.TYPE == 'Mask-VPT-V3':
                                # load vae
                self.vae = AutoencoderKL.from_pretrained('oaaoaa/mask_vae')
                # load dit
                self.dit = load_dit(model_type='DiT-My/2', path='/home1/yanweicai/workspace/prompt/PromptDiffusion/DiT-main/results/002-DiT-My-2-timestep100/checkpoints/0094200.pt', latent_size=28)
                # load diffusion

                diffusion_steps = 100
                self.diffusion = create_diffusion(timestep_respacing="ddim25", diffusion_steps=diffusion_steps, noise_schedule='squaredcos_cap_v2') 
                # FIXME change to dit dim and vision dim
                self.dit_latent_size = 4
                self.dit_latent_dim = 28 
                self.dit_dim = self.dit_latent_dim * self.dit_latent_dim

                #   VISUAL_PROMPT: True
                #   VISUAL_GLOBAL_PROMPT: True
                #   TEXTUAL_PROMPT: True
                #   TEXTUAL_GLOBAL_PROMPT: False
                self.dit2vision = DiT_Prompt(8, self.prompt_length, self.vision_dim, self.prompt_depth, cfg.PROMPT.VISUAL_PROMPT, cfg.PROMPT.VISUAL_GLOBAL_PROMPT)
                self.clip_model, preprocess = clip.load("ViT-B/32")
                self.dit2language = DiT_Prompt(8, self.prompt_length, self.language_dim, self.prompt_depth, cfg.PROMPT.TEXTUAL_PROMPT, cfg.PROMPT.TEXTUAL_GLOBAL_PROMPT)
            elif cfg.PROMPT.TYPE == 'CLIP-Adapter':
                self.adapters = nn.ModuleList([CLIP_Adapter(28*28, 4), CLIP_Adapter(14*14, 4), CLIP_Adapter(7*7, 4), CLIP_Adapter(4*4, 4), CLIP_Adapter(2*2, 4)])
                # self.adapter1 = CLIP_Adapter(28*28, 4)
                # self.adapter2 = CLIP_Adapter(14*14, 4)
                # self.adapter3 = CLIP_Adapter(7*7, 4)
                # self.adapter4 = CLIP_Adapter(4*4, 4)
                # self.adapter5 = CLIP_Adapter(2*2, 4)
            elif cfg.PROMPT.TYPE == 'FedTPG':
                self.clip_model, preprocess = clip.load("ViT-B/32")
                self.prompt_generator = FedTPGPromptGenerator(512, self.language_dim, self.prompt_length, 4)
            elif cfg.PROMPT.TYPE == 'FedTPG_v2':
                self.clip_model, preprocess = clip.load("ViT-B/32")
                self.prompt_generators = nn.ModuleList([FedTPGPromptGenerator(512, self.language_dim, self.prompt_length, 4) for i in range(self.prompt_depth)])
        
        # self.dit2vision = nn.Linear(self.dit_dim, self.vision_dim)
        # self.dit_dim => [self.prompt_length, self.vision_dim]

        # self.dit2vision = nn.Sequential(
        #     nn.Conv2d(self.dit_latent_size, self.prompt_length, 1),
        #     nn.BatchNorm2d(self.prompt_length),
        #     nn.ReLU(), # 16*32*32
        #     nn.Flatten(-2, -1),
        #     nn.Linear(self.dit_dim, self.vision_dim)
        # )

        # n1 = nn.Conv2d(4, 64, 1)

        # n2_0 = nn.Conv2d(64, 32, 3, 2, padding=1)
        # n2_1 = nn.ReLU()
        # n3_0 = nn.Conv2d(32, 32, 3, padding=1)
        # n3_1 = nn.ReLU()

        # n4_0 = nn.Conv2d(32, 16, 3,2,1)
        # n4_1 = nn.ReLU()
        # n5_0 = nn.Conv2d(16, 16, 3, padding=1)
        # n5_1 = nn.ReLU()

        # n6 = nn.Flatten()
        # n7 = nn.Linear(16*49, 16*96)


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if self.cfg.PROMPT.ENABLE:
            self.backbone.body.eval()
            self.backbone.fpn.eval()
            self.rpn.head.eval()
            if hasattr(self.rpn.head, 'cls_logits'):
                self.rpn.head.cls_logits.eval()
            self.language_backbone.eval()
            if self.cfg.PROMPT.TYPE == 'Mask-VPT':
                for k, p in self.named_parameters():
                    if 'dit2vision' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'VPT':
                for k, p in self.named_parameters():
                    if 'visual_prompts' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'MaPLe':
                for k, p in self.named_parameters():
                    if 'prompts' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'Sprompts':
                for k, p in self.named_parameters():
                    if 'visual_prompts' in k or 'textual_prompts' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'Mask-VPT-V2':
                for k, p in self.named_parameters():
                    if 'dit2vision' in k or 'textual_prompts' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'Mask-VPT-V3':
                for k, p in self.named_parameters():
                    if 'dit2vision' in k or 'dit2language' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'CoOp':
                for k, p in self.named_parameters():
                    if 'textual_prompts' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE == 'CLIP-Adapter':
                for k, p in self.named_parameters():
                    if 'adapter' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif self.cfg.PROMPT.TYPE in ['FedTPG', 'FedTPG_v2']:
                for k, p in self.named_parameters():
                    if 'prompt_generator' in k:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        else:
            if self.freeze_backbone:
                self.backbone.body.eval()
                for p in self.backbone.body.parameters():
                    p.requires_grad = False
            if self.freeze_fpn:
                self.backbone.fpn.eval()
                for p in self.backbone.fpn.parameters():
                    p.requires_grad = False
            if self.freeze_rpn:
                if hasattr(self.rpn, 'head'):
                    self.rpn.head.eval()
                for p in self.rpn.parameters():
                    p.requires_grad = False
            if self.linear_prob:
                if self.rpn is not None:
                    for key, value in self.rpn.named_parameters():
                        if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                            value.requires_grad = False
                if self.roi_heads is not None:
                    for key, value in self.roi_heads.named_parameters():
                        if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                            value.requires_grad = False
            if self.freeze_cls_logits:
                if hasattr(self.rpn.head, 'cls_logits'):
                    self.rpn.head.cls_logits.eval()
                    for p in self.rpn.head.cls_logits.parameters():
                        p.requires_grad = False
            if self.add_linear_layer:
                if self.rpn is not None:
                    for key, p in self.rpn.named_parameters():
                        if 'tunable_linear' in key:
                            p.requires_grad = True

            if self.freeze_language_backbone:
                self.language_backbone.eval()
                for p in self.language_backbone.parameters():
                    p.requires_grad = False

    def forward(self, 
        images, 
        targets=None, 
        captions=None, 
        positive_map=None,
        greenlight_map=None):
        # def cal_torch_model_params(model):
        #     '''
        #     :param model:
        #     :return:
        #     '''
        #     # Find total parameters and trainable parameters
        #     total_params = sum(p.numel() for p in model.parameters())
        #     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     return {'total_params': total_params/10000, 'total_trainable_params': total_trainable_params/10000}

        # print(cal_torch_model_params(self.clip_model))
        # print(cal_torch_model_params(self.dit2vision))
        # print(cal_torch_model_params(self.dit))
        # print(cal_torch_model_params(self.vae))
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        visual_prompts, textual_prompts = None, None
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # textual_prompts = None
        if self.cfg.PROMPT.ENABLE:
            if self.cfg.PROMPT.TYPE == 'Sprompts':
                textual_prompts = [textual_prompt().unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for textual_prompt in self.textual_prompts]
            elif self.cfg.PROMPT.TYPE =='MaPLe':
                textual_prompts, visual_prompts = self.prompts()
                textual_prompts = [textual_prompt.unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for textual_prompt in textual_prompts]
                visual_prompts = [visual_prompt.unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for visual_prompt in visual_prompts]
            elif self.cfg.PROMPT.TYPE == 'Mask-VPT-V2':
                textual_prompts = [textual_prompt().unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for textual_prompt in self.textual_prompts]
            elif self.cfg.PROMPT.TYPE == 'Mask-VPT-V3':
                visual_prompts, textual_prompts = self.extract_visual_textual_dit_features(captions, images.tensors, self.prompt_depth)  
            elif self.cfg.PROMPT.TYPE == 'CoOp':
                textual_prompts = [textual_prompt().unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for textual_prompt in self.textual_prompts]
            elif self.cfg.PROMPT.TYPE == 'FedTPG':
                caption_tokenized = clip.tokenize(captions, truncate=True).to(device)                
                text_features = self.clip_model.encode_text(caption_tokenized).unsqueeze(0).float()
                # print(text_features.dtype)
                # exit(0)

                # print(self.prompt_generator(text_features).shape)
                # exit(0)
                textual_prompts = [self.prompt_generator(text_features)]
                # print(textual_prompts.shape)
                # exit(0)
                # assert textual_prompts.shape[1] == self.prompt_length, "Prompt length mismatch!"
                # assert textual_prompts.shape[2] == self.language_dim, "Prompt dim mismatch!"
            elif self.cfg.PROMPT.TYPE == 'FedTPG_v2':
                caption_tokenized = clip.tokenize(captions, truncate=True).to(device)                
                text_features = self.clip_model.encode_text(caption_tokenized).unsqueeze(0).float()
                textual_prompts = []
                for i in range(self.prompt_depth):
                    textual_prompts.append(self.prompt_generators[i](text_features))
            
        if self.cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
            language_dict_features, positive_map = self._forward_language_parallel(
                    captions=captions, targets=targets, device=device,
                    positive_map=positive_map)
        else:
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions,
                                                            max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                            padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True,
                                                            return_tensors='pt',
                                                            truncation=True).to(device)
                if self.use_mlm_loss:
                    if not self.mlm_loss_for_only_positives:
                        greenlight_map = None
                    input_ids, mlm_labels = random_word(
                        input_ids=tokenized.input_ids, 
                        mask_token_id=self.tokenizer.mask_token_id,
                        vocabs=self.tokenizer_vocab_ids,
                        padding_token_id=self.tokenizer.pad_token_id,
                        greenlight_map=greenlight_map)
                else:
                    input_ids = tokenized.input_ids
                    mlm_labels = None
                
                
                tokenizer_input = {"input_ids": input_ids,
                                "attention_mask": tokenized.attention_mask,
                                "prompts": textual_prompts}

                if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                    with torch.no_grad():
                        language_dict_features = self.language_backbone(tokenizer_input)
                else:
                    language_dict_features = self.language_backbone(tokenizer_input)
                
                # ONE HOT
                if self.cfg.DATASETS.ONE_HOT:
                    new_masks = torch.zeros_like(language_dict_features['masks'],
                                                device=language_dict_features['masks'].device)
                    new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
                    language_dict_features['masks'] = new_masks

                # MASK ALL SPECIAL TOKENS
                if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
                    language_dict_features["masks"] = 1 - tokenized.special_tokens_mask
                
                language_dict_features["mlm_labels"] = mlm_labels

        if self.cfg.PROMPT.ENABLE:
            if self.cfg.PROMPT.TYPE in ['Mask-VPT', 'Mask-VPT-V2']:
            # extract dit features language_dict_features['hidden'] and images
                visual_prompts = self.extract_dit_features(captions, images.tensors, self.prompt_depth)  
            elif self.cfg.PROMPT.TYPE in ['Sprompts', 'VPT']:
                visual_prompts = [visual_prompt().unsqueeze(0).repeat(images.tensors.shape[0], 1,1) for visual_prompt in self.visual_prompts]
            elif self.cfg.PROMPT.TYPE == 'MaPLe':
                visual_prompts = visual_prompts
                
        
        # visual embedding
        swint_feature_c4 = None
        if 'vl' in self.cfg.MODEL.SWINT.VERSION:
            # the backbone only updates the "hidden" field in language_dict_features
            inputs = {"img": images.tensors, "lang": language_dict_features}
            visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
        else:
            # TODO inject dit features
            visual_features = self.backbone((images.tensors, visual_prompts))

        if self.cfg.PROMPT.ENABLE and self.cfg.PROMPT.TYPE == 'CLIP-Adapter':
            ratio = 0.2
            visual_features_new = []
            for idx, visual_feature in enumerate(visual_features):
                bs = visual_features[idx].shape[0]
                w, h = visual_feature.shape[-2], visual_feature.shape[-1]
                x = self.adapters[idx](visual_feature.flatten(-2, -1)).view(bs,-1, w, h)
                visual_features_new.append(ratio * x + (1 - ratio) * visual_feature) 
            visual_features = tuple(visual_features_new)


        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]

        if self.force_boxes:
        # if True:
            proposals = []
            for t in targets:
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                proposals.append(tb)
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                _, proposal_losses, fused_visual_features = self.rpn(
                    images, visual_features, targets, language_dict_features,
                    positive_map, captions, swint_feature_c4)
            elif self.training:
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {('rpn_null_loss', null_loss)}
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map,
                                              captions, swint_feature_c4)
        if self.roi_heads:
            if self.cfg.MODEL.ROI_MASK_HEAD.PREDICTOR.startswith("VL"):
                if self.training:
                    # "Only support VL mask head right now!!"
                    assert len(targets) == 1 and len(targets[0]) == len(positive_map), "shape match assert for mask head!!"
                    # Not necessary but as a safe guard:
                    # use the binary 0/1 positive map to replace the normalized positive map
                    targets[0].add_field("positive_map", positive_map)
            # TODO: make sure that this use of language_dict_features is correct!! Its content should be changed in self.rpn
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def extract_dit_features(self, captions: torch.Tensor, images: torch.Tensor, prompt_depth: int) -> list[torch.Tensor]:
        device = images.device
        bs = images.shape[0]
        """
        Extract dit features from language features and images
        """
        z = torch.randn(bs, self.dit_latent_size, self.dit_latent_dim, self.dit_latent_dim, device=device)
        y = torch.randint(0, 1000, (bs,), device=device)

        
        image_features = self.clip_model.encode_image(images)

        caption_tokenized = clip.tokenize(captions, truncate=True).to(device)
        text_features = self.clip_model.encode_text(caption_tokenized)
        
        dit_kwargs = dict(image_features=image_features, text_features=text_features) # image_features: 
        sample_fn = self.dit.forward
        # samples, samples_latent_features = self.diffusion.p_sample_loop(
        #     sample_fn, z.shape, z, clip_denoised=False, model_kwargs=dit_kwargs, progress=False, device=device, sample_distance=sample_distance, prompt_depth=prompt_depth
        # )
        samples, samples_latent_features = self.diffusion.ddim_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=dit_kwargs, progress=False, device=device, prompt_depth=prompt_depth
        )
        # extract dit features
        # dit_features = self.dit2vision(language_features['hidden'])
        # FIXME
        # dit_features = [torch.randn(bs, self.dim_latent_size, self.dit_dim) for i in range(self.prompt_depth)]
        to_return = []
        for idx, sample in enumerate(samples_latent_features):
            sample = self.vae.decode(sample).sample # [bs,1,224,224]
            to_return.append(self.dit2vision(sample, idx).view(bs, -1, self.vision_dim))
        # for i in range(self.prompt_depth):
        #     t = torch.randn(bs, self.dit_latent_size, self.dit_latent_dim, self.dit_latent_dim).to(device)
        #     t = self.dit2vision(t)
        #     to_return.append(t)
        return to_return

    def extract_visual_textual_dit_features(self, captions: torch.Tensor, images: torch.Tensor, prompt_depth: int) -> list[torch.Tensor]:
        device = images.device
        bs = images.shape[0]
        """
        Extract dit features from language features and images
        """
        z = torch.randn(bs, self.dit_latent_size, self.dit_latent_dim, self.dit_latent_dim, device=device)
        y = torch.randint(0, 1000, (bs,), device=device)

        
        image_features = self.clip_model.encode_image(images)

        caption_tokenized = clip.tokenize(captions, truncate=True).to(device)
        text_features = self.clip_model.encode_text(caption_tokenized)
        
        dit_kwargs = dict(image_features=image_features, text_features=text_features) # image_features: 
        sample_fn = self.dit.forward
        # samples, samples_latent_features = self.diffusion.p_sample_loop(
        #     sample_fn, z.shape, z, clip_denoised=False, model_kwargs=dit_kwargs, progress=False, device=device, sample_distance=sample_distance, prompt_depth=prompt_depth
        # )
        samples, samples_latent_features = self.diffusion.ddim_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=dit_kwargs, progress=False, device=device, prompt_depth=prompt_depth
        )
        # extract dit features
        # dit_features = self.dit2vision(language_features['hidden'])
        # FIXME
        # dit_features = [torch.randn(bs, self.dim_latent_size, self.dit_dim) for i in range(self.prompt_depth)]
        to_return_visual = []
        to_return_textual = []
        for idx, sample in enumerate(samples_latent_features):
            sample = self.vae.decode(sample).sample # [bs,1,224,224]
            to_return_visual.append(self.dit2vision(sample, idx).view(bs, -1, self.vision_dim))
            to_return_textual.append(self.dit2language(sample, idx).view(bs, -1, self.language_dim))
        # for i in range(self.prompt_depth):
        #     t = torch.randn(bs, self.dit_latent_size, self.dit_latent_dim, self.dit_latent_dim).to(device)
        #     t = self.dit2vision(t)
        #     to_return.append(t)
        return to_return_visual, to_return_textual
    

    def _forward_language_parallel(self, captions=None, targets=None,
            device=None, positive_map=None):
        ktype = self.cfg.GLIPKNOW.KNOWLEDGE_TYPE
        def _construct_captions_from_class_names(class_names):
            captions = []
            for c in class_names:
                try:
                    info = self.class_name_to_knowledge[c]
                    cap = info['clean_name']

                    # combine wiki and gpt3 knowledge
                    if self.cfg.GLIPKNOW.WIKI_AND_GPT3:
                        ktype = 'def_wiki'
                        know_seq = info[ktype]

                        ktype = 'gpt3'
                        if ktype == 'gpt3' or type(info[ktype]) == list:
                            know_seq += ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])

                        cap += ': ' + know_seq

                    # only one knoweldge source is used        
                    else:
                        if ktype and ktype in info and info[ktype]:
                            if ktype == 'gpt3' or type(info[ktype]) == list:
                                know_seq = ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])
                            else: 
                                know_seq = info[ktype]
                            cap += ': ' + know_seq
                except:
                    cap = c
                    print(f'cap {cap}, c {c}')
                    
                    
                captions.append(cap)
            return captions

        if self.training:
            assert captions is None
            assert targets is not None

            max_classes_per_batch = self.cfg.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN
            if max_classes_per_batch >= len(self.class_name_list):
                shuffled_class_names = self.class_name_list.copy()
                random.shuffle(shuffled_class_names)
                if max_classes_per_batch > len(shuffled_class_names):
                    shuffled_class_names.extend(shuffled_class_names[:max_classes_per_batch
                        -len(shuffled_class_names)])
                    random.shuffle(shuffled_class_names)
            else:
                label_list = []
                label_to_idx = {}
                for target_per_im in targets:
                    labels_per_im = target_per_im.get_field('label_names')
                    for label in labels_per_im:
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_list)
                            label_list.append(label)

                label_list = label_list[:max_classes_per_batch]
                if len(label_list) < max_classes_per_batch:
                    all_neg_classes = [c for c in self.class_name_list if c not
                            in label_to_idx]
                    neg_label_list = random.sample(all_neg_classes,
                            max_classes_per_batch - len(label_list))
                    label_list.extend(neg_label_list)
                random.shuffle(label_list)
                shuffled_class_names = label_list

            label_to_shuffled_idx = {l: i for i, l in
                    enumerate(shuffled_class_names)}
            total_boxes = sum(len(t) for t in targets)
            positive_map = torch.zeros((total_boxes, max_classes_per_batch+1),
                device=device)
            offset = 0
            for target_per_im in targets:
                labels_per_im = target_per_im.get_field('label_names')
                for label in labels_per_im:
                    j = label_to_shuffled_idx.get(label, -1)
                    if j >= 0:
                        positive_map[offset, j] = 1
                    offset += 1
            captions = _construct_captions_from_class_names(shuffled_class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719
            batch_size = len(targets)

        else:
            assert captions is not None
            batch_size = 1
            assert len(captions) == 1
            class_names = captions[0]
            max_classes_per_batch = len(class_names)
            captions = _construct_captions_from_class_names(class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719

        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                     padding="longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)
        assert not self.use_mlm_loss
        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        else:
            language_dict_features = self.language_backbone(tokenizer_input)

        assert not self.cfg.DATASETS.ONE_HOT
        assert not self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL

        agg_type = self.cfg.GLIPKNOW.LAN_FEATURE_AGG_TYPE
        agg_feats = language_dict_features['hidden']
        agg_emb = language_dict_features['embedded']
        if agg_type == 'first':
            agg_feats = agg_feats[:, 0, :]
            agg_emb = agg_emb[:, 0, :]
        elif agg_type == 'mean':
            attn_mask = language_dict_features['masks']
            seq_len = attn_mask.sum(-1).unsqueeze(-1).float()
            agg_feats = agg_feats * attn_mask.unsqueeze(-1).float()
            agg_feats = agg_feats.sum(1) / seq_len
            agg_emb = agg_emb * attn_mask.unsqueeze(-1).float()
            agg_emb = agg_emb.sum(1) / seq_len
        else:
            raise ValueError('not supported GLIPKNOW.LAN_FEATURE_AGG_TYPE: {}'.format(agg_type))

        expanded_features = agg_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_embedding = agg_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        lang_dict = {}
        lang_dict["mlm_labels"] = None
        lang_dict["aggregate"] = None
        lang_dict["embedded"] = expanded_embedding
        lang_dict['hidden'] = expanded_features
        lang_dict["masks"] = torch.ones((batch_size, max_classes_per_batch+1),
                device=device, dtype=language_dict_features['masks'].dtype)
        # in GLIP setting, the token at the end of seqence is usually [PAD], and is masked out
        # if [noobj] is not masked out, the loss sum is very big, as most
        # anchors are matched to [noobj]
        lang_dict["masks"][:,-1] = 0
        return lang_dict, positive_map

