from turtle import forward
import scipy as sp
import torch
from torch import nn
import torch.nn.functional as F

class VPT_Prompt(nn.Module):
    def __init__(self, prompt_length, prompt_dim):
        super(VPT_Prompt, self).__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, prompt_dim))
        nn.init.kaiming_normal_(self.prompt, a=0, mode='fan_out')

    def forward(self):
        return self.prompt
    
class DiT_Prompt_v1(nn.Module):
    def __init__(self, base_channels, prompt_length, prompt_dim):
        super(DiT_Prompt, self).__init__()
        self.base_channels = base_channels
        self.prompt_length = prompt_length
        self.vision_dim = prompt_dim
        self.model = nn.Sequential(
            # nn.LayerNorm(),
            nn.Conv2d(in_channels=1, out_channels=self.base_channels, kernel_size=1), # [1,224,224]=>[4,224,224]

            nn.GroupNorm(1,self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.base_channels, out_channels=2*self.base_channels, kernel_size=3, stride=2, padding=1), #[4,224,224]=>[8,112,112]
            nn.GroupNorm(1,2*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=2*self.base_channels, out_channels=2*self.base_channels, kernel_size=3, padding=1), #[8,112,112]=>[8,112,112]

            nn.GroupNorm(1,2*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=2*self.base_channels, out_channels=4*self.base_channels, kernel_size=3, stride=2, padding=1), #[8,112,112]=>[16,56,56]
            nn.GroupNorm(1,4*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=4*self.base_channels, out_channels=4*self.base_channels, kernel_size=3, padding=1), #[16,56,56]=>[16,56,56]

            nn.GroupNorm(1,4*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=4*self.base_channels, out_channels=8*self.base_channels, kernel_size=3, stride=2, padding=1), #[16,56,56]=>[32,28,28]
            nn.GroupNorm(1,8*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=8*self.base_channels, out_channels=8*self.base_channels, kernel_size=3, padding=1), #[32,28,28]=>[32,28,28]

            nn.GroupNorm(1,8*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=8*self.base_channels, out_channels=16*self.base_channels, kernel_size=3, stride=2, padding=1), #[32,28,28]=>[64,14,14]
            nn.GroupNorm(1,16*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=16*self.base_channels, out_channels=16*self.base_channels, kernel_size=3, padding=1), #[64,14,14]=>[64,14,14]

            nn.GroupNorm(1,16*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=16*self.base_channels, out_channels=32*self.base_channels, kernel_size=3, stride=2, padding=1), #[64,14,14]=>[128,7,7]
            nn.GroupNorm(1,32*self.base_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=32*self.base_channels, out_channels=32*self.base_channels, kernel_size=3, padding=1), #[128,7,7]=>[8,96]

            nn.Flatten(),
            nn.SiLU(),
            nn.Linear(self.base_channels*32*49, self.prompt_length*self.vision_dim, bias=True)
        )

    def forward(self, x):
        return self.model(x)
    
class DiT_Prompt(nn.Module):
    def __init__(self, base_channels, prompt_length, prompt_dim, prompt_depth, specific_prompt=True, global_prompt=True):
        super(DiT_Prompt, self).__init__()
        self.base_channels = base_channels
        self.prompt_length = prompt_length//2
        self.prompt_global = prompt_length//2
        self.prompt_dim = prompt_dim
        self.specific_prompt = specific_prompt
        self.global_prompt = global_prompt
        if specific_prompt:
            self.model = nn.Sequential(
                nn.Conv2d(1, self.base_channels, kernel_size=3, stride=2, padding=1), # [1,224,224] => [16, 112,112]
                # nn.BatchNorm2d(self.base_channels),
                nn.SiLU(),

                nn.Conv2d(self.base_channels, 2*self.base_channels, kernel_size=3, stride=2, padding=1), # [16,112,112]=>[32,56,56]
                # nn.BatchNorm2d(2*self.base_channels),
                nn.SiLU(),

                nn.Conv2d(2*self.base_channels, 4*self.base_channels, kernel_size=3, stride=2, padding=1),# [32,56,56]=>[64,28,28]
                # nn.BatchNorm2d(4*self.base_channels),
                nn.SiLU(),

                nn.Conv2d(4*self.base_channels, 8*self.base_channels, kernel_size=3, stride=2, padding=1),#[64,28,28]=>[128,14,14]
                # nn.BatchNorm2d(8*self.base_channels),
                nn.SiLU(),

                nn.Conv2d(8*self.base_channels, 16*self.base_channels, kernel_size=3, stride=2, padding=1),#[128,14,14]=>[256,7,7]
                # nn.BatchNorm2d(16*self.base_channels),
                nn.SiLU(),

                nn.Flatten(),
                nn.Linear(16*self.base_channels*7*7, self.prompt_length*self.prompt_dim, bias=True)
            )
        # self.global_prompt = nn.Parameter(torch.randn(self.prompt_global, self.vision_dim))
        if global_prompt:
            self.global_prompt = nn.ModuleList([
                VPT_Prompt(self.prompt_global, self.prompt_dim) for i in range(prompt_depth)
            ])

    def forward(self, x, i):
        if self.specific_prompt and self.global_prompt:
            x = self.model(x)
            x = x.view(-1, self.prompt_length, self.prompt_dim)
            global_prompt = self.global_prompt[i]().unsqueeze(0).expand(x.size(0), -1, -1)
            return torch.concat([x, global_prompt], dim=1)
        elif self.specific_prompt:
            x = self.model(x)
            x = x.view(-1, self.prompt_length, self.prompt_dim)
            return x
        elif self.global_prompt:
            global_prompt = self.global_prompt[i]().unsqueeze(0).expand(x.size(0), -1, -1)
            return global_prompt
        else:
            return None

class MaPLe_Prompt(nn.Module):
    def __init__(self, prompt_length, visual_dim, textual_dim, prompt_depth, shared=True):
        super(MaPLe_Prompt, self).__init__()
        self.prompt_depth = prompt_depth
        self.shared = shared
        self.textual_prompts = nn.ModuleList([
            VPT_Prompt(prompt_length, textual_dim) for i in range(prompt_depth)
        ])
        if shared:
            self.t2i = nn.Linear(textual_dim, visual_dim)
        else:
            self.t2i = nn.ModuleList([
                nn.Linear(textual_dim, visual_dim) for i in range(prompt_depth)
            ])
    
    def forward(self):
        visual_prompts = []
        textual_prompts = []
        for i in range(self.prompt_depth):
            tmp_textual_prompts = self.textual_prompts[i]()
            textual_prompts.append(tmp_textual_prompts)
            if self.shared:
                visual_prompts.append(self.t2i(tmp_textual_prompts))
            else:
                visual_prompts.append(self.t2i[i](tmp_textual_prompts))
        return textual_prompts, visual_prompts
    
class CLIP_Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(CLIP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class FedTPGPromptGenerator(nn.Module):
    def __init__(self, kv_dim, embed_dim, prompt_length, num_heads):
        super(FedTPGPromptGenerator, self).__init__()
        self.prompt_length = prompt_length
        
        # 可学习的查询向量 Q
        self.query = nn.Parameter(torch.randn(prompt_length, embed_dim))
        
        # 键向量和值向量的线性变换矩阵
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
        # 后续的 MLP 处理
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, text_embeddings):
        # 将文本嵌入进行线性变换得到键向量 K 和值向量 V
        keys = self.key_proj(text_embeddings)
        values = self.value_proj(text_embeddings)
        
        # 进行交叉注意力计算
        # 输入为 (query, key, value)
        query = self.query.unsqueeze(1).expand(-1, text_embeddings.size(1), -1)
        attn_output, _ = self.cross_attention(query, keys, values)
        
        # 通过 MLP 生成最终的提示向量 P
        # prompt_vectors = self.mlp(attn_output.squeeze(1))
        attn_output = attn_output.permute(1, 0, 2)
        prompt_vectors = self.mlp(attn_output)
        
        return prompt_vectors