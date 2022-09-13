# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_attmask13_ground

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )
    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

# 主要网络
class Net_attmaskturnsof_ground(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net_attmaskturnsof_ground, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_attmask13_ground(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.proj_norm_2 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_2 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, img_feat, ques_ix):
        # img_feat [64, 100, 2048]
        # ques_ix [64, 14]

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))   # torch.Size([64, 1, 1, 14])
        img_feat_mask = self.make_mask(img_feat)    # torch.Size([64, 1, 1, 100])

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)  # torch.Size([64, 14, 300])
        lang_feat, _ = self.lstm(lang_feat)     # torch.Size([64, 14, 512])

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)   # torch.Size([64, 100, 512])

        # Backbone Framework
        proj_feat, proj_feat_2, c_visual = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )   # [b, 14, 512] [b, 100, 512]
        proj_feat = self.proj_norm(proj_feat)   # torch.Size([64, 1024])
        proj_feat = self.proj(proj_feat)

        proj_feat_2 = self.proj_norm_2(proj_feat_2)  # torch.Size([64, 1024])
        proj_feat_2 = self.proj_2(proj_feat_2)
        # lang_feat = self.attflat_lang(
        #     lang_feat,
        #     lang_feat_mask
        # )   # torch.Size([64, 1024])
        #
        # img_feat = self.attflat_img(
        #     img_feat,
        #     img_feat_mask
        # )   # torch.Size([64, 1024])

        return proj_feat, proj_feat_2, c_visual


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
