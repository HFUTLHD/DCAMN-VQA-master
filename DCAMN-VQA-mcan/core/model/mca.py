# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.capsatt_sdcam import caps_att
from core.model.capsatt_visualmap import caps_visual
import torch.nn as nn
import torch.nn.functional as F
import torch, math

class AttFlat_caps(nn.Module):
    def __init__(self, __C):
        super(AttFlat_caps, self).__init__()
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
            __C.HIDDEN_SIZE
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
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)  # b,head,seq_len,512/8
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))  # b,seq_len,512

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x
        # b,seq_len,512


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))  # b,100,512

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)  # b,100,512
        ))  # b,100,512

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))  # b,100,512

        return x

class MHAtt_ground(nn.Module):
    def __init__(self, __C):
        super(MHAtt_ground, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted, visual_map = self.att(v, k, q, mask)  # b,head,seq_len,512/8
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted, visual_map

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        visual_map = scores.masked_fill(mask, 0).sum(dim=-1).mean(dim=1)  # b,100
        att_map = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), visual_map

class SGA_ground(nn.Module):
    def __init__(self, __C):
        super(SGA_ground, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt_ground(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))  # b,100,512

        h, visual_map = self.mhatt2(y, y, x, y_mask)
        visual_map = visual_map.masked_fill(x_mask.squeeze(1).squeeze(1), -1e9)
        visual_map = F.softmax(visual_map, dim=-1)
        x = self.norm2(x + self.dropout2(
            h  # b,100,512
        ))  # b,100,512

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))  # b,100,512

        return x, visual_map

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y


def att_mask(att, att_mask):
    value, att_argmax = att.topk(att.size(1), dim=1, largest=True)
    b = att.size(1) - att_mask.sum(dim=1)
    b = b * 0.5
    mid_list = []
    for i in range(att.size(0)):
        mid = value[i][int(b[i].item())]
        mid_list.append(mid)
    mid_t = torch.stack(mid_list, dim=0).unsqueeze(-1)  # b,1
    mid_t = mid_t.repeat(1, att.size(1))
    mask = mid_t > att
    return mask

class MCA_attmask15(nn.Module):
    def __init__(self, __C):
        super(MCA_attmask15, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list_1 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.dec_list_2 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.attflat_lang = AttFlat_caps(__C)
        self.attflat_img = AttFlat_caps(__C)
        self.attflat_img_2 = AttFlat_caps(__C)

        self.mid_feat_extract_img = caps_att(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                             out_dim=__C.HIDDEN_SIZE)
        self.mid_feat_extract_lang = caps_att(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                              out_dim=__C.HIDDEN_SIZE)

        self.caps_visualmap_branch1_img = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                      out_dim=__C.HIDDEN_SIZE)
        self.caps_visualmap_branch1_lang = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                       out_dim=__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        lang_query_reserve = self.attflat_lang(x, x_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve
        y_branch1 = y_branch2 = y

        for dec in self.dec_list_1:
            y_branch1 = dec(y_branch1, x, y_mask, x_mask)
            img_query_branch1, c_visual = self.caps_visualmap_branch1_img(lang_query_branch1, y_branch1,y_mask)  # b,1024 b,100
            lang_query_branch1, c_lang = self.caps_visualmap_branch1_lang(img_query_branch1, x, x_mask)  # b,1024 b,14

        img_feat_query = self.attflat_img(y_branch1, y_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,2048
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,2048
        proj_feat = lang_feat_f + img_feat_f

        y_att_mask = att_mask(c_visual, y_mask.squeeze(1).squeeze(1))  # b,100
        mul_y = torch.ones_like(y_branch2)  # b,100,512
        mul_y = mul_y.masked_fill(y_att_mask.unsqueeze(-1), 0.3)
        y_branch2 = y_branch2 * mul_y

        for dec in self.dec_list_2:
            y_branch2 = dec(y_branch2, x, y_mask, x_mask)
            img_query_branch2 = self.mid_feat_extract_img(lang_query_branch2, y_branch2,y_mask)  # b,1024 b,100
            lang_query_branch2 = self.mid_feat_extract_lang(img_query_branch2, x, x_mask)  # b,1024 b,14

        img_feat_query_2 = self.attflat_img_2(y_branch2, y_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query_2, img_query_branch2], dim=-1)  # b,2048
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,2048
        proj_feat_2 = lang_feat_f_2 + img_feat_f_2

        return proj_feat, proj_feat_2

class MCA_attmask13_ground(nn.Module):
    def __init__(self, __C):
        super(MCA_attmask13_ground, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list_1 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.dec_list_2 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.attflat_lang = AttFlat_caps(__C)
        self.attflat_img = AttFlat_caps(__C)
        self.attflat_img_2 = AttFlat_caps(__C)

        self.mid_feat_extract_img = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                             out_dim=__C.HIDDEN_SIZE)
        self.mid_feat_extract_lang = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                              out_dim=__C.HIDDEN_SIZE)

        self.caps_visualmap_branch1_img = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                      out_dim=__C.HIDDEN_SIZE)
        self.caps_visualmap_branch1_lang = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                       out_dim=__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        lang_query_reserve = self.attflat_lang(x, x_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve
        y_branch1 = y_branch2 = y

        for idx, dec in enumerate(self.dec_list_1):
            y_branch1 = dec(y_branch1, x, y_mask, x_mask)
            img_query_branch1, c_visual = self.caps_visualmap_branch1_img(lang_query_branch1, y_branch1,y_mask)  # b,1024 b,100
            lang_query_branch1, c_lang = self.caps_visualmap_branch1_lang(img_query_branch1, x, x_mask)  # b,1024 b,14
            if idx == 5:
                c_out1 = c_visual

        img_feat_query = self.attflat_img(y_branch1, y_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,2048
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,2048
        proj_feat = lang_feat_f + img_feat_f

        y_att_mask = att_mask(c_visual, y_mask.squeeze(1).squeeze(1))  # b,100
        mul_y = torch.ones_like(y_branch2)  # b,100,512
        mul_y = mul_y.masked_fill(y_att_mask.unsqueeze(-1), 0.3)
        y_branch2 = y_branch2 * mul_y

        for idx, dec in enumerate(self.dec_list_2):
            y_branch2 = dec(y_branch2, x, y_mask, x_mask)
            img_query_branch2, c_visual_2 = self.mid_feat_extract_img(lang_query_branch2, y_branch2,y_mask)  # b,1024 b,100
            lang_query_branch2, c_lang_2 = self.mid_feat_extract_lang(img_query_branch2, x, x_mask)  # b,1024 b,14
            if idx == 5:
                c_out2 = c_visual_2

        img_feat_query_2 = self.attflat_img_2(y_branch2, y_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query_2, img_query_branch2], dim=-1)  # b,2048
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,2048
        proj_feat_2 = lang_feat_f_2 + img_feat_f_2

        return proj_feat, proj_feat_2, c_out1 + c_out2

class MCA_ED_ground(nn.Module):
    def __init__(self, __C):
        super(MCA_ED_ground, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA_ground(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y, visual_map = dec(y, x, y_mask, x_mask)

        return x, y, visual_map
        # b,14,512  b,100,512

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

        # self.linear_merge = nn.Linear(
        #     __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
        #     __C.FLAT_OUT_SIZE
        # )
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

        return x_atted
