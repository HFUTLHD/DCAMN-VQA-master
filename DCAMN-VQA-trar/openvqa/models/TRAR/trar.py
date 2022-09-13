from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.TRAR.capsatt_sdcam import caps_att
from openvqa.models.TRAR.capsatt_visualmap import caps_visual

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


# ---------------------------
# ---- Attention Pooling ----
# ---------------------------
class AttFlat(nn.Module):
    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            in_size=in_channel,
            mid_size=in_channel,
            out_size=glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            in_channel
        )
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted

# --------------------------------
# ---- Local Window Generator ----
# --------------------------------
def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool)
    return masks


def getMasks(x_mask, __C):
    mask_list = []
    ORDERS = __C.ORDERS
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask = torch.from_numpy(getImgMasks(__C.IMG_SCALE, order)).byte().cuda()
            mask = torch.logical_or(x_mask, mask)
            mask_list.append(mask)
    return mask_list


# ------------------------------------
# ---- Soft or Hard Routing Block ----
# ------------------------------------

# Routing weight prediction layer
# Weight obtained by softmax or gumbel softmax
class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = F.softmax(logits, dim=-1)  #
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


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
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
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


# -------------------------------------
# ---- Dynmaic Span Self-Attention ----
# -------------------------------------

class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, __C):
        super(SARoutingBlock, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        if __C.ROUTING == 'hard':
            self.routing_block = HardRoutingBlock(__C.HIDDEN_SIZE, len(__C.ORDERS), __C.POOLING)
        elif __C.ROUTING == 'soft':
            self.routing_block = SoftRoutingBlock(__C.HIDDEN_SIZE, len(__C.ORDERS), __C.POOLING)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.size(0)
        x = v

        alphas = self.routing_block(x, tau, masks)

        if self.__C.BINARIZE:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        att_list = self.routing_att(v, k, q, masks)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list)

        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        for i in range(len(masks)):
            mask = masks[i]
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(1)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(1)), 1)

        return att_list

    def argmax_binarize(self, alphas):
        n = alphas.size()[0]
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        out[torch.arange(n), indexes] = 1
        return out


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


# -----------------------------
# ---- Transformer Encoder ----
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, __C):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# ---------------------------------
# ---- Multimodal TRAR Decoder ----
# ---------------------------------
class TRAR(nn.Module):
    def __init__(self, __C):
        super(TRAR, self).__init__()

        self.mhatt1 = SARoutingBlock(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_masks, y_mask, tau, training):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, masks=x_masks, tau=tau, training=training)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class AttFlat_cap(nn.Module):
    def __init__(self, __C):
        super(AttFlat_cap, self).__init__()
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
        # b,m,1

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED(nn.Module):
    def __init__(self, __C):
        super(TRAR_ED, self).__init__()
        self.__C = __C
        self.tau = __C.TAU_MAX
        self.training = True
        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        x_masks = getMasks(x_mask, self.__C)
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_masks, y_mask, self.tau, self.training)

        return y, x

    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training

def att_mask(att, att_mask):
    value, att_argmax = att.topk(att.size(1), dim=1, largest=True)
    b = att.size(1) - att_mask.sum(dim=1)
    b = b * 0.7
    mid_list = []
    for i in range(att.size(0)):
        mid = value[i][int(b[i].item())]
        mid_list.append(mid)
    mid_t = torch.stack(mid_list, dim=0).unsqueeze(-1)  # b,1
    mid_t = mid_t.repeat(1, att.size(1))
    mask = mid_t > att
    return mask


class TRAR_ED_attmask(nn.Module):
    def __init__(self, __C):
        super(TRAR_ED_attmask, self).__init__()
        self.__C = __C
        self.tau = __C.TAU_MAX
        self.training = True
        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.LAYER)])
        self.dec_list_1 = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])
        self.dec_list_2 = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])
        self.attflat_lang = AttFlat_cap(__C)
        self.attflat_img = AttFlat_cap(__C)
        self.attflat_img_2 = AttFlat_cap(__C)

        self.mid_feat_extract_img = caps_att(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                             out_dim=__C.HIDDEN_SIZE)
        self.mid_feat_extract_lang = caps_att(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                              out_dim=__C.HIDDEN_SIZE)

        self.caps_visualmap_branch1_img = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                      out_dim=__C.HIDDEN_SIZE)
        self.caps_visualmap_branch1_lang = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                       out_dim=__C.HIDDEN_SIZE)

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        x_masks = getMasks(x_mask, self.__C)
        for enc in self.enc_list:
            y = enc(y, y_mask)
        lang_query_reserve = self.attflat_lang(y, y_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve

        x_branch1 = x_branch2 = x
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors

        for dec in self.dec_list_1:
            # x = dec(x, y, x_masks, y_mask, self.tau, self.training)
            x_branch1 = dec(x_branch1, y, x_masks, y_mask, self.tau, self.training)
            img_query_branch1, c_visual = self.caps_visualmap_branch1_img(lang_query_branch1, x_branch1,
                                                                          x_mask)  # b,512 b,100
            lang_query_branch1, c_lang = self.caps_visualmap_branch1_lang(img_query_branch1, y, y_mask)  # b,512 b,14

        img_feat_query = self.attflat_img(x_branch1, x_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,1024
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,1024
        proj_feat = lang_feat_f + img_feat_f

        x_att_mask = att_mask(c_visual, x_mask.squeeze(1).squeeze(1))  # b,64
        y_att_mask = att_mask(c_lang, y_mask.squeeze(1).squeeze(1))
        mul_x = torch.ones_like(x)  # b,64,512
        mul_x = mul_x.masked_fill(x_att_mask.unsqueeze(-1), 0.5)
        x_branch2 = x_branch2 * mul_x
        mul_y = torch.ones_like(y)  # b,14,512
        mul_y = mul_y.masked_fill(y_att_mask.unsqueeze(-1), 0.5)
        y = y * mul_y

        for dec in self.dec_list_2:
            # x = dec(x, y, x_masks, y_mask, self.tau, self.training)
            x_branch2 = dec(x_branch2, y, x_masks, y_mask, self.tau, self.training)
            img_query_branch2 = self.mid_feat_extract_img(lang_query_branch2, x_branch2,
                                                          x_mask)  # b,512 b,100
            lang_query_branch2 = self.mid_feat_extract_lang(img_query_branch2, y, y_mask)  # b,512 b,14

        img_feat_query = self.attflat_img_2(x_branch2, x_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query, img_query_branch2], dim=-1)  # b,1024
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,1024
        proj_feat_2 = lang_feat_f_2 + img_feat_f_2

        return proj_feat, proj_feat_2

    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training
