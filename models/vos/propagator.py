import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from models.vos import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


class MaskEncoder(nn.Module):
    def __init__(self, out_c=256, stride=16):
        super().__init__()

        self.mask_encode = nn.Conv2d(1, out_c, kernel_size=stride, stride=stride, padding=0)
        self.fuser = FeatureFusionBlock(out_c + out_c, out_c)

    def forward(self, image, mask, feat):
        mask_embed = self.mask_encode(mask)
        res = self.fuser(mask_embed, feat)
        return res


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)

class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels=[384*2, 192, 96]):
        super().__init__()

        self.fuse = FeatureFusionBlock(channels[0]+2, channels[1])
        self.up_16_8 = UpsampleBlock(channels[1], channels[1], channels[2])  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(channels[2], channels[2], channels[2])  # 1/8 -> 1/4
        self.pred = nn.Conv2d(channels[2], 16, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, v16, f16, f8, f4, slot_map):
        x = self.fuse(torch.cat([v16.view_as(f16), slot_map], dim=1), f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        x = self.pred(F.relu(x))
        x = F.pixel_shuffle(x, 4)
        return x

def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([
        torch.prod(1 - prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))
    prob = F.softmax(logits, dim=dim)

    if return_logits:
        return prob, logits
    else:
        return prob


def aggregate_with_logits(logits, dim, return_logits=False):
    prob = torch.sigmoid(logits)

    new_prob = torch.cat([
        torch.prod(1 - prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))
    prob = F.softmax(logits, dim=dim)

    if return_logits:
        return prob, logits
    else:
        return prob


class VOSHead(nn.Module):
    def __init__(self, c=[96,192,384]):
        super(VOSHead, self).__init__()
        self.lvl_fuser = ResBlock(c[2]+c[1], c[2], 1)
        self.mask_encoder = MaskEncoder(c[2])
        self.key_encoder = KeyProjection(c[2], 64)
        self.value_encoder = nn.Conv2d(c[2], c[2], kernel_size=3, padding=1)
        self.decoder = Decoder(channels=[c[2]*2, c[1], c[0]])
        self.slot_pred = ResBlock(c[2]*2, 64)

        self.memory_key = None
        self.memory_mask_feature = None
        self.memory_fg_bg_count = None
        self.memory_fg_bg = None

    def clean_memory(self):
        self.memory_key = None
        self.memory_mask_feature = None
        self.memory_fg_bg_count = None
        self.memory_fg_bg = None

    def compute_affinity(self, mk, qk, top_k=None):
        B, CK, H, W = qk.shape
        qk = qk.flatten(start_dim=2)
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, THW, HW

        if top_k is None:
            maxes = torch.max(affinity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum
        else:
            top_k = min(top_k, affinity.shape[1])
            values, indices = torch.topk(affinity, k=top_k, dim=1)
            values = F.softmax(values, dim=1)
            affinity.zero_().scatter_(1, indices, values)  # B * THW * HW

        return affinity

    def propagate_feat(self, affinity, mv):
        mem = torch.bmm(mv, affinity)  # Weighted-sum B, CV, HW
        return mem

    def extract_slot(self, mask, slot_feat, feat_h, feat_w, resize='nearest'):
        mask = F.interpolate(mask, size=(feat_h, feat_w), mode=resize)
        bg_mask, fg_mask = mask[:,:1] * (mask[:,:1] > 0.5).int(), mask[:,1:] * (mask[:,1:] > 0.5).int()
        fg_embed = torch.sum((slot_feat * fg_mask).flatten(2), dim=-1)
        fg_count = torch.sum(fg_mask.flatten(2), dim=-1) + 1e-5  # b c
        bg_embed = torch.sum((slot_feat * bg_mask).flatten(2), dim=-1)
        bg_count = torch.sum(bg_mask.flatten(2), dim=-1) + 1e-5
        return fg_embed, bg_embed, fg_count, bg_count

    def update_slot(self, slot_feat, mask, feat_h, feat_w, resize='nearest', multi_frame=False):
        fg_embed, bg_embed, fg_count, bg_count = self.extract_slot(
            torch.cat([1 - mask, mask], dim=1), slot_feat, feat_h, feat_w, resize=resize)
        if self.memory_fg_bg is None:
            if multi_frame:
                self.memory_fg_bg = torch.sum(torch.stack([fg_embed, bg_embed], dim=1), dim=0, keepdim=True)  # b 2 c
                self.memory_fg_bg_count = torch.sum(torch.stack([fg_count, bg_count], dim=1), dim=0, keepdim=True)  # b 2 c
            else:
                self.memory_fg_bg = torch.stack([fg_embed, bg_embed], dim=1)  # b 2 c
                self.memory_fg_bg_count = torch.stack([fg_count, bg_count], dim=1)  # b 2 c
        else:
            self.memory_fg_bg += torch.stack([fg_embed, bg_embed], dim=1)
            self.memory_fg_bg_count += torch.stack([fg_count, bg_count], dim=1)

    def forward_eval(self, mem_feat, mem_mask, mem_image, query_feat, query_image, frame_memory, frame_query, topk=None, mem_gap=3):
        mem_t, query_t = len(frame_memory), len(frame_query)
        feat_h, feat_w = mem_feat[-1].shape[-2:]
        assert mem_t >= 1 and query_t >= 1, "segment frames is NONE. {}/{}".format(mem_t, query_t)

        mem_feat_16 = [F.interpolate(x_, mem_feat[-1].shape[-2:], mode='bilinear', align_corners=False) for x_ in mem_feat[1:]] # 8x-16x
        mem_feat_16 = self.lvl_fuser(torch.cat(mem_feat_16, dim=1))  # bt c h w
        query_feat_16 = [F.interpolate(x_, query_feat[-1].shape[-2:], mode='bilinear', align_corners=False) for x_ in query_feat[1:]] # 8x-16x
        query_feat_16 = self.lvl_fuser(torch.cat(query_feat_16, dim=1))  # bt c h w

        mem_key_feat = rearrange(self.key_encoder(mem_feat_16), '(b t) c h w -> b c t h w', t=mem_t)
        mem_image_feat = self.value_encoder(mem_feat_16)
        query_key_feat = rearrange(self.key_encoder(query_feat_16), '(b t) c h w -> b c t h w', t=query_t)
        query_image_feat = rearrange(self.value_encoder(query_feat_16), '(b t) c h w -> b c t h w', t=query_t)
        query_feat_16 = rearrange(query_feat_16, '(b t) c h w -> b t c h w', t=query_t)

        ############# init memory
        if self.memory_key is None or self.memory_mask_feature is None:
            self.memory_key = mem_key_feat.flatten(2)  # b c thw
            mask_feat = self.mask_encoder(mem_image.flatten(0,1), mem_mask.flatten(0,1), mem_feat_16)  # bt c h w x16 384
            self.memory_mask_feature = rearrange(mask_feat, '(b t) c h w -> b c (t h w)', t=mem_t)  # b c thw
            slot_feat = self.slot_pred(torch.cat([mask_feat, mem_image_feat], dim=1))  # bt c h w
            self.update_slot(slot_feat, mem_mask.flatten(0,1), feat_h, feat_w, resize='nearest', multi_frame=True)
        #############

        ############# temporal propagation
        query_feat = [rearrange(x, '(b t) c h w -> b t c h w', t=query_t) for x in query_feat]
        results_logits, results_prob = [], []
        for i in range(query_t):
            assert len(self.memory_key) == len(self.memory_mask_feature), "memory size unmatch."

            affinity = self.compute_affinity(mk=self.memory_key, qk=query_key_feat[:, :, i], top_k=topk)  # b thw hw
            prop_mask_feat = self.propagate_feat(affinity, self.memory_mask_feature).view_as(query_image_feat[:, :, i])  # b c h w
            slot_feat = self.slot_pred(torch.cat([prop_mask_feat, query_image_feat[:, :, i]], dim=1))  # b c h w
            slot_map = torch.bmm(self.memory_fg_bg/self.memory_fg_bg_count, slot_feat.flatten(2)) / math.sqrt(64.)
            slot_map = rearrange(slot_map, 'b c (h w) -> b c h w', h=feat_h, w=feat_w)

            logits = self.decoder(prop_mask_feat, query_image_feat[:, :, i], query_feat[1][:, i], query_feat[0][:, i], slot_map)  # x16  x8  x4
            prob, logits = aggregate_with_logits(logits, dim=1, return_logits=True)
            prob = prob[:, 1:]
            results_logits.append(logits)
            results_prob.append(prob)

            if i % mem_gap == 0 and i < query_t-1:
                self.memory_key = torch.cat([self.memory_key, query_key_feat[:, :, i].flatten(2)], dim=-1)
                mask_feat = self.mask_encoder(query_image[:, i], prob, query_feat_16[:, i])  # b c h w
                self.memory_mask_feature = torch.cat([self.memory_mask_feature, mask_feat.flatten(2)], dim=-1)

        return results_logits, results_prob
