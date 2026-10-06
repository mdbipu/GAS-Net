"""
GAS-Net: context-enriched image captioning with Geometry, Attribute and Scene guided
multimodal fusion (paper Sec. 3).

Built on top of the baseline `AttModel_MAD_SAP.py` (Up-Down decoder, label-smoothing XE,
SCST reward criterion, beam search through CaptionModel).  Everything that is new is
marked with "[GAS-Net]" together with the paper equation it implements.

Pipeline (paper Fig. 2)
    region feats V (B,M,dv) + boxes (B,M,4) + global feat Fg + scene feat s
      -> GeometricAttention  -> v_geom   (Eq. 3-17)
      -> AttributeAttention  -> v_attr   (Eq. 18-30)   [+ BCE attribute loss, Eq. 21]
      -> SceneAttention      -> v_scene  (Eq. 39-48)   [+ BCE scene loss,     Eq. 41]
      -> MultimodalFusion (projection + CMI + TGA)  -> v_fusion (Eq. 49-62)
      -> two-layer Up-Down LSTM decoder, v_fusion fed to BOTH LSTMs (Eq. 63-70)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import itertools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .CaptionModel import CaptionModel


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def expand_rep(x, rep):
    """[B/rep, ...] -> [B, ...] where every row is repeated `rep` times *consecutively*
    (the same layout DataLoader.get_batch uses for the seq_per_img captions of an image)."""
    if rep == 1:
        return x
    shape = x.size()
    x = x.unsqueeze(1).expand(shape[0], rep, *shape[1:]).contiguous()
    return x.view(shape[0] * rep, *shape[1:])


# ----------------------------------------------------------------------------------------
# [GAS-Net] Geometric information (Sec. 3.3)
# ----------------------------------------------------------------------------------------
def pairwise_geometry(boxes, signed=True, eps=1e-4, min_size=1.0):
    """Pairwise geometric descriptors g_ij between all region pairs.

    boxes : [B, M, 4] as (x1, y1, x2, y2) - the format stored by the bottom-up feature
            extractor in cocobu_box_36.  Converted to (x, y, w, h), (x, y) = top-left (Eq. 2).
    signed: True  -> Eq. (5)  signed offsets  (keeps left/right, above/below)
            False -> Eq. (4)  |offsets|       (original, direction-less)
    returns g : [B, M, M, 4],  g[b, i, j] = [dx/w_i, dy/h_i, log(w_j/w_i), log(h_j/h_i)]
    """
    x = boxes[..., 0]
    y = boxes[..., 1]
    # min_size only guards against degenerate (zero-area) boxes; boxes are in pixels.
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=min_size)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=min_size)

    dx = x.unsqueeze(1) - x.unsqueeze(2)      # [B,M,M]  entry (i,j) = x_j - x_i
    dy = y.unsqueeze(1) - y.unsqueeze(2)
    if not signed:
        dx = dx.abs()
        dy = dy.abs()
    wi, wj = w.unsqueeze(2), w.unsqueeze(1)   # [B,M,1], [B,1,M]
    hi, hj = h.unsqueeze(2), h.unsqueeze(1)

    g = torch.stack([dx / (wi + eps),
                     dy / (hi + eps),
                     torch.log((wj + eps) / (wi + eps)),
                     torch.log((hj + eps) / (hi + eps))], dim=-1)
    return g


class GeometricAttention(nn.Module):
    """Pairwise geometry -> embedding (Eq. 6) -> per-region context (mean Eq. 7 or
    relation-aware Eq. 8-10) -> geometric attention over regions (Eq. 11-17)."""

    def __init__(self, opt):
        super(GeometricAttention, self).__init__()
        self.signed = (opt.geo_repr == 'signed')
        self.agg = opt.geo_agg
        self.eps = opt.geo_eps
        self.min_size = opt.geo_min_size
        self.dg = opt.geo_dim
        dv, da = opt.att_feat_size, opt.gas_att_size

        self.embed = nn.Sequential(nn.Linear(4, self.dg), nn.ReLU(inplace=True))   # Eq. (6)
        if self.agg == 'relation':
            self.q_proj = nn.Linear(dv, self.dg)        # q_i^g : query of region i (from appearance)
            self.k_proj = nn.Linear(self.dg, self.dg)   # k_ij^g: key of the pairwise relation
        self.g_proj = nn.Linear(self.dg, da)            # Eq. (12)
        self.alpha_net = nn.Linear(da, 1)               # Eq. (14)

    def forward(self, v, r, boxes):
        """v: [B,M,dv] raw region feats; r: [B,M,da] = W_vr v (Eq. 11); boxes: [B,M,4]."""
        B, M = v.size(0), v.size(1)
        g = pairwise_geometry(boxes, self.signed, self.eps, self.min_size)   # [B,M,M,4]
        z = self.embed(g)                                                     # [B,M,M,dg]

        eye = torch.eye(M, device=v.device, dtype=v.dtype)                    # j != i only
        if self.agg == 'mean':
            offdiag = (1.0 - eye).view(1, M, M, 1)
            ctx = (z * offdiag).sum(2) / float(M - 1)                         # Eq. (7)
        else:
            q = self.q_proj(v)                                                # [B,M,dg]
            k = self.k_proj(z)                                                # [B,M,M,dg]
            e = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(self.dg)             # Eq. (8)  [B,M,M]
            e = e + eye.unsqueeze(0) * (-1e9)                                 # exclude self-relation
            alpha_g = F.softmax(e, dim=2)                                     # Eq. (9)
            ctx = (alpha_g.unsqueeze(-1) * z).sum(2)                          # Eq. (10) [B,M,dg]

        g_prime = self.g_proj(ctx)                                            # Eq. (12) [B,M,da]
        v_g = torch.tanh(r + g_prime)                                         # Eq. (13)
        e_i = self.alpha_net(v_g).squeeze(-1)                                 # Eq. (14) [B,M]
        alpha = F.softmax(e_i, dim=1)                                         # Eq. (15)
        return torch.bmm(alpha.unsqueeze(1), v).squeeze(1)                    # Eq. (17) [B,dv]


# ----------------------------------------------------------------------------------------
# [GAS-Net] Attribute branch (Sec. 3.4.1-3.4.2)
# ----------------------------------------------------------------------------------------
class AttributeAttention(nn.Module):
    """Multi-label attribute prediction from the global feature Fg (Eq. 20), top-T selection
    (Eq. 22), embedding + mean (Eq. 23-24) and attribute-guided attention (Eq. 25-30)."""

    def __init__(self, opt):
        super(AttributeAttention, self).__init__()
        da = opt.gas_att_size
        self.top_t = min(opt.top_t_attrs, opt.num_attrs)
        self.classifier = nn.Linear(opt.fc_feat_size, opt.num_attrs)          # Eq. (20)
        self.embed = nn.Embedding(opt.num_attrs, opt.attr_embed_size)         # W_e, Eq. (23)
        self.a_proj = nn.Linear(opt.attr_embed_size, da)                      # Eq. (25)
        self.alpha_net = nn.Linear(da, 1)                                     # Eq. (27)

    def forward(self, fg, v, r):
        """fg: [B,dv] global feature; v: [B,M,dv]; r: [B,M,da]."""
        logits = self.classifier(fg)                                          # sigma(.) applied inside the BCE loss
        # Eq. (22): top-T predicted attributes.  The selection itself is not differentiable;
        # the classifier is trained by the BCE loss, the embedding by the captioning loss.
        top_idx = torch.topk(logits.detach(), self.top_t, dim=1)[1]           # [B,T]
        a = self.embed(top_idx).mean(1)                                       # Eq. (24) [B,de]
        a_p = self.a_proj(a)                                                  # [B,da]
        v_a = torch.tanh(r + a_p.unsqueeze(1))                                # Eq. (26)
        alpha = F.softmax(self.alpha_net(v_a).squeeze(-1), dim=1)             # Eq. (27-28)
        v_attr = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)                  # Eq. (30)
        return v_attr, logits


# ----------------------------------------------------------------------------------------
# [GAS-Net] Scene branch (Sec. 3.4.3-3.4.6)
# ----------------------------------------------------------------------------------------
class SceneAttention(nn.Module):
    """Scene classifier head (Eq. 40) + scene-conditioned attention (Eq. 44-48).
    `s` is the globally pooled scene feature (Eq. 43), pre-extracted offline (see
    scripts/extract_scene_feats.py) or, as a fallback, the mean-pooled region feature."""

    def __init__(self, opt):
        super(SceneAttention, self).__init__()
        da = opt.gas_att_size
        self.drop = nn.Dropout(opt.drop_prob_lm)
        self.classifier = nn.Linear(opt.scene_feat_size, opt.num_scenes)      # Eq. (40)
        self.s_proj = nn.Linear(opt.scene_feat_size, da)                      # Eq. (44)
        self.alpha_net = nn.Linear(da, 1)                                     # Eq. (46)

    def forward(self, s, v, r):
        s = self.drop(s)
        logits = self.classifier(s)
        s_p = self.s_proj(s)                                                  # [B,da]
        v_s = torch.tanh(r + s_p.unsqueeze(1))                                # Eq. (45)
        alpha = F.softmax(self.alpha_net(v_s).squeeze(-1), dim=1)             # Eq. (46-47)
        v_scene = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)                 # Eq. (48)
        return v_scene, logits


# ----------------------------------------------------------------------------------------
# [GAS-Net] Multimodal fusion: projection + Cross-Modal Interaction + Triple Gating (Sec. 3.5)
# ----------------------------------------------------------------------------------------
class MultimodalFusion(nn.Module):
    """v_fusion = sum_m g_m * z_m  +  z_int        (Eq. 62)

    use_cmi=False -> drop z_int          (ablation "- CMI")
    use_tga=False -> use z_m instead of g_m*z_m (ablation "- TGA"; both False = direct sum)
    fusion_type='concat' -> Linear([z_1;...;z_n])   (Table 11 "Concatenation + Linear")
    With fewer than 3 active modalities (leave-one-out / single-branch ablations) only the
    available modalities / pairs are used.
    """

    def __init__(self, opt, names):
        super(MultimodalFusion, self).__init__()
        df, dv = opt.fusion_size, opt.att_feat_size
        self.names = list(names)
        self.fusion_type = opt.fusion_type
        self.proj = nn.ModuleList([nn.Linear(dv, df) for _ in self.names])       # Eq. (49-51)
        self.pairs = list(itertools.combinations(range(len(self.names)), 2))     # (g,a),(g,s),(a,s)

        self.use_cmi = bool(opt.use_cmi) and len(self.pairs) > 0
        self.use_tga = bool(opt.use_tga)
        if self.fusion_type == 'concat':
            self.concat = nn.Linear(df * len(self.names), df)
        else:
            if self.use_cmi:
                self.cmi = nn.ModuleList([nn.Linear(df, df) for _ in self.pairs])  # W_ga, W_gs, W_as
            if self.use_tga:
                self.gates = nn.ModuleList([nn.Linear(df, df) for _ in self.names])  # Eq. (56-58)

    def forward(self, vs):
        z = [proj(v) for proj, v in zip(self.proj, vs)]
        if self.fusion_type == 'concat':
            return self.concat(torch.cat(z, 1))

        if self.use_tga:
            out = sum(torch.sigmoid(gate(zi)) * zi for gate, zi in zip(self.gates, z))   # Eq. (56-61)
        else:
            out = sum(z)
        if self.use_cmi:
            z_int = sum(F.relu(w(z[i] * z[j])) for w, (i, j) in zip(self.cmi, self.pairs))  # Eq. (52-55)
            out = out + z_int
        return out


# ----------------------------------------------------------------------------------------
# Captioning model
# ----------------------------------------------------------------------------------------
class AttModel_GASNet(CaptionModel):
    def __init__(self, opt):
        super(AttModel_GASNet, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.beam_size = getattr(opt, 'beam_size', 3)        # was hard-coded to 3 in the baseline

        # [GAS-Net] context branches (each can be switched off for the ablations)
        names = []
        self.use_geo = bool(opt.use_geo)
        self.use_attr = bool(opt.use_attr)
        self.use_scene = bool(opt.use_scene)
        if self.use_geo or self.use_attr or self.use_scene:
            # W_vr of Eq. (11); r_i is shared by the geometry / attribute / scene attention
            self.region_proj = nn.Linear(opt.att_feat_size, opt.gas_att_size)
        if self.use_geo:
            self.geo = GeometricAttention(opt)
            names.append('geo')
        if self.use_attr:
            assert opt.num_attrs > 0, 'attribute branch needs opt.num_attrs (attribute label file)'
            self.attr = AttributeAttention(opt)
            names.append('attr')
        if self.use_scene:
            assert opt.num_scenes > 0, 'scene branch needs opt.num_scenes (scene label file)'
            self.scene = SceneAttention(opt)
            names.append('scene')
        self.num_modal = len(names)
        if self.num_modal > 0:
            self.fusion = MultimodalFusion(opt, names)
            self.fusion_drop = nn.Dropout(self.drop_prob_lm)
            self.fusion_size = opt.fusion_size
        else:                       # no branch -> plain Up-Down (the "Baseline" row of the ablations)
            self.fusion_size = 0

        self.lstm_core = TopDownCore_GASNet(opt, self.fusion_size)
        self.ss_prob = 0.0  # Schedule sampling probability
        self.embed = torch.nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(self.drop_prob_lm))
        self.p_att_emd = nn.Linear(self.rnn_size, self.att_hid_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.crit = LabelSmoothing(smoothing=0.2)  # LanguageModelCriterion() using label smoothing performs a little better
        self.scst_crit = RewardCriterion()
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p, 0)
        return

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # embed fc and att feats (decoder side, identical to the baseline)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        return fc_feats, att_feats

    # ------------------------------------------------------------------------------------
    # [GAS-Net] encoder: everything that is computed once per image
    # ------------------------------------------------------------------------------------
    def _encode(self, fc_feats, att_feats, boxes=None, scene_feats=None, rep=1):
        """fc_feats [B,dv], att_feats [B,M,dv], boxes [B,M,4], scene_feats [B,ds] or None.

        `rep` = number of consecutive identical rows per image (seq_per_img during XE / SCST
        training, 1 at test time).  The (expensive, per-image) context branches are computed
        once per unique image and expanded afterwards.

        returns (fc_emb, att_mem, p_att_mem, v_fusion or None, aux_logits dict)
        """
        B = fc_feats.size(0)
        assert B % rep == 0, 'batch rows must be a multiple of rep'
        fc_emb, att_mem = self._prepare_feature(fc_feats, att_feats, None)   # [B,rnn], [B,M,rnn]
        p_att_mem = self.p_att_emd(att_mem)                                   # [B,M,att_hid]

        v_fusion, aux = None, {}
        if self.num_modal > 0:
            v = att_feats[::rep]                     # [B/rep, M, dv]  raw region features
            fg = fc_feats[::rep]                     # [B/rep, dv]     global feature Fg (mean-pooled)
            r = self.region_proj(v)                  # Eq. (11)        [B/rep, M, da]
            vs = []
            if self.use_geo:
                if boxes is None:
                    raise ValueError('use_geo=1 requires region boxes (--input_box_dir)')
                vs.append(self.geo(v, r, boxes[::rep]))
            if self.use_attr:
                v_attr, attr_logits = self.attr(fg, v, r)
                vs.append(v_attr)
                aux['attr_logits'] = attr_logits
            if self.use_scene:
                s = scene_feats[::rep] if scene_feats is not None else fg   # fallback: global region feature
                v_scene, scene_logits = self.scene(s, v, r)
                vs.append(v_scene)
                aux['scene_logits'] = scene_logits
            v_fusion = self.fusion_drop(expand_rep(self.fusion(vs), rep))    # [B, df]
        return fc_emb, att_mem, p_att_mem, v_fusion, aux

    def _forward(self, fc_feats, att_feats, seq, seq_mask, boxes=None, scene_feats=None,
                 attr_labels=None, scene_labels=None, rep=1):
        att_feats, att_masks = self.clip_att(att_feats, None)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)   # [BS, 17, V+1]
        fc_emb, att_mem, p_att_mem, v_fusion, aux = self._encode(fc_feats, att_feats, boxes, scene_feats, rep)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()

            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_emb, att_mem, p_att_mem, v_fusion, state)
            outputs[:, i] = output

        word_loss = self.crit(outputs, seq[:, 1:], seq_mask[:, 1:])

        # [GAS-Net] auxiliary multi-label BCE losses (Eq. 21 and Eq. 41).  They are only used in
        # the XE stage; during SCST the model is called through _sample / _scst_forward instead.
        attr_loss = torch.zeros_like(word_loss)
        scene_loss = torch.zeros_like(word_loss)
        if 'attr_logits' in aux and attr_labels is not None:
            # Eq. (21): mean over the L attributes (and over the batch)
            attr_loss = F.binary_cross_entropy_with_logits(aux['attr_logits'], attr_labels[::rep].float())
        if 'scene_logits' in aux and scene_labels is not None:
            tgt = scene_labels[::rep].float()
            # Eq. (41): sum over the K scene classes, averaged over the batch
            scene_loss = F.binary_cross_entropy_with_logits(aux['scene_logits'], tgt) * tgt.size(1)

        # [1,3] per replica -> [num_gpu,3] after DataParallel gather: (word, attr, scene)
        return torch.stack([word_loss, attr_loss, scene_loss]).unsqueeze(0)

    def get_logprobs_state(self, it, fc_feats, att_feats_mem, p_att_feats_mem, v_fusion, state):
        xt = self.embed(it)  # [BS, input_encoding_size]   # 'it' contains a word index
        output, state = self.lstm_core(xt, fc_feats, att_feats_mem, p_att_feats_mem, v_fusion, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)  # [BS, vocab_size+1]
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, boxes=None, scene_feats=None, rep=1):
        # the beam size is now self.beam_size (opt.beam_size); CaptionModel.beam_search takes it as an argument
        beam_size = self.beam_size
        batch_size = fc_feats.size(0)
        fc_emb, att_mem, p_att_mem, v_fusion, _ = self._encode(fc_feats, att_feats, boxes, scene_feats, rep)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]

        # lets process every image independently for now, for simplicity
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc = fc_emb[k:k + 1].expand(beam_size, fc_emb.size(1))
            tmp_att = att_mem[k:k + 1].expand(*((beam_size,) + att_mem.size()[1:])).contiguous()
            tmp_p_att = p_att_mem[k:k + 1].expand(*((beam_size,) + p_att_mem.size()[1:])).contiguous()
            tmp_vf = None
            if v_fusion is not None:
                tmp_vf = v_fusion[k:k + 1].expand(beam_size, v_fusion.size(1)).contiguous()

            it = fc_emb.new_zeros([beam_size], dtype=torch.long)   # <bos>
            logprobs, state = self.get_logprobs_state(it, tmp_fc, tmp_att, tmp_p_att, tmp_vf, state)

            # extra positional args are chunked / forwarded to get_logprobs_state by CaptionModel.beam_search
            self.done_beams[k] = self.beam_search(beam_size, state, logprobs, tmp_fc, tmp_att, tmp_p_att, tmp_vf)
            seq[:, k] = self.done_beams[k][0]['seq']   # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1).cuda()

    def _sample(self, fc_feats, att_feats, boxes=None, scene_feats=None, sm=1, opt={}, rep=1):
        att_feats, att_masks = self.clip_att(att_feats, None)
        sample_max = sm
        temperature = 1.0
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        fc_emb, att_mem, p_att_mem, v_fusion, _ = self._encode(fc_feats, att_feats, boxes, scene_feats, rep)
        seq = fc_emb.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_emb.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_emb.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_emb, att_mem, p_att_mem, v_fusion, state)

        return seq, seqLogprobs

    def _scst_forward(self, sample_logprobs, gen_result, reward):
        word_loss = self.scst_crit(sample_logprobs, gen_result, reward)
        return word_loss.unsqueeze(0)


class TopDownCore_GASNet(nn.Module):
    """Two-layer Up-Down decoder.  [GAS-Net] v_fusion is concatenated to the input of the
    attention LSTM (Eq. 63) and of the language LSTM (Eq. 68)."""

    def __init__(self, opt, fusion_size=0, use_maxout=False):
        super(TopDownCore_GASNet, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.fusion_size = fusion_size
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2 + fusion_size, opt.rnn_size)  # we, fc, h^2_t-1, v_fusion
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2 + fusion_size, opt.rnn_size)                           # h^1_t, \hat v, v_fusion
        self.attention = Attention(opt)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p, 0)
        return

    def forward(self, xt, fc_feats, att_feats, p_att_feats, v_fusion, state):
        prev_h = state[0][-1]
        att_in = [self.dropout(prev_h), fc_feats, xt]
        if v_fusion is not None:
            att_in.append(v_fusion)
        h_att, c_att = self.att_lstm(torch.cat(att_in, 1), (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats, None)   # \hat v_t (Eq. 65-67)
        lang_in = [att, self.dropout(h_att)]
        if v_fusion is not None:
            lang_in.append(v_fusion)
        h_lang, c_lang = self.lang_lstm(torch.cat(lang_in, 1), (state[0][1], state[1][1]))
        output = self.dropout(h_lang)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, state


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p, 0)
        return

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)   # 36
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)

        weight = F.softmax(dot, dim=1)

        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res


class LSTM_GASNet(AttModel_GASNet):
    def __init__(self, opt):
        super(LSTM_GASNet, self).__init__(opt)
        self.num_layers = 2


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class RewardCriterion(nn.Module):
    """SCST loss, Eq. (73): -(r(sample) - r(greedy)) * sum_t log p(y_t)."""

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()

        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)

        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, input, target, mask):  # input: [BS, 17, vocab_size+1]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        self.size = input.size(1)
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
