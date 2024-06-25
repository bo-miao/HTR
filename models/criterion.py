import torch
import torch.nn.functional as F
from torch import nn


from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)
from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha

        self.vos_loss = BootstrappedCE()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')
        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            pass
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(target_masks).float()
        im_h, im_w = target_masks.shape[-2:]

        # upsample predictions to the target size
        vos_target_masks = target_masks.clone()  # [b, t, h, w]
        vos_target_masks = vos_target_masks[:, 1:]  # [b, t-1, h, w]

        # main mask loss
        src_masks = F.interpolate(src_masks, size=(im_h, im_w), mode='bilinear', align_corners=False)
        assert src_masks.size(2) == vos_target_masks.size(2) == target_masks.size(2), 'size error.'
        src_masks = src_masks.flatten(1)  # [b, thw]
        target_masks = target_masks.flatten(1)  # [b, thw]
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

        # VOS head mask (main loss only)
        if 'vos_logits' in outputs:
            vos_logits = outputs['vos_logits']  # b t-1 2 h w
            vos_prob = outputs['vos_prob']  # b t-1 1 h w
            vos_target_masks = vos_target_masks  # [b, t-1, h, w]

            b, t = vos_logits.shape[:2]
            loss_vos = 0.
            assert vos_target_masks.shape[1] == vos_logits.shape[1], "frame num mismatch."
            for j in range(t):
                loss_vos_single = 0.
                for i in range(b):
                    # b o h w & b h w
                    l = self.vos_loss(vos_logits[i:i+1, j], vos_target_masks[i:i+1, j].long(), outputs['run_ratio'])
                    loss_vos_single += l / b

                # b o-1 h w & b h w
                l2 = vos_dice_loss(vos_prob[:, j], vos_target_masks[:, j].long())
                loss_vos_single += l2

                loss_vos += loss_vos_single

            losses["loss_vos"] = loss_vos

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # use all results for matching to find best one. use that query to compute loss.
        indices = outputs["main_matcher_index"]
        aux_indices = outputs["aux_matcher_index"]

        # Compute the average number of target boxes [I think is frame?] accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T] tensor([1, 1, 0, 0, 0]
        num_boxes = target_valid.sum().item()
        device = outputs['pred_masks'].device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():  # True
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  # 5

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer. change outputs->aux_outputs
        if 'aux_outputs' in outputs:
            # for each aux output
            assert len(aux_indices) == len(outputs['aux_outputs']), "Aux index len not match."
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = aux_indices[i]
                # for each type of loss
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=0.2, end_warm=0.6, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target) #, 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean() #, this_p


def vos_dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()
