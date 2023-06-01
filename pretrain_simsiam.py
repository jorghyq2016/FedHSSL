"""
part of the code adopted from https://github.com/lightly-ai/lightly
"""
import torch.nn as nn
import torch.utils
import torch

from models.ssl_utils import NegCosineSimilarityLoss, projection_mlp, prediction_mlp
from pretrain_template import ClientTemplate


def get_clients(encoder_local_bottom_list, encoder_local_list, encoder_cross_list, args):
    client_list = []
    for i in range(args.k):
        client = SSClient(i, [encoder_local_bottom_list[i], encoder_local_list[i], encoder_cross_list[i]], args)
        client_list.append(client)
    return client_list


class SSClient(ClientTemplate):

    def __init__(self, client_idx, models, args):
        super(SSClient, self).__init__(client_idx, models, args)

        self.projection_mlp_cross = projection_mlp(args.num_ftrs, self.proj_hidden_dim, self.out_dim,
                                                         self.num_mlp_layers).to(args.device)
        self.prediction_mlp_cross = prediction_mlp(self.out_dim, self.pred_hidden_dim, self.out_dim).\
            to(args.device)
        self.projection_mlp_local = projection_mlp(args.num_ftrs, self.proj_hidden_dim, self.out_dim,
                                                         self.num_mlp_layers).to(args.device)
        self.prediction_mlp_local = prediction_mlp(self.out_dim, self.pred_hidden_dim, self.out_dim).\
            to(args.device)

        # loss
        self.cross_criterion = NegCosineSimilarityLoss().to(args.device)
        self.local_criterion = NegCosineSimilarityLoss().to(args.device)

        # models for aggregation
        self.model_local_top = nn.ModuleList(
            [self.encoder_local_top, self.projection_mlp_local, self.prediction_mlp_local])

        # models for batch operation
        self.models = nn.ModuleList([self.encoder_cross, self.projection_mlp_cross, self.prediction_mlp_cross,
                                     self.encoder_local_bottom, self.encoder_local_top, self.projection_mlp_local,
                                     self.prediction_mlp_local])

        # rescale learning rate
        self.pretrain_lr_ratio = 0.5/(self.args.local_ratio + self.args.constraint_ratio)

        # get optimizer
        self.optimizer_list_cross = [self.get_optimizer(self.encoder_cross, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_cross),
                                     self.get_optimizer(self.prediction_mlp_cross)]

        self.optimizer_list_local = [self.get_optimizer(self.encoder_local_bottom, self.args.optimizer),
                                     self.get_optimizer(self.encoder_local_top, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_local),
                                     self.get_optimizer(self.prediction_mlp_local)]

        # learning rate scheduler
        self.scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                                  for optimizer in self.optimizer_list_cross[:-1] if optimizer is not None] + \
                              [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                               for optimizer in self.optimizer_list_local[:-1] if optimizer is not None]

        self.model_to_device(args.device)

    def compute_cross_loss(self, x, y, z_cross_own, z_cross_received, epoch):
        meta = {}
        p_cross_own = self.prediction_mlp_cross(z_cross_own)
        loss_debug = []
        if isinstance(z_cross_received, list):
            loss = torch.tensor(0)
            for z_item in z_cross_received:
                loss_ind = self.cross_criterion(p_cross_own, z_item.detach())
                loss_debug.append(loss_ind.item())
                loss = loss + loss_ind
            loss = loss / len(z_cross_received)
        else:
            loss = self.cross_criterion(p_cross_own, z_cross_received.detach())
        meta['loss_debug'] = loss_debug
        return loss, meta

    def compute_local_loss(self, x, y, epoch=0, debug=False):
        meta = {}
        loss_constraint = torch.tensor(0).to(self.args.device)

        x1 = x[0].float().to(self.args.device)
        x2 = x[1].float().to(self.args.device)

        # model outputs of two augemented views
        f1 = self.encoder_local_bottom(x1)
        h1 = self.encoder_local_top(f1).flatten(start_dim=1)
        z1 = self.projection_mlp_local(h1)
        p1 = self.prediction_mlp_local(z1)

        f2 = self.encoder_local_bottom(x2)
        h2 = self.encoder_local_top(f2).flatten(start_dim=1)
        z2 = self.projection_mlp_local(h2)
        p2 = self.prediction_mlp_local(z2)

        # local contrastive loss using augmented views
        loss = self.args.local_ratio * (self.local_criterion(p1, z2.detach()) + self.local_criterion(p2, z1.detach()))

        # regularization loss of cross encoder
        if self.args.constraint_ratio > 0:
            h1_cross = self.encoder_cross(x1).flatten(start_dim=1)
            h2_cross = self.encoder_cross(x2).flatten(start_dim=1)
            z1_cross = self.projection_mlp_cross(h1_cross)
            z2_cross = self.projection_mlp_cross(h2_cross)

            loss_constraint = self.args.constraint_ratio * (self.local_criterion(p1, z1_cross.detach()) +
                                                            self.local_criterion(p2, z2_cross.detach()))
            loss = loss + loss_constraint
        meta['loss_debug'] = loss_constraint.item()
        return loss, meta
