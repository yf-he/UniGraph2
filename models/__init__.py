from models.unigraph2 import UniGraph2
from gnn_modules import Supervised_gnn_classification
import torch


def build_model(args):
    if args.model == 'unigraph2':
        return UniGraph2(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            num_dec_layers=args.num_dec_layers,
            num_remasking=args.num_remasking,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            mask_rate=args.mask_rate,
            remask_rate=args.remask_rate,
            mask_method=args.mask_method,
            norm=args.norm,
            loss_fn=args.loss_fn,
            drop_edge_rate=args.drop_edge_rate,
            alpha_l=args.alpha_l,
            lam=args.lam,
            delayed_ema_epoch=args.delayed_ema_epoch,
            replace_rate=args.replace_rate,
            remask_method=args.remask_method,
            momentum=args.momentum,
            zero_init=args.dataset in ("cora", "pubmed", "citeseer"),
            top_k=args.top_k,
            hhsize_time=args.hiddenhidden_size_times,
            num_expert=args.num_expert,
            moe=args.moe,
            moe_use_linear=args.moe_use_linear,
            decoder_no_moe=args.decoder_no_moe, 
            moe_layer=args.moe_layer,
            deepspeed=args.deepspeed
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")


