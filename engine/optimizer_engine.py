from utils import optimizer

def build_optimizer(args, param_groups):
    params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ]
    if len(param_groups) == 5:
        params.append(
            {
               "params": param_groups[4],
               "lr": args.lr * 1.0,
               "weight_decay": args.wt_decay,
            },
        )
    optim = getattr(optimizer, args.optimizer)(
        params=params,
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    return optim