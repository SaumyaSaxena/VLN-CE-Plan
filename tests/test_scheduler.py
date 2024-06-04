import torch
from timm.scheduler.scheduler_factory import create_scheduler

from matplotlib import pyplot as plt

from timm import create_model 
from timm.optim import create_optimizer
from types import SimpleNamespace


def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch)[0])
    return lr_per_epoch

if __name__== "__main__":

    adamw_params = {
        'lr': 2.5e-4,
        'eps': 1.0e-8,
        'weight_decay': 0.01,
    }

    model = create_model('resnet34')

    args = SimpleNamespace()
    args.weight_decay = 0.01
    args.lr = 1e-4
    args.opt = 'adam' 
    args.momentum = 0.9

    optimizer = create_optimizer(args, model)

    scheduler_config = SimpleNamespace()
    scheduler_config.epochs = 10
    scheduler_config.cooldown_epochs = 5
    scheduler_config.sched = 'cosine' 
    scheduler_config.min_lr = 1.0e-5
    scheduler_config.warmup_lr = 1.0e-6
    scheduler_config.warmup_epochs = 1

    scheduler, num_epochs = create_scheduler(scheduler_config, optimizer)

    sch_lr = []
    optim_lr = []
    for e in range(20):
        for s in range(num_epochs):
            sch_lr.append(scheduler.get_epoch_values(s)[0])
            optim_lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step(s)
        scheduler.step(s)

    plt.plot([i for i in range(len(sch_lr))], sch_lr, label="Without warmup", alpha=0.8)
    plt.savefig(f"/home/saumyas/Projects/VLN-CE-Plan/tests/media/Scheduler_lr.jpg")
    plt.close()

    plt.plot([i for i in range(len(optim_lr))], optim_lr, label="Without warmup", alpha=0.8)
    plt.savefig(f"/home/saumyas/Projects/VLN-CE-Plan/tests/media/Optim_lr.jpg")
    plt.close()
    
    lr_per_epoch = get_lr_per_epoch(scheduler, num_epochs)
    plt.plot([i for i in range(num_epochs)], lr_per_epoch, label="Without warmup", alpha=0.8)
    plt.savefig(f"/home/saumyas/Projects/VLN-CE-Plan/tests/media/timm_Scheduler.jpg")
    plt.close()
    import ipdb; ipdb.set_trace()
