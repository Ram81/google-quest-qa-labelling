from torch.optim.lr_scheduler import LambdaLR


def linear_schedule_warmup(optimizer, num_training_steps=1000, warmup_prop=0, min_factor=0, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_training_steps * warmup_prop:
            return min_factor + (float(current_step) / float(max(1, num_training_steps * warmup_prop))) * (1 - min_factor) 
        return max(min_factor, min_factor +  (1 - min_factor) * 
                   (float(num_training_steps - current_step) / 
                    float(max(1, (1 - warmup_prop) * num_training_steps))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(epoch):
    if epoch < 20:
        return 5e-4
    else:
        return 5e-5


def schedule_lr(optimizer, epoch, scheduler, scheduler_name='', avg_val_loss=0, epochs=100, 
                warmup_prop=0, lr_init=1e-3, min_lr=1e-6, verbose_eval=1):

    if epoch:
        if epoch <= epochs * warmup_prop and warmup_prop > 0:
            lr = min_lr + (lr_init - min_lr) * epoch / (epochs * warmup_prop)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        else:
            if scheduler_name == 'cosine':
                scheduler.step()
            elif scheduler_name == 'reduce_lr':
                if (epoch + 1) % verbose_eval == 1:
                    scheduler.step(avg_val_loss)
            else:  # Manual scheduling
                lr = get_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    lr = optimizer.param_groups[-1]['lr']

    return lr