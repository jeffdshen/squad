"""Learning rate schedulers.

Author:
    Jeffrey Shen
"""

import torch.optim.lr_scheduler as sched


def get_linear_warmup_power_decay_scheduler(
    optimizer, warmup_steps, max_num_steps, end_multiplier=0.0, power=1
):
    """Uses a power function a * x^power + b, such that it equals the 1.0 at start_step=1
    and the end_multiplier at end_step. Afterwards, returns the end_multiplier forever.
    For the first warmup_steps, linearly increase the learning rate until it hits the power
    learning rate.
    """

    # a = end_lr - start_lr / (end_step ** power - start_step ** power)
    start_multiplier = 1.0
    start_step = 1
    scale = (end_multiplier - start_multiplier) / (max_num_steps**power - start_step**power)
    # b = start_lr - scale * start_step ** power
    constant = start_multiplier - scale * (start_step**power)

    def lr_lambda(step):
        step = start_step + step
        if step < warmup_steps:
            warmup_multiplier = scale * (warmup_steps ** power) + constant
            return float(step) / float(max(1, warmup_steps)) * warmup_multiplier
        elif step >= max_num_steps:
            return end_multiplier
        else:
            return scale * (step ** power) + constant

    return sched.LambdaLR(optimizer, lr_lambda)



class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class PowerScheduler(LRScheduler):
    """Uses a power function a * x^power + b, such that it equals the start_lr at start_step
    and the end_lr at end_step. Afterwards, returns the end_lr forever.
    Automatically steps the first step if take_initial_step is True.
    If take_last_step is True, step returns True only after end_step.
    """

    def __init__(
        self,
        optimizer,
        start_lr,
        end_lr,
        start_step,
        end_step,
        power=1,
        take_initial_step=True,
        take_last_step=True,
    ):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.end_step = end_step if end_step > 0 else float("inf")
        self.power = power

        self.step_count = start_step - 1
        self.last_lr = start_lr
        self.multiplier = (end_lr - start_lr) / (self.end_step ** power - start_step ** power)
        self.constant = start_lr - self.multiplier * (start_step ** power)
        self.take_last_step = take_last_step

        if take_initial_step:
            self.step()

    def get_last_lr(self):
        return self.last_lr

    def get_lr(self):
        if self.step_count >= self.end_step:
            return self.end_lr

        return self.multiplier * (self.step_count ** self.power) + self.constant

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.last_lr = lr

    
        if self.step_count >= self.end_step - (not self.take_last_step):
            return True
        return False


class ConcatScheduler(LRScheduler):
    """Concats schedulers together."""

    def __init__(self, optimizer, schedulers):
        super().__init__(optimizer)
        self.schedulers = schedulers
        self.index = 0
        self.last_lr = schedulers[0].get_last_lr()

    def state_dict(self):
        return {
            "index": self.index,
            "last_lr": self.last_lr,
            "schedulers": [s.state_dict() for s in self.schedulers],
        }

    def load_state_dict(self, state_dict):
        self.index = state_dict["index"]
        self.last_lr = state_dict["last_lr"]
        for s, sd in zip(self.schedulers, state_dict["schedulers"]):
            s.load_state_dict(sd)

    def get_last_lr(self):
        return self.last_lr

    def step(self):
        done = self.schedulers[self.index].step()
        self.last_lr = self.schedulers[self.index].get_last_lr()
        if done:
            if self.index + 1 < len(self.schedulers):
                self.index += 1
            else:
                return True

        return False


class LinearWarmupPowerDecay(ConcatScheduler):
    def __init__(
        _,
        optimizer,
        start_lr,
        peak_lr,
        end_lr,
        warmup_steps,
        max_num_steps,
        power=1,
        take_initial_step=True,
        take_last_step=True
    ):
        schedulers = [
            PowerScheduler(
                optimizer,
                start_lr,
                peak_lr,
                1,
                warmup_steps,
                power=1,
                take_initial_step=take_initial_step,
                take_last_step=False,
            ),
            PowerScheduler(
                optimizer,
                peak_lr,
                end_lr,
                warmup_steps,
                max_num_steps,
                power=power,
                take_initial_step=False,
                take_last_step=take_last_step,
            ),
        ]

        super().__init__(optimizer, schedulers)
