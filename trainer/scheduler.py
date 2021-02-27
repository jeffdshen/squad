"""Learning rate schedulers.

Author:
    Jeffrey Shen
"""


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
    """Uses a power function a * x^power + b, such that it equals the start_lr at 0
    and the end_lr at max_num_steps. Afterwards, returns the end_lr forever.
    Automatically steps the first step if take_initial_step is True.
    """

    def __init__(
        self,
        optimizer,
        start_lr,
        end_lr,
        max_num_steps,
        power=1,
        take_initial_step=True,
    ):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.max_num_steps = max_num_steps
        self.power = power

        self.step_count = 0
        self.last_lr = start_lr
        self.multiplier = (end_lr - start_lr) / (max_num_steps ** power)

        if take_initial_step:
            self.step()

    def get_last_lr(self):
        return self.last_lr

    def get_lr(self):
        if self.step_count >= self.max_num_steps:
            return self.end_lr

        return self.multiplier * (self.step_count ** self.power) + self.start_lr

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.last_lr = lr
        if self.step_count >= self.max_num_steps:
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
            if self.index < len(self.schedulers):
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
    ):
        schedulers = [
            PowerScheduler(
                optimizer,
                start_lr,
                peak_lr,
                warmup_steps,
                power=1,
                take_initial_step=take_initial_step,
            ),
            PowerScheduler(
                optimizer,
                peak_lr,
                end_lr,
                max_num_steps - warmup_steps,
                power=power,
                take_initial_step=False,
            ),
        ]

        super().__init__(optimizer, schedulers)
