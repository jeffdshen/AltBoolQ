import numpy as np


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        self.count += count
        self.sum += avg * count
        self.avg = self.sum / self.count


class EMAMeter:
    def __init__(self, weight):
        self.avg = 0
        self.sum = 0
        self.inv_count = 1.0
        self.weight = weight

    def reset(self):
        self.__init__(self.weight)

    def add(self, avg, count=1):
        batch_inv_count = self.weight**count
        self.sum = self.sum * batch_inv_count + (1 - batch_inv_count) * avg
        self.inv_count = self.inv_count * batch_inv_count
        self.avg = self.sum / (1 - self.inv_count)


class MinMeter:
    def __init__(self):
        self.min = float("inf")

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        if avg < self.min:
            self.min = avg
            return True

        return False


class MaxMeter:
    def __init__(self):
        self.max = float("-inf")

    def reset(self):
        self.__init__()

    def add(self, avg, count=1):
        if avg > self.max:
            self.max = avg
            return True

        return False


class AccMeter:
    def __init__(self):
        self.scores = np.zeros((2, 2))
        self.acc = 0.0

    def reset(self):
        self.__init__()

    def add(self, scores, count=1):
        self.scores += scores
        total = self.scores.sum()
        self.acc = (self.scores[0, 0] + self.scores[1, 1]) / total


class AccEMAMeter:
    def __init__(self, weight):
        self.weight = weight
        self.epoch = round(1.0 / (1.0 - weight))
        # approximately 1 / e
        self.epoch_weight = 0.36603234127322950
        self.count = 0
        self.total_inv_weight = 1.0
        self.acc_meter = AccMeter()
        self.acc = self.acc_meter.acc

    def reset(self):
        self.__init__(self.weight)

    def add(self, scores, count=1):
        self.acc_meter.add(scores, count)
        self.count += count
        if self.count > self.epoch:
            acc = self.acc_meter.acc
            self.acc_meter.reset()

            self.count = 0
            self.total_inv_weight *= self.epoch_weight
            self.acc = (self.acc * self.epoch_weight + acc * (1 - self.epoch_weight)) / (
                1 - self.total_inv_weight
            )
