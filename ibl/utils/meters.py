from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Reset the state.

        Args:
            self: (todo): write your description
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the value

        Args:
            self: (todo): write your description
            val: (float): write your description
            n: (array): write your description
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count