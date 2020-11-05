from __future__ import absolute_import
import os
import errno


def mkdir_if_missing(dir_path):
    """
    Make a directory if the given.

    Args:
        dir_path: (str): write your description
    """
    if not dir_path: return
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
