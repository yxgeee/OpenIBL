from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing


class Logger(object):
    def __init__(self, fpath=None):
        """
        Initialize a file.

        Args:
            self: (todo): write your description
            fpath: (str): write your description
        """
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        """
        Closes the stream.

        Args:
            self: (todo): write your description
        """
        self.close()

    def __enter__(self):
        """
        Enter the callable

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args):
        """
        Exit the exit.

        Args:
            self: (todo): write your description
        """
        self.close()

    def write(self, msg):
        """
        Write a message to the console.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        """
        Flush the file.

        Args:
            self: (todo): write your description
        """
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """
        Close the console.

        Args:
            self: (todo): write your description
        """
        self.console.close()
        if self.file is not None:
            self.file.close()
