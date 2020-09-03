import os.path as osp

from ..utils.data import Dataset
from ..utils.serialization import write_json
from ..utils.dist_utils import synchronize


class Demo(Dataset):
    """
    examples/data
    └── demo
        ├── raw/
        ├── meta.json
        └── splits.json

    Inputs:
        root (str): the path to demo_dataset
        verbose (bool): print flag, default=True
    """

    def __init__(self, root, scale=None, verbose=True):
        super(Demo, self).__init__(root)

        self.arrange()
        self.load(verbose)

    def arrange(self):
        if self._check_integrity():
            return

        try:
            rank = dist.get_rank()
        except:
            rank = 0

        # the root path for raw dataset
        raw_dir = osp.join(self.root, 'raw')
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")

        """
        TODO add the following variables:

            1. identities: List[List[str,],], str is the relative path for each image
                        e.g. [['demo/query/1_1.jpg', 'demo/query/1_2.jpg'],
                              ['demo/query/2_1.jpg', 'demo/query/2_2.jpg']]

                        Note that the images in each sub list should belong to the same location/coordinates,
                        e.g. 'demo/query/1_1.jpg' and 'demo/query/1_2.jpg' come from the same location/coordinates.

                        Also note the `raw_dir` should be excluded from image path,
                        e.g. the absolute path for 'demo/query/1_1.jpg' is `osp.join(raw_dir, 'demo/query/1_1.jpg')`

            2. utms: List[[float, float],], [float, float] is [abscissa, ordinate] in world coordinates,
                    e.g. [[585089.3603214071, 4477427.575588894],
                          [585085.8930572948, 4477435.629250713]]

                    Note that identities and utms should be consistent to each other, which means that
                        in the above example, the coordinates for 'demo/query/1_1.jpg' and 'demo/query/1_2.jpg'
                        are [585089.3603214071, 4477427.575588894].

            3. q_train_pids: List[int,], int is the indexes of queries for training,
                        e.g. if q_train_pids = [0,],
                            image paths in identities[0] (['demo/query/1_1.jpg', 'demo/query/1_2.jpg']) are query paths for training,
                            and their coordinates are utms[0].

            4. db_train_pids: List[int,], int is the indexes of galleries for training
            5. q_val_pids: List[int,], int is the indexes of queries for validation
            6. db_val_pids: List[int,], int is the indexes of galleries for validation
            7. q_test_pids: List[int,], int is the indexes of queries for testing
            8. db_test_pids: List[int,], int is the indexes of galleries for testing

            Note that in our setup, query and gallery cannot share the same coordinates (utms).
            Also, train/val/test splits cannot share the same coordinates (utms).
            The reason is that, in real-world applications, probes and matching images would not be captured at exactly identical locations.

        """

        # Save meta information into a json file
        meta = {
                'name': 'demo', # change it to your dataset name
                'identities': identities,
                'utm': utms
                }

        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the training / test / val split into a json file
        splits = {
            'q_train': sorted(q_train_pids),
            'db_train': sorted(db_train_pids),
            'q_val': sorted(q_val_pids),
            'db_val': sorted(db_val_pids),
            'q_test': sorted(q_test_pids),
            'db_test': sorted(db_test_pids)}

        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits.json'))

        synchronize()
