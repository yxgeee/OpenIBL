from __future__ import print_function, absolute_import
import os.path as osp
from collections import namedtuple
import torch.distributed as dist

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json, read_mat
from ..utils.dist_utils import synchronize

def parse_dbStruct(path):
    matStruct = read_mat(path)
    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T
    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T
    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    dbStruct = namedtuple('dbStruct',
            ['dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ'])
    return dbStruct(dbImage, utmDb, qImage, utmQ, numDb, numQ)

class Pittsburgh(Dataset):

    def __init__(self, root, scale='250k', verbose=True):
        super(Pittsburgh, self).__init__(root)
        self.scale = scale

        self.arrange()
        self.load(verbose, scale)

    def arrange(self):
        if self._check_integrity(self.scale):
            return

        raw_dir = osp.join(self.root, 'raw')
        if (not osp.isdir(raw_dir)):
            raise RuntimeError("Dataset not found.")
        db_root = osp.join('Pittsburgh', 'images')
        q_root = osp.join('Pittsburgh', 'queries')

        identities = []
        utms = []
        q_pids, db_pids = {}, {}
        def register(split):
            struct = parse_dbStruct(osp.join(raw_dir, 'pitts'+self.scale+'_'+split+'.mat'))
            q_ids = []
            for fpath, utm in zip(struct.qImage, struct.utmQ):
                sid = fpath.split('_')[0]
                if (sid not in q_pids.keys()):
                    pid = len(identities)
                    q_pids[sid] = pid
                    identities.append([])
                    utms.append(utm.tolist())
                    q_ids.append(pid)
                identities[q_pids[sid]].append(osp.join(q_root, fpath))
                assert(utms[q_pids[sid]]==utm.tolist())
            db_ids = []
            for fpath, utm in zip(struct.dbImage, struct.utmDb):
                sid = fpath.split('_')[0]
                if (sid not in db_pids.keys()):
                    pid = len(identities)
                    db_pids[sid] = pid
                    identities.append([])
                    utms.append(utm.tolist())
                    db_ids.append(pid)
                identities[db_pids[sid]].append(osp.join(db_root, fpath))
                assert(utms[db_pids[sid]]==utm.tolist())
            return q_ids, db_ids

        q_train_pids, db_train_pids = register('train')
        # train_pids = q_train_pids + db_train_pids
        q_val_pids, db_val_pids = register('val')
        q_test_pids, db_test_pids = register('test')
        assert len(identities)==len(utms)

        # for pid in q_test_pids:
        #     if (len(identities[pid])!=24):
        #         print (identities[pid])

        # Save meta information into a json file
        meta = {'name': 'Pittsburgh_'+self.scale,
                'identities': identities, 'utm': utms}
        try:
            rank = dist.get_rank()
        except:
            rank = 0
        if rank == 0:
            write_json(meta, osp.join(self.root, 'meta_'+self.scale+'.json'))

        # Save the training / test split
        splits = {
            # 'train': sorted(train_pids),
            'q_train': sorted(q_train_pids),
            'db_train': sorted(db_train_pids),
            'q_val': sorted(q_val_pids),
            'db_val': sorted(db_val_pids),
            'q_test': sorted(q_test_pids),
            'db_test': sorted(db_test_pids)}
        if rank == 0:
            write_json(splits, osp.join(self.root, 'splits_'+self.scale+'.json'))
        synchronize()
