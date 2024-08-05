from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset
import random



class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None, good_fraction = 0.1):

        self.transform = transform
        self.good_fraction = good_fraction


        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue

            sections = np.where(np.diff(data['lig_mask']))[0] + 1 \
                if 'lig' in k \
                else np.where(np.diff(data['pocket_mask']))[0] + 1
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                # print("Got here: ")
                self.data['num_lig_atoms'] = \
                    torch.tensor([len(x) for x in self.data['lig_mask']])
                # print(self.data['num_lig_atoms'].size())
                # raise ValueError("Done")
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return int(self.good_fraction*len(self.data['names']))

    def __getitem__(self, idx):
        # Get a random bad index which can't be a good index
        bad_idx = int(random.uniform(self.good_fraction, 1)*len(self.data['names']))

        # Get good data from idx
        good_data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            good_data = self.transform(good_data)

        # Get bad data from bad_idx
        bad_data = {key: val[bad_idx] for key, val in self.data.items()}
        if self.transform is not None:
            bad_data = self.transform(bad_data)
        
        return [good_data, bad_data]

    @staticmethod
    def collate_fn(batch):

        # Convert: [[good1, bad1], [good2, bad2]...] -> [good1, good2, .... bad1, bad2, ...]
        result = [item[0] for item in batch] + [item[1] for item in batch]

        out = {}
        for prop in result[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in result]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                # print("Got here: ")
                out[prop] = torch.tensor([x[prop] for x in result])
                
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(result)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in result], dim=0)

        return out
