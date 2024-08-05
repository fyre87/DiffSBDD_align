from pathlib import Path
from time import time
import argparse
import shutil
import random

import matplotlib.pyplot as plt
# import seaborn as sns

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from rdkit import Chem
from scipy.ndimage import gaussian_filter

import torch

from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params


def process_ligand_and_pocket(pdbfile, ligand,
                              atom_dict, dist_cutoff, ca_only, amino_acid_dict):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    # given a molecule and pdbfile, rather than a sdffile and a pdbfile

    # try:
    #     ligand = Chem.SDMolSupplier(str(sdffile))[0]
    # except:
    #     raise Exception(f'cannot read sdf mol ({sdffile})')

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict or a.element != 'H')]
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
            for a in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(
            f'{e} not in atom dict')

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(
                    -1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }
    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                                                     amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    else:
        full_atoms = np.concatenate(
            [np.array([atom.element for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        full_coords = np.concatenate(
            [np.array([atom.coord for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        try:
            pocket_one_hot = []
            for a in full_atoms:
                if a in amino_acid_dict:
                    atom = np.eye(1, len(amino_acid_dict),
                                  amino_acid_dict[a.capitalize()]).squeeze()
                elif a != 'H':
                    atom = np.eye(1, len(amino_acid_dict),
                                  len(amino_acid_dict)).squeeze()
                pocket_one_hot.append(atom)
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
                f'{e} not in atom dict ({pdbfile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    return ligand_data, pocket_data


def compute_smiles(positions, one_hot, mask):
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)),
                total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)
        mol = rdmol_to_smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f'bonds{i + 1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                if a1 == 'others' or a2 == 'others':
                    bond_len = 0
                else:
                    # Replace missing values with sum of average covalent radii
                    bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(filename, pdb_and_mol_ids, lig_coords, lig_one_hot, lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask):
    np.savez(filename,
             names=pdb_and_mol_ids,
             lig_coords=lig_coords,
             lig_one_hot=lig_one_hot,
             lig_mask=lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask
             )
    return True

def process_all_ligands_and_pockets(pdbfile, sdffile, csv_file):
    # Process and order them!
    ligands = Chem.SDMolSupplier(str(sdffile))

    ligands_list = [mol for mol in ligands if mol is not None]

    # print(ligands_list[0])
    # print(ligands_list)
    # print(type(ligands_list))

    # Just take the QED for now
    ##** Adjust formula later!!!
    scores = np.genfromtxt(csv_file, delimiter=',', skip_header=1, usecols=1) #usecols=(0,1))
    ordered_scores_ids = np.argsort(scores)[::-1]

    # Sort the ligands in order of the ordered_scores
    sorted_ligands = [ligands_list[i] for i in ordered_scores_ids]

    ##** Note: We use the crossdock full dataset rather than the crossdock dataset cuz that's what we processed originally
    # Could use either
    dataset_info = dataset_params['crossdock']

    ##** Chaning this from original sampling!
    # amino_acid_dict = dataset_info['aa_encoder']
    amino_acid_dict = dataset_info['atom_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    lig_coords = []
    lig_one_hot = []
    lig_mask = []
    pocket_coords = []
    pocket_one_hot = []
    pocket_mask = []
    pdb_and_mol_ids = []
    count_protein = []
    count_ligand = []
    count_total = []
    count = 0

    for i in range(0, len(sorted_ligands)):
        ##** Using default values for cutoff and ca_only
        ##** Using "False" for ca_only
        ##** Using 8.0 for dist_cutoff
        try:
            ligand_data, pocket_data = process_ligand_and_pocket(pdbfile, sorted_ligands[i], atom_dict=atom_dict, dist_cutoff=8.0,
                    ca_only=False, amino_acid_dict=amino_acid_dict)
        except:
            # This try catch was here in the original thing too
            print(f"Failed to process ligand {i} in {sdffile}")
            continue
        pdb_and_mol_ids.append(i)
        lig_coords.append(ligand_data['lig_coords'])
        lig_one_hot.append(ligand_data['lig_one_hot'])
        lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))
        pocket_coords.append(pocket_data['pocket_coords'])
        pocket_one_hot.append(pocket_data['pocket_one_hot'])
        pocket_mask.append(count * np.ones(len(pocket_data['pocket_coords'])))
        count_protein.append(pocket_data['pocket_coords'].shape[0])
        count_ligand.append(ligand_data['lig_coords'].shape[0])
        count_total.append(pocket_data['pocket_coords'].shape[0]+ligand_data['lig_coords'].shape[0])

        count += 1


    lig_coords = np.concatenate(lig_coords, axis=0)
    lig_one_hot = np.concatenate(lig_one_hot, axis=0)
    lig_mask = np.concatenate(lig_mask, axis=0)
    pocket_coords = np.concatenate(pocket_coords, axis=0)
    pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
    pocket_mask = np.concatenate(pocket_mask, axis=0)

    processed_dir = args.pdb_file[:args.pdb_file.rindex("/")+1]
    print("Saving to: ", processed_dir)

    # Train.npz
    saveall(Path(processed_dir+'train.npz'), pdb_and_mol_ids, lig_coords,
            lig_one_hot, lig_mask, pocket_coords,
            pocket_one_hot, pocket_mask)
    
    # Make valid and test npz files for now. Just put them as the same
    saveall(Path(processed_dir+'valid.npz'), pdb_and_mol_ids, lig_coords,
            lig_one_hot, lig_mask, pocket_coords,
            pocket_one_hot, pocket_mask)
    saveall(Path(processed_dir+'test.npz'), pdb_and_mol_ids, lig_coords,
            lig_one_hot, lig_mask, pocket_coords,
            pocket_one_hot, pocket_mask)

    with np.load(Path(processed_dir+'train.npz'), allow_pickle=True) as data:
        lig_mask = data['lig_mask']
        pocket_mask = data['pocket_mask']
        lig_coords = data['lig_coords']
        lig_one_hot = data['lig_one_hot']
        pocket_one_hot = data['pocket_one_hot']


    n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_csv', type=str)
    parser.add_argument('--sdf_file', type=str)
    parser.add_argument('--pdb_file', type=str)
    # parser.add_argument('--create_dataloader', action='store_true')
    args = parser.parse_args()


    process_all_ligands_and_pockets(args.pdb_file, args.sdf_file, args.eval_csv)
    print("Done processing ligands and pockets!")

