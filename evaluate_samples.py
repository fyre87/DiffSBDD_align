import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
# from torch.utils.data import DataLoader

import os
import csv
import numpy as np


from analysis.metrics import MoleculeProperties



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str) # SDF file
    # parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--output_name', type=str, default=None)
    args = parser.parse_args()

    # assert args.sdf_file.endswith('.sdf')
    # assert args.pdb_file.endswith('.pdb')

    # attempt to calculate vina score
    # input_pdb = args.pdb_file
    # output_pdbqt = str(args.pdb_file[:-4] + ".pdbqt")
    # output_pdbqt = "sample_cool/structure.pdbqt"

    # Read the PDB file
    # mol = meeko.MoleculePreparation.read_pdb(args.pdb_file)
    # mol = Chem.MolFromPDBFile(input_pdb, removeHs=True)
    # mol = Chem.AddHs(mol)
    # AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # mol = Chem.RemoveHs(mol, implicitOnly=False)
    # AllChem.ComputeGasteigerCharges(mol)

    # Prepare the molecule for docking (this step adds hydrogens, assigns charges, etc.)
    # mol_prep = meeko.MoleculePreparation()
    # mol_prep.prepare(mol)

    # Write the prepared molecule to a PDBQT file
    # pdbqt_file = mol_prep.write_pdbqt_string(output_pdbqt)

    # print(pdbqt_file)

    # scores = calculate_qvina2_score(output_pdbqt, "sample_cool/1dtl.sdf", "")
    # print(scores)
    # raise ValueError("Done")

    if args.output_name == None:
        raise ValueError("Specify an output file name using \"--output_name [name]\"")
    if args.output_name.endswith(".csv") == False:
        args.output_name = args.output_name + ".csv"

    # Function to process a single SDF file
    def process_sdf(file_path):
        suppl = Chem.SDMolSupplier(file_path)
        molecules = [mol for mol in suppl]
        return molecules  # Return a list of molecules

    # Create an instance of MoleculeProperties
    mol_props = MoleculeProperties()

    # Specify the folder containing SDF files
    # sdf_folder = args.file

    # Prepare CSV file for writing results
    # csv_file_path = '/home/user/DiffSBDD-orig/molecule_metrics.csv'
    csv_header = ['Filename', 'QED', 'SA', 'LogP', 'Lipinski']

    with open(args.output_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        # Loop through all SDF files in the folder
        # for filename in os.listdir(sdf_folder):
        #     if filename.endswith('.sdf'):
        #         file_path = os.path.join(sdf_folder, filename)
                
        # Process the SDF file
        molecules = process_sdf(args.file)
        
        qeds = []
        sas = []
        lipinskis = []
        
        # Evaluate each molecule individually
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
            qed = mol_props.calculate_qed(mol)
            sa = mol_props.calculate_sa(mol)
            logp = mol_props.calculate_logp(mol)
            lipinski = mol_props.calculate_lipinski(mol)
            
            # Create the filename_molecule_number identifier
            # molecule_id = f"{args.file}_{i+1}"
            
            # Write results to CSV
            csv_writer.writerow([i, qed, sa, logp, lipinski])
            
            qeds.append(qed)
            sas.append(sa)
            lipinskis.append(lipinski)
        
        print(f"Processed {args.file}")

    print(f"Results saved to {args.output_name}")
    
    # print mean and top
    qeds = np.array(qeds)
    sas = np.array(sas)
    lipinskis = np.array(lipinskis)
    
    print(f"Mean QED: {np.mean(qeds)}, Top QED: {np.max(qeds)}, Median QED: {np.median(qeds)}")
    print(f"Mean SA: {np.mean(sas)}, Top SA: {np.max(sas)}, Median SA: {np.median(sas)}")
    print(f"Mean Lipinski: {np.mean(lipinskis)}, Top Lipinski: {np.max(lipinskis)}, Median Lipinski: {np.median(lipinskis)}")