import csv
from rdkit import Chem
from rdkit.Chem import Draw

# Read output.txt
substructures = []
with open('week4/output.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            smiles = row['description']
            support = float(row['s_abs'])
            substructures.append({'smiles': smiles, 'support': support, 'id': row['id']})
        except ValueError:
            continue

# Sort by support descending
substructures.sort(key=lambda x: x['support'], reverse=True)

# Get top 5
top5 = substructures[:5]

print(f"Found {len(substructures)} substructures.")
print("Top 5 frequent substructures:")

for i, sub in enumerate(top5):
    print(f"Rank {i+1}: ID={sub['id']}, Support={sub['support']}, SMILES={sub['smiles']}")
    
    mol = Chem.MolFromSmiles(sub['smiles'])
    if mol is None:
        mol = Chem.MolFromSmarts(sub['smiles'])

    if mol:
        Chem.SanitizeMol(mol)

        filename = f"week4/top{i+1}_freq_substructure.png"
        Draw.MolToFile(mol, filename)
        print(f"Saved image to {filename}")
    else:
        print(f"Failed to generate molecule from SMILES/SMARTS: {sub['smiles']}")
