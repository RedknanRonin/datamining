import csv


mol_substructures = {} # index -> set of ints

with open('week4/ids.txt', 'r') as f:
    next(f) # Skip header
    for line in f:
        parts = line.strip().split(':')
        if len(parts) < 2:
            continue
        sub_id = int(parts[0])
        mol_ids = [int(x) for x in parts[1].split(',')]
        
        for mid in mol_ids:
            if mid not in mol_substructures:
                mol_substructures[mid] = set()
            mol_substructures[mid].add(sub_id)


cancer_map = {
    'anti_cancer_1': 101,
    'anti_cancer_2': 102,
    'anti_cancer_3': 103
}

print(len(mol_substructures))
with open('week4/molecule.csv', 'r') as csvfile, open('week4/kingfisher_input.txt', 'w', newline='\n') as outfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for i, row in enumerate(reader):
        subs = mol_substructures.get(i, set())
        cancers = []
        for c_name, c_id in cancer_map.items():
            if int(row[c_name]) == 1:
                cancers.append(c_id)

        # kingfisher wants space separated integers
        attributes = sorted(list(subs) + cancers)
        
        line = " ".join(map(str, attributes))
        outfile.write(line + "\n")

print("Created week4/kingfisher_input.txt")
