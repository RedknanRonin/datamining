import pandas as pd
import numpy as np
import csv

df_molecules = pd.read_csv('week4/molecule.csv', sep=';')
n_molecules = len(df_molecules)
cancer_types = ['anti_cancer_1', 'anti_cancer_2', 'anti_cancer_3']
cancer_data = {c: df_molecules[c].values for c in cancer_types}


priors = {c: np.sum(cancer_data[c]) / n_molecules for c in cancer_types}


substructure_presence = {}
with open('week4/ids.txt', 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(':')
        if len(parts) < 2: continue
        sub_id = int(parts[0])
        mol_ids = [int(x) for x in parts[1].split(',')]
        vec = np.zeros(n_molecules, dtype=int)
        vec[mol_ids] = 1
        substructure_presence[sub_id] = vec


rules = []
with open('week4/kingfisher_representative_rules.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Parse antecedent "Sub_1 AND Sub_2"
        ant_str = row['Rule'].split('->')[0].strip()
        parts = ant_str.split(' AND ')
        antecedent = []
        for p in parts:
            if p.startswith('Sub_'):
                antecedent.append(int(p.split('_')[1]))
        
        rules.append({
            'antecedent': antecedent,
            'cancer': row['Cancer'],
            'mi': float(row['MI']),
            'conf': float(row['Confidence'])
        })


TARGET_ID = 33

print(f"\nAnalyzing Molecule {TARGET_ID}...")


true_labels = {c: df_molecules.iloc[TARGET_ID][c] for c in cancer_types}
print("True Labels:")
for c, val in true_labels.items():
    print(f"  {c}: {val}")


present_substructures = set()
for sub_id, vec in substructure_presence.items():
    if vec[TARGET_ID] == 1:
        present_substructures.add(sub_id)

print(f"Present Substructures: {sorted(list(present_substructures))}")

# i) Select matching rules
matching_rules = []
for r in rules:
    
    if all(sub_id in present_substructures for sub_id in r['antecedent']):
        if len(r['antecedent']) == 1:
            x_vec = substructure_presence[r['antecedent'][0]]
        else:
            x_vec = substructure_presence[r['antecedent'][0]] & substructure_presence[r['antecedent'][1]]
            
        y_vec = cancer_data[r['cancer']]
        
        p_x = np.sum(x_vec) / n_molecules
        p_c = priors[r['cancer']]
        p_xc = np.sum(x_vec & y_vec) / n_molecules
        
        lift = r['conf'] / p_c if p_c > 0 else 0
        leverage = p_xc - (p_x * p_c)
        mi_conf = r['mi'] * r['conf']
        
        r_extended = r.copy()
        r_extended.update({
            'lift': lift,
            'leverage': leverage,
            'mi_conf': mi_conf
        })
        matching_rules.append(r_extended)

print(f"Found {len(matching_rules)} matching rules for Molecule {TARGET_ID}.")

# ii) Group by C and compute goodness measures
grouped_metrics = {}

for c in cancer_types:
    c_rules = [r for r in matching_rules if r['cancer'] == c]
    
    if not c_rules:
        grouped_metrics[c] = None
        continue
        
    avg_conf = np.mean([r['conf'] for r in c_rules])
    avg_mi = np.mean([r['mi'] for r in c_rules])
    avg_mi_conf = np.mean([r['mi_conf'] for r in c_rules])
    avg_lift = np.mean([r['lift'] for r in c_rules])
    avg_leverage = np.mean([r['leverage'] for r in c_rules])
    count = len(c_rules)
    
    grouped_metrics[c] = {
        'count': count,
        'avg_conf': avg_conf,
        'avg_mi': avg_mi,
        'avg_mi_conf': avg_mi_conf,
        'avg_lift': avg_lift,
        'avg_leverage': avg_leverage
    }

# iii) Output and Prediction
print("\nPrediction Analysis:")
print("-" * 81)
print(f"{'Cancer Type':<15} | {'True':<5} | {'Count':<5} | {'Avg Conf':<8} | {'Avg MI':<8} | {'Avg Lift':<8} | {'Prediction'}")
print("-" * 81)

for c in cancer_types:
    m = grouped_metrics[c]
    true_val = true_labels[c]
    
    if m:
        prediction = 1 
        print(f"{c:<15} | {true_val:<5} | {m['count']:<5} | {m['avg_conf']:.4f}   | {m['avg_mi']:.4f}   | {m['avg_lift']:.4f}   | {prediction}")
    else:
        prediction = 0
        print(f"{c:<15} | {true_val:<5} | {'0':<5} | {'-':<8} | {'-':<8} | {'-':<8} | {prediction}")


