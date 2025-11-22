import pandas as pd
import numpy as np
import math
import csv
import re

def calculate_entropy(y):
    n = len(y)
    if n == 0: return 0
    p1 = np.sum(y) / n
    p0 = 1 - p1
    if p0 == 0 or p1 == 0: return 0
    return -p0 * math.log2(p0) - p1 * math.log2(p1)

def calculate_conditional_entropy(y, x):
    n = len(y)
    if n == 0: return 0
    
    # Split y by x
    y0 = y[x == 0]
    y1 = y[x == 1]
    
    p0 = len(y0) / n
    p1 = len(y1) / n
    
    h0 = calculate_entropy(y0)
    h1 = calculate_entropy(y1)
    
    return p0 * h0 + p1 * h1

def calculate_conditional_mi(y, x_new, x_given):
    # H(Y | X_given)
    h_y_given_x = calculate_conditional_entropy(y, x_given)
    
    # H(Y | X_given, X_new)
    n = len(y)
    groups = {}
    for i in range(n):
        key = (x_given[i], x_new[i])
        if key not in groups:
            groups[key] = []
        groups[key].append(y[i])
        
    h_y_given_both = 0
    for key, group in groups.items():
        p_group = len(group) / n
        h_group = calculate_entropy(np.array(group))
        h_y_given_both += p_group * h_group
        
    return h_y_given_x - h_y_given_both

def get_confidence(y, x):
    """P(Y=1 | X=1)"""
    if np.sum(x) == 0: return 0
    return np.sum((y == 1) & (x == 1)) / np.sum(x)

df_molecules = pd.read_csv('week4/molecule.csv', sep=';')
n_molecules = len(df_molecules)
cancer_types = ['anti_cancer_1', 'anti_cancer_2', 'anti_cancer_3']
cancer_data = {c: df_molecules[c].values for c in cancer_types}

cancer_id_map = {
    101: 'anti_cancer_1',
    102: 'anti_cancer_2',
    103: 'anti_cancer_3'
}

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


significant_rules = []
# Regex: 65 98 -> 6 fr=12 (0.2553), cf=1.000, gamma=3.917, delta=0.190, M=8.196e-01
#regex is kinda fun though
rule_pattern = re.compile(r'([\d\s]+)->\s*(\d+).*cf=([0-9\.]+).*M=([0-9\.eE\+\-]+)')

try:
    f = open('week4/kingfisher_output_large.txt', 'r', encoding='utf-16')
    f.read(100)
    f.seek(0)
except:
    f = open('week4/kingfisher_output_large.txt', 'r', encoding='utf-8', errors='ignore')
with f:
    for line in f:
        match = rule_pattern.search(line)
        if match:
            antecedent_str = match.group(1).strip()
            consequence_str = match.group(2).strip()
            conf_str = match.group(3).strip()
            mi_str = match.group(4).strip()
            
            antecedent = [int(x) for x in antecedent_str.split()]
            consequence = int(consequence_str)
            conf = float(conf_str)
            mi = float(mi_str)
            
        
            if consequence not in cancer_id_map:
                continue
            if any(x in cancer_id_map for x in antecedent):
                continue
            
            if len(antecedent) > 2:
                continue
            
            significant_rules.append({
                'antecedent': antecedent,
                'cancer': cancer_id_map[consequence],
                'conf': conf,
                'mi': mi,
                'size': len(antecedent)
            })
                

print(f"{len(significant_rules)} significant rules from Kingfisher.")

MIC_THRESHOLD = 0.05
representative_rules = []

for rule in significant_rules:
    # size 1 always representative
    if rule['size'] == 1:
        representative_rules.append(rule)
        continue

    # Rule: AB -> C
    sub1 = rule['antecedent'][0]
    sub2 = rule['antecedent'][1]
    c_name = rule['cancer']
    y = cancer_data[c_name]
    
    x1 = substructure_presence[sub1]
    x2 = substructure_presence[sub2]
    x_combined = x1 & x2
    conf_combined = get_confidence(y, x_combined)
    
    prune = False
    
    # Check subset 1
    conf1 = get_confidence(y, x1)
    if conf1 >= conf_combined:
        prune = True
    else:
        mic1 = calculate_conditional_mi(y, x2, x1) # I(C; sub2 | sub1)
        if mic1 <= MIC_THRESHOLD:
            prune = True
            
    if prune: continue
    
    # Check subset 2
    conf2 = get_confidence(y, x2)
    if conf2 >= conf_combined:
        prune = True
    else:
        mic2 = calculate_conditional_mi(y, x1, x2) # I(C; sub1 | sub2)
        if mic2 <= MIC_THRESHOLD:
            prune = True
            
    if not prune:
        representative_rules.append(rule)

# Sort by MI
significant_rules.sort(key=lambda x: x['mi'], reverse=True)
representative_rules.sort(key=lambda x: x['mi'], reverse=True)

print(f"Selected {len(representative_rules)} representative rules.")

with open('kingfisher_significant_rules.txt', 'w') as f:
    f.write("Rank,Rule,Cancer,MI,Confidence,Size\n")
    for i, r in enumerate(significant_rules):
        ant_str = " AND ".join([f"Sub_{x}" for x in r['antecedent']])
        f.write(f"{i+1},{ant_str},{r['cancer']},{r['mi']:.6f},{r['conf']:.6f},{r['size']}\n")

with open('kingfisher_representative_rules.txt', 'w') as f:
    f.write("Rank,Rule,Cancer,MI,Confidence,Size\n")
    for i, r in enumerate(representative_rules):
        ant_str = " AND ".join([f"Sub_{x}" for x in r['antecedent']])
        f.write(f"{i+1},{ant_str},{r['cancer']},{r['mi']:.6f},{r['conf']:.6f},{r['size']}\n")

print("Saved rules to 'week4/kingfisher_significant_rules.txt' and 'week4/kingfisher_representative_rules.txt'.")
