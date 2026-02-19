"""
jsonl_to_3di.py  —  CATH jsonl to PDB files to Foldseek 3Di to merged jsonl

Usage:
    python jsonl_to_3di.py \
        --jsonl     data/cath/chain_set.jsonl \
        --pdb_dir   data/cath/pdb_files \
        --tsv       data/cath/3di_raw.tsv \
        --out_jsonl data/cath/chain_set_3di.jsonl \
"""

import json
import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


AA3 = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
    'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
    'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
    'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR',
}
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
ATOM_LINE = (
    "ATOM  {serial:5d} {name:<4s}{altLoc:1s}{resName:3s} {chainID:1s}"
    "{resSeq:4d}{iCode:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}"
    "{occupancy:6.2f}{tempFactor:6.2f}          "
    "{element:>2s}\n"
)


def is_nan(v):
    return v != v



def write_pdb(entry: dict, path: str) -> int:
    seq, coords = entry['seq'], entry['coords']
    serial, res_idx, lines = 1, 0, []
    for i in range(len(seq)):
        atom_coords = {}
        for atom in BACKBONE_ATOMS:
            row = coords.get(atom, [[]])[i]
            if row is None or len(row) != 3 or any(is_nan(v) for v in row):
                return 0   # NaN check 
            atom_coords[atom] = row
        res_idx += 1
        for atom in BACKBONE_ATOMS:
            x, y, z = atom_coords[atom]
            lines.append(ATOM_LINE.format(
                serial=serial, name=f" {atom:<3s}", altLoc=' ',
                resName=AA3[seq[i]], chainID='A', resSeq=res_idx, iCode=' ',
                x=x, y=y, z=z, occupancy=1.0, tempFactor=0.0, element=atom[0],
            ))
            serial += 1
    lines.append("END\n")
    with open(path, 'w') as f:
        f.writelines(lines)
    return res_idx


def jsonl_to_pdbs(jsonl_path: str, pdb_dir: str, max_length: int = 500) -> list:
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)
    alphabet_set = set('ACDEFGHIKLMNPQRSTVWY')
    entries, skipped = [], 0

    with open(jsonl_path) as f:
        lines = f.readlines()

    print(f"\nWriting PDBs  {pdb_dir}/  ({len(lines)} entries in jsonl)")
    for line in tqdm(lines, desc="  PDBs"):
        entry  = json.loads(line)
        seq    = entry['seq']
        name   = entry['name']
        coords = entry['coords']

        # Filter too long
        if len(set(seq) - alphabet_set) > 0 or len(seq) > max_length:
            skipped += 1
            continue

        # Filter: any NaN in any backbone atom, skip entire protein
        has_nan = False
        for i in range(len(seq)):
            for atom in BACKBONE_ATOMS:
                row = coords.get(atom, [[]])[i]
                if row is None or len(row) != 3 or any(is_nan(v) for v in row):
                    has_nan = True
                    break
            if has_nan:
                break
        if has_nan:
            skipped += 1
            continue

        safe_name = name.replace('/', '_').replace('.', '_')
        pdb_path  = os.path.join(pdb_dir, f"{safe_name}.pdb")
        if write_pdb(entry, pdb_path) == 0:
            skipped += 1
            continue

        entries.append({
            'name':      name,
            'safe_name': safe_name,
            'seq':       seq,
            'CATH':      entry.get('CATH', []),
        })

    print(f"  Written: {len(entries)}  |  Skipped: {skipped}")
    return entries


def run_foldseek(pdb_dir: str, tsv_path: str):
    print(f"\nRunning foldseek  {tsv_path}")
    subprocess.run(
        ['foldseek', 'structureto3didescriptor', pdb_dir, tsv_path],
        check=True,
    )
    print("  Done.")


def parse_tsv(tsv_path: str) -> dict:
    """Returns dict: "12as_A.pdb" to "DVQTA..." """
    mapping = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                mapping[parts[0]] = parts[2].upper()
    print(f"  Parsed {len(mapping)} 3Di sequences  (sample key: {next(iter(mapping))})")
    return mapping

def merge(entries: list, mapping: dict) -> list:
    matched, unmatched = [], []
    for e in entries:
        seq_3di = mapping.get(e['safe_name'] + '.pdb')
        if seq_3di:
            e['seq_3di'] = seq_3di
            matched.append(e)
        else:
            unmatched.append(e['name'])
    print(f"\n[Step 3] Matched: {len(matched)}  |  Unmatched: {len(unmatched)}")
    if unmatched:
        print(f"  First unmatched: {unmatched[:3]}")
    return matched

def write_jsonl(matched: list, out_path: str):
    import random
    # Sanity check
    mismatches = sum(len(e['seq']) != len(e['seq_3di']) for e in matched)
    print(f"\n Length mismatches: {mismatches}  (must be 0)")
    assert mismatches == 0, "seq / seq_3di length mismatch — check pipeline"

    print("  Examples:")
    for e in random.sample(matched, min(5, len(matched))):
        print(f"    {e['name']:<15s}  AA : {e['seq'][:30]}")
        print(f"    {'':15s}  3Di: {e['seq_3di'][:30]}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for e in matched:
            f.write(json.dumps({
                'name':    e['name'],
                'seq':     e['seq'],
                'seq_3di': e['seq_3di'],
                'CATH':    e['CATH'],
            }) + '\n')
    print(f"\n  Wrote {len(matched)} entries  {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl',      default='data/cath/chain_set.jsonl')
    parser.add_argument('--pdb_dir',    default='data/cath/pdb_files')
    parser.add_argument('--tsv',        default='data/cath/3di_raw.tsv')
    parser.add_argument('--out_jsonl',  default='data/cath/chain_set_3di.jsonl')
    parser.add_argument('--max_length', type=int, default=500)
    args = parser.parse_args()

    entries = jsonl_to_pdbs(args.jsonl, args.pdb_dir, args.max_length)
    run_foldseek(args.pdb_dir, args.tsv)
    mapping = parse_tsv(args.tsv)
    matched = merge(entries, mapping)
    write_jsonl(matched, args.out_jsonl)
    print("Done.")


if __name__ == '__main__':
    main()