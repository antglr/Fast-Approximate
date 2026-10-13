#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:49:19 2026

@author: qb275442
"""

import sys
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

def snapshot(obj, n=5):
    """Lit seulement un petit échantillon, pour ne pas saturer la mémoire."""
    try:
        if obj.shape == ():                 # dataset scalaire
            return obj[()]
        if obj.size == 0:
            return "(vide)"
        # Prend les n premiers éléments sur le 1er axe
        return obj[:n]
    except Exception as e:
        return f"(lecture impossible : {e})"

def explorer(chemin, n=5):
    with h5py.File(chemin, "r") as f:
        print(f"Fichier : {chemin}\n")

        # Attributs racine
        if f.attrs:
            print("Attributs :")
            for k, v in f.attrs.items():
                print(f"  {k} = {v}")
            print()

        def afficher(name, obj):
            profondeur = name.count("/")
            indent = "  " * profondeur
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[dataset] {name}")
                print(f"{indent}   shape={obj.shape}  dtype={obj.dtype}  "
                      f"compression={obj.compression}")
                for k, v in obj.attrs.items():
                    print(f"{indent}   attr {k} = {v}")
                print(f"{indent}   aperçu : {snapshot(obj, n)}")
            else:
                print(f"{indent}[groupe]  {name}")
                for k, v in obj.attrs.items():
                    print(f"{indent}   attr {k} = {v}")

        f.visititems(afficher)
        

def h5_to_dict(chemin, attributs=True):
    """Charge un fichier HDF5 entier dans un dictionnaire imbriqué."""

    def convert(valeur):
        # bytes -> str, et tableaux de bytes -> tableaux de str
        if isinstance(valeur, bytes):
            return valeur.decode("utf-8", errors="replace")
        if isinstance(valeur, np.ndarray) and valeur.dtype.kind in ("S", "O"):
            try:
                return valeur.astype(str)
            except Exception:
                return valeur
        return valeur

    def lire(obj):
        if isinstance(obj, h5py.Dataset):
            return convert(obj[()])
        # Groupe : on parcourt récursivement
        d = {}
        for nom, enfant in obj.items():
            d[nom] = lire(enfant)
            if attributs and enfant.attrs:
                d[nom + "__attrs"] = {k: convert(v) for k, v in enfant.attrs.items()}
        return d

    with h5py.File(chemin, "r") as f:
        resultat = lire(f)
        if attributs and f.attrs:
            resultat["__attrs__"] = {k: convert(v) for k, v in f.attrs.items()}
    return resultat


def combiner(folder, extension="*.h5"):
    res = defaultdict(list)
    for file in sorted(Path(folder).glob(extension)):
        d = h5_to_dict(file)   # ta fonction, renvoie un dict
        for key, value in d.items():
            res[key].append(value)
    return dict(res)

if __name__ == "__main__":
    folder_path = Path("../test_data/")
    
    d = combiner(folder_path)
        
