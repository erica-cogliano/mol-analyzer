import io
import os
import pubchempy as pcp
import pandas as pd
from tqdm import tqdm

from loguru import logger

from rdkit.Chem import (
    ForwardSDMolSupplier,
    Mol,
    SDWriter,
    rdFMCS,
    rdFingerprintGenerator,
    Draw,
    SDMolSupplier,
    DataStructs,
    MolFromSmarts,
)
from rdkit.ML.Cluster import Butina

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from rdkit.Chem.Scaffolds import MurckoScaffold


def InitLogger():
    # Prendi il livello dalla variabile d'ambiente,
    # ma usa "INFO" come paracadute se non è impostata
    current_level = os.getenv("LOGURU_LEVEL", "INFO")
    logger.remove()  # Rimuove il logger predefinito
    logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True, level=current_level
    )  # Aggiunge un nuovo logger con il livello specificato


def LoadMolecules(file_path="data/compounds.sdf") -> SDMolSupplier:
    """
    Questa funzione carica le molecole da un file SDF. Se il file non esiste,
    scarica i dati da PubChem usando i nomi dei farmaci e salva il file SDF per usi futuri.
    """
    compounds_sdf_file_name = file_path
    if not os.path.exists(compounds_sdf_file_name):
        # Carica un file csv
        ttd_file_name = "data/ttd_drug_disease_ovarian_by_drug.csv"
        df = pd.read_csv(ttd_file_name)
        # Prendi solo i nomi delle molecole nella colonna "drug_name"
        molecule_names = df["drug_name"].tolist()
        logger.info(
            "Caricati {} nomi di molecole dal file {}".format(
                len(molecule_names), ttd_file_name
            )
        )

        logger.info(
            "Il file {} non esiste. Scaricando i dati da PubChem...".format(
                compounds_sdf_file_name
            )
        )
        mols = FindMolsForNames(molecule_names)
        SaveMolsToSDF(mols, compounds_sdf_file_name)
    else:
        mols = SDMolSupplier(compounds_sdf_file_name)

    return mols


def FindMolsForNames(names):
    mols = []
    for name in tqdm(names, "Cercando molecole per nomi"):
        mol = FindMolByName(name)
        if mol is not None:
            mols.append(mol)
    logger.info(f"Trovate {len(mols)} molecole per {len(names)} nomi")
    return mols


def FindMolByName(name: str) -> Mol:
    """
    Questa funzione cerca una molecola su PubChem usando il nome e restituisce un oggetto Mol.
    Se la ricerca fallisce o non viene trovato nulla, restituisce None.
    """
    logger.debug(f"Cercando la molecola '{name}' su PubChem")
    try:
        cids = pcp.get_cids(name, namespace="name")
    except Exception as e:
        logger.error(f"Connessione fallita per '{name}': {e}")
        return None
    if cids is None or len(cids) == 0:
        logger.warning(f"Nessun CID trovato per '{name}'")
        cids = FindCidsBySubstanceName(name)

    # una volta ottenuti i CID, possiamo ottenere la migliore molecola associata a quei CID
    return GetBestMolFromCids(cids, name)


def FindCidsBySubstanceName(name: str) -> Mol:
    """
    Questa funzione cerca una molecola su PubChem usando il nome come sostanza e restituisce una lista di CID associati a quella sostanza.
    Se la ricerca fallisce o non viene trovato nulla, restituisce None.
    """
    logger.debug(f"Cercando la molecola '{name}' su PubChem come sostanza")
    try:
        # Cerca le substance ID (SID) per il nome dato.
        sids = pcp.get_sids(name, namespace="name", domain="substance")
    except Exception as e:
        logger.error(f"Connessione fallita per '{name}' come sostanza: {e}")
        return None
    if sids is None or len(sids) == 0:
        logger.warning(f"Nessuna substance trovata per '{name}'")
        return None

    # Le substance potrebbero essere composte da piu' compound, quindi
    # prendiamo tutti i SID trovati e cerchiamo i CID a loro associati.
    return FindCidsBySids(sids, name)


def FindCidsBySids(sids, name):
    """
    Questa funzione prende una lista di SIDs e restituisce una lista di CID associati a quei SIDs.
    """
    # Cerca i CID associati a ciascun SID.
    cids_dict = pcp.get_cids(sids, namespace="sid", domain="substance")

    # cids_dict e' un array di dizionari [ {..}, {..}, .. ]
    # dove ogni entry e' un dizionario con attributi SID e CID,
    # ovvero ha la forma { "SID": 12345, "CID": [ 67890, 67891 ] }
    # Si puo' notare anche che CID e' una lista, perche' una substance puo' essere associata a piu' CID.
    cids = []
    for entry in cids_dict:
        if "CID" in entry:
            cids.extend(entry["CID"])
    if len(cids) == 0:
        logger.warning(f"Nessun CID trovato per '{name}' come substance")
        return None
    # Deduplica i CID
    cids = list(set(cids))


def GetBestMolFromCids(cids, name: str) -> Mol:
    """
    Questa funzione prende una lista di compound ID e restituisce la molecola con il maggior numero di atomi.
    Inoltre associa il nome alla molecola usando la proprietà `_Name`.
    Se il testo SDF è vuoto o non valido, restituisce None.
    """
    if cids is None or len(cids) == 0:
        logger.warning(f"Nessun CID valido trovato per '{name}'")
        return None

    # Otteniamo il testo SDF per i CID dati
    sdf_text = pcp.get_sdf(cids)

    if sdf_text is None or sdf_text.strip() == "":
        logger.warning(f"SDF vuoto o non valido per '{name}'")
        return None

    # Trasformare il testo SDF in un oggetto richiede una particolare attenzione.
    # Bisogna creare uno stream di byte a partire dal testo SDF, e poi passarlo a ForwardSDMolSupplier.
    sdf_stream = io.BytesIO(sdf_text.encode("utf-8"))
    supplier = ForwardSDMolSupplier(sdf_stream)

    # Se ci sono più molecole, prendiamo quella con il maggior numero di atomi
    best_mol = GetBestMolFromSupplier(supplier)

    # Assicuriamoci di associare name al risultato
    if best_mol is not None:
        best_mol.SetProp("_Name", name)
    return best_mol


def GetBestMolFromSupplier(supplier: ForwardSDMolSupplier) -> Mol:
    """
    Questa funzione prende un ForwardSDMolSupplier e restituisce la molecola con il maggior numero di atomi.
    """
    best_mol = None
    max_atoms = 0
    for mol in supplier:
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms
                best_mol = mol
    return best_mol


def GetMolName(mol):
    return mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"


def SaveMolsToSDF(mols, file_path):
    writer = SDWriter(file_path)
    for mol in mols:
        writer.write(mol)
    writer.flush()
    writer.close()
    logger.info(f"Salvate {len(mols)} molecole in {file_path}")


def PrintBiggestMol(mols):
    biggest_mol = None
    max_atoms = 0
    for mol in mols:
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms
                biggest_mol = mol
    if biggest_mol is not None:
        logger.info(
            f"La molecola più grande è '{GetMolName(biggest_mol)}' con {max_atoms} atomi"
        )
    else:
        logger.warning("Nessuna molecola valida trovata")


# Definiamo una funzione che ci trasforma le molecole di un SDF in un vettore di fingerprint
# @input: SDMol
# @output: Vettore di fingerprint
def GetFingerprintFromSDMol(mols: SDMolSupplier):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for mol in tqdm(mols, "Generando fingerprint"):
        logger.debug("Generando fingerprint per molecola: {}".format(GetMolName(mol)))
        fp = gen.GetFingerprint(mol)
        fps.append(fp)
    return fps


# Definiamo una funzione che generi la matrice di similarita' di tanimoto a partire da fingerprints
# input: fingerprints
# output: matrice di similarita' in formato adatto a Butina.ClusterData
def GetSimilarityMatrixFromFingerprints(fingerprints):
    sim_mat = []
    n_fps = len(fingerprints)

    for i in tqdm(range(1, n_fps), "Calcolando similarita'"):
        # Per ogni fingerprint, calcolo la sua similarita' con tutte le altre fingerprint
        logger.debug("Calcolando similarita' per fingerprint {}".format(i))
        current_fp = fingerprints[i]
        similarita_con_altre = DataStructs.BulkTanimotoSimilarity(
            current_fp, fingerprints[:i]
        )
        sim_mat.extend(similarita_con_altre)

    return sim_mat


# Definiamo una funzione che generi la matrice delle distanze a partire da fingerprints
# input: vettore di fingerprint
# output: matrice di distanze in formato adatto a Butina.ClusterData
def GetDistanceMatrixFromFingerprints(fingerprints):
    dist_mat = []
    sim_mat = GetSimilarityMatrixFromFingerprints(fingerprints)

    for sim_value in sim_mat:
        # Per ogni valore della matrice di similarita'
        # Trasformo il valore di similarita' in un valore di distanza con la seguente formula
        dist_value = 1.0 - sim_value
        dist_mat.append(dist_value)

    return dist_mat


# Definiamo una funzione che genera i cluster con Butina a partire da una matrice delle distanze
# input:
#  - distance_matrix: matrice delle distanze in formato adatto a Butina.ClusterData
#  - n: Numero di righe e colonne della matrice
# output: clusters
def GetClustersFromDistanceMatrix(distance_matrix, n, dist_tresh):
    return Butina.ClusterData(
        distance_matrix, n, distThresh=dist_tresh, isDistData=True
    )


# Questa funzione espande una distance matrix da un formato Butina ad un formato adatto a MDS e KMeans
# input:
#  - compressed_distance_matrix: distance matrix adatto a Butina.ClusterData
#  - n: Numero di righe e colonne della matrice
# output: matrice completa
def ExpandDistanceMatrix(compressed_distance_matrix, n):
    expanded_matrix = []

    for i in range(n):
        new_row = []
        for j in range(n):
            if i == j:
                new_row.append(0.0)
            elif i < j:
                # Butina distance matrix is compressed, so we need to calculate the index in the compressed format
                index = int((j * (j - 1)) / 2 + i)
                new_row.append(compressed_distance_matrix[index])
            else:
                # The distance matrix is symmetric, so we can use the value from the upper triangle
                index = int((i * (i - 1)) / 2 + j)
                new_row.append(compressed_distance_matrix[index])
        expanded_matrix.append(new_row)

    return expanded_matrix


# Definiamo una funzione che genera i cluster con KMeans a partire da una matrice delle distanze
# input:
#  - distance_matrix: matrice delle distanze in formato adatto a MDS e KMeans
#  - cluster_count: numero di cluster da generare
# output: clusters
def GetKMeansClustersFromDistanceMatrix(distance_matrix, cluster_count):
    clusters = []

    logger.debug("Eseguendo MDS per ridurre la matrice di distanze a 2 dimensioni")
    mds = MDS(dissimilarity="precomputed")
    fprints_2d = mds.fit_transform(distance_matrix)

    logger.debug("Eseguendo KMeans per generare {} cluster".format(cluster_count))
    km = KMeans(cluster_count)
    # KMeans fit predict restituisce un vettore di cluster a cui ogni molecola appartiene,
    # ad esempio [0, 0, 1, 1, 2] significa che le prime due molecole appartengono al cluster 0,
    # le successive due al cluster 1 e l'ultima al cluster 2
    clusters = km.fit_predict(fprints_2d)

    # Ora vogliamo trasformare questo vettore di cluster in una lista di cluster,
    # dove ogni cluster e' una tupla di indici di molecole
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(idx)

    # Convertiamo il dizionario in una lista di tuple
    clusters_list = [tuple(v) for v in cluster_dict.values()]

    return clusters_list


# definiamo una funzione che genera MCSs a partire dai clusters
# input:
# -clusters: un vettore di tuple di indici di molecole
# -mols: contenitore di molecole di cui gli indici fanno riferimento
# output:
# -MCSs
def GetClustersMCS(clusters, mols):
    cluster_results = []
    for i, cluster in tqdm(
        enumerate(clusters), total=len(clusters), desc="Calcolando MCS per ogni cluster"
    ):
        # un cluster e' una tupla di indici (es.(1,5,10))
        if len(cluster) < 2:
            # salta i clustert con una sola molecola
            continue
        # trasforma cluster (che contiene indici delle molecole) in un vettore di molecole vere e proprie
        mols_in_cluster = [mols[idx] for idx in cluster]
        # trova la MCS per questo specifico cluster
        logger.debug(
            "Calcolando MCS per cluster {} con {} molecole".format(i, len(cluster))
        )
        mcs_res = rdFMCS.FindMCS(mols_in_cluster)
        # converti il risultato MCS in una molecola visualizzabile
        mcs_mol = MolFromSmarts(mcs_res.smartsString)
        cluster_results.append(
            {
                "cluster_id": i,
                "size": len(cluster),
                "mcs_smarts": mcs_res.smartsString,
                "mcs_mol": mcs_mol,
                "mols_in_cluster": mols_in_cluster,
            }
        )
    return cluster_results


# Funzione che disegna gli MCS di ogni cluster
# input: clusters_mcs - risultato della funzione GetClustersMCS
def DrawClustersMCS(clusters_mcs):
    for cluster_mcs in clusters_mcs:
        mcs = cluster_mcs["mcs_mol"]
        id = cluster_mcs["cluster_id"]
        cluster_size = cluster_mcs["size"]
        legend = f"MCS Cluster {id} (size {cluster_size})"
        for mol in cluster_mcs["mols_in_cluster"]:
            legend += f"\n- {GetMolName(mol)}"

        DrawMol(mcs, out_dir="out/mcs", name_prefix=f"cluster_{id}_mcs", legend=legend)


def DrawMols(mols):
    for mol in mols:
        DrawMol(mol)


def DrawMol(
    mol, out_dir: str = "out/mols", name_prefix: str = "mol", legend: str = None
):
    name = GetMolName(mol)
    img = Draw.MolToImage(mol, size=(600, 600), legend=legend if legend else name)

    # Se la cartella out non esiste, creala
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Salva l'immagine in out/mol_{name}.png
    img.save(f"{out_dir}/{name_prefix}_{name}.png")


# Funzione che ottiene i murcko scaffold di ogni molecola
# input: mols - contenitore di molecole di cui vogliamo ottenere i murcko scaffold
# output: lista di murcko scaffold (sempre formato Mol)
def GetScaffoldsFromSDMol(mols):
    return [MurckoScaffold.GetScaffoldForMol(m) for m in mols]


# Funzione che rende i murcko scaffold generici, ignorando anche i tipi di atomi e di legami
# input: scaffolds - lista di murcko scaffold (sempre formato Mol)
# output: lista di murcko scaffold generici (sempre formato Mol)
def MakeScaffoldsGeneric(scaffolds):
    return [MurckoScaffold.MakeScaffoldGeneric(s) for s in scaffolds]
