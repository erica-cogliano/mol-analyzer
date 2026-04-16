from enum import Enum
import io
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pubchempy as pcp
import pandas as pd
import numpy as np
from sammon.sammon import sammon
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
from PIL import Image

from loguru import logger

from rdkit.Chem import (
    ForwardSDMolSupplier,
    Mol,
    SDWriter,
    rdFMCS,
    rdFingerprintGenerator,
    Draw,
    rdDepictor,
    SDMolSupplier,
    DataStructs,
    MolFromSmarts,
    FastFindRings,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina

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


def LoadMolecules(file_path, fallback_path, include_substances=False) -> SDMolSupplier:
    """
    Questa funzione carica le molecole da un file SDF in `file_path`. Se il file non esiste,
    scarica i dati da PubChem usando i nomi dei farmaci trovati in file csv in `fallback_path`
    e salva le MOL nel file SDF in `file_path` per usi futuri.

    - `file_path`: percorso del file SDF da cui caricare le molecole
    - `fallback_path`: percorso del file CSV da cui caricare i nomi dei farmaci se il file SDF non esiste
    - `include_substances`: se True, include anche le sostanze (substance) nella ricerca su PubChem,
       altrimenti cerca solo nei composti (compound). Nota che includere le sostanze può aumentare
       significativamente il tempo di ricerca e non garantisce risultati aggiuntivi,
       quindi è disabilitato per default.
    """
    compounds_sdf_file_name = file_path
    if not os.path.exists(compounds_sdf_file_name):
        df = pd.read_csv(fallback_path)
        # Prendi solo i nomi delle molecole nella colonna "drug_name"
        molecule_names = df["drug_name"].tolist()
        logger.info(
            "Caricati {} nomi di molecole dal file {}".format(
                len(molecule_names), fallback_path
            )
        )

        logger.info(
            "Il file {} non esiste. Scaricando i dati da PubChem...".format(
                compounds_sdf_file_name
            )
        )
        mols = FindMolsForNames(molecule_names, include_substances=include_substances)
        SaveMolsToSDF(mols, compounds_sdf_file_name)
    else:
        mols = SDMolSupplier(compounds_sdf_file_name)

    return mols


def GetListFromSDMolSupplier(supplier: SDMolSupplier) -> list[Mol]:
    """Transforma un SDMolSupplier in una lista di Mol, filtrando eventuali None che indicano molecole non caricate correttamente"""
    return [m for m in supplier if m is not None]


def FindMolsForNames(names, include_substances=True) -> list[Mol]:
    mols = []
    for name in tqdm(names, "Cercando molecole per nomi"):
        mol = FindMolByNameWithSynonyms(name, include_substances)
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
    namespace = PubChemNamespace.CID
    domain = PubChemDomain.COMPOUND
    try:
        ids = pcp.get_cids(name, namespace="name")
    except Exception as e:
        logger.error(f"Connessione fallita per '{name}': {e}")
        return None
    if ids is None or len(ids) == 0:
        logger.warning(f"Nessun ID trovato per '{name}'")
        namespace = PubChemNamespace.SID
        domain = PubChemDomain.SUBSTANCE
        ids = FindSidsBySubstanceName(name)

    # una volta ottenuti gli ID, possiamo ottenere la migliore molecola associata a quegli ID
    return GetBestMol(ids, name, namespace=namespace, domain=domain)


def FindSidsBySubstanceName(name: str) -> list[int]:
    """
    Questa funzione cerca una molecola su PubChem usando il nome come sostanza e restituisce una lista di SID associati a quella sostanza.
    Se la ricerca fallisce o non viene trovato nulla, restituisce None.
    """
    logger.debug(f"Cercando la molecola '{name}' su PubChem come sostanza")
    try:
        sids = pcp.get_sids(name, namespace="name", domain="substance")
    except Exception as e:
        logger.error(f"Connessione fallita per '{name}' come sostanza: {e}")
        return None
    if sids is None or len(sids) == 0:
        logger.warning(f"Nessuna substance trovata per '{name}'")
        return None

    return sids


def FindCidsBySubstanceName(name: str) -> Mol:
    """
    Questa funzione cerca una molecola su PubChem usando il nome come sostanza e restituisce una lista di CID associati a quella sostanza.
    Se la ricerca fallisce o non viene trovato nulla, restituisce None.
    """
    sids = FindSidsBySubstanceName(name)
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
    return cids


def FindMolByNameWithSynonyms(name: str, include_substances=True) -> Mol:
    """
    Questa funzione cerca una molecola su PubChem usando i sinonimi
    e restituisce il primo Mol trovato.
    Se la ricerca fallisce o non viene trovato nulla, restituisce None.
    """
    logger.debug(f"Cercando la molecola '{name}' su PubChem come sinonimo")
    synonyms = []
    try:
        synonyms = pcp.get_synonyms(name, namespace="name", domain="compound")
    except Exception as e:
        logger.error(f"Connessione fallita per '{name}' come sinonimo: {e}")
    if synonyms is None or len(synonyms) == 0:
        if include_substances:
            try:
                synonyms = pcp.get_synonyms(name, namespace="name", domain="substance")
            except Exception as e:
                logger.error(
                    f"Connessione fallita per '{name}' come sinonimo di sostanza: {e}"
                )
    if synonyms is None or len(synonyms) == 0:
        logger.warning(f"Nessun sinonimo trovato per '{name}'")

    # Synonyms e' una lista di dizionari, dove ogni dizionario ha la forma { "CID": 12345, "Synonym": ["...", "..."] }
    # Appiattiamo la lista di sinonimi in un'unica lista di stringhe
    flat_synonyms = []
    for entry in synonyms:
        if "Synonym" in entry:
            flat_synonyms.extend(entry["Synonym"])

    # Prova a cercare la molecola usando i sinonimi trovati
    for synonym in flat_synonyms:
        mol = FindMolByName(synonym)
        # Se troviamo una molecola valida usando un sinonimo, restituiamo quella molecola
        if mol is not None:
            SetMolName(mol, name)
            return mol

    logger.warning(f"Nessuna molecola trovata per '{name}' usando sinonimi")
    return None


class PubChemNamespace(Enum):
    CID = "cid"
    SID = "sid"


class PubChemDomain(Enum):
    COMPOUND = "compound"
    SUBSTANCE = "substance"


def GetBestMol(
    ids,
    name: str,
    namespace: PubChemNamespace.COMPOUND,
    domain: PubChemDomain.COMPOUND,
) -> Mol:
    """
    Questa funzione prende una lista di compound ID e restituisce la molecola con il maggior numero di atomi.
    Inoltre associa il nome alla molecola usando la proprietà `_Name`.
    Se il testo SDF è vuoto o non valido, restituisce None.
    """
    if ids is None or len(ids) == 0:
        logger.warning(f"Nessun ID valido trovato per '{name}'")
        return None

    # Otteniamo il testo SDF per gli ID dati
    try:
        sdf_text = pcp.get_sdf(ids, namespace=namespace.value, domain=domain.value)
    except Exception as e:
        logger.error(f"Errore durante il recupero dello SDF per '{name}': {e}")
        return None

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
        logger.success(
            f"Trovata molecola per '{name}' con {best_mol.GetNumAtoms()} atomi"
        )
        best_mol.SetProp("_Name", name)
    return best_mol


def GetBestMolFromSupplier(supplier: ForwardSDMolSupplier) -> Mol:
    """
    Questa funzione prende un ForwardSDMolSupplier e restituisce la molecola con il maggior numero di atomi.
    """
    ATOM_COUNT_MAX = 100  # Imposta un limite massimo di atomi per evitare di processare molecole troppo grandi
    ATOM_COUNT_MIN = 2  # Imposta un limite minimo di atomi per evitare di considerare molecole troppo semplici o non valide

    best_mol = None
    max_atoms = 0
    for mol in supplier:
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms and num_atoms <= ATOM_COUNT_MAX and num_atoms >= ATOM_COUNT_MIN:
                max_atoms = num_atoms
                best_mol = mol
    return best_mol


def GetMolName(mol):
    return mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"


def SetMolName(mol, name):
    mol.SetProp("_Name", name)


def GetMolProperty(mol: Mol, property_name: str):
    if mol.HasProp(property_name):
        return mol.GetProp(property_name)
    else:
        return None


def SetMolProperty(mol: Mol, property_name: str, value):
    mol.SetProp(property_name, value)


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


class FingerprintStrategy(Enum):
    MORGAN = "morgan"
    RDKIT = "rdkit"


# Definiamo una funzione che ci trasforma le molecole di un SDF in un vettore di fingerprint
# @input: SDMol
# @output: Vettore di fingerprint
def GetFingerprintFromSDMol(mols: SDMolSupplier, strategy: FingerprintStrategy = FingerprintStrategy.MORGAN) -> list:
    if strategy == FingerprintStrategy.MORGAN:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        msg = "Generando fingerprint Morgan"
    elif strategy == FingerprintStrategy.RDKIT:
        gen = rdFingerprintGenerator.GetRDKitFPGenerator()
        msg = "Generando fingerprint RDKit"
    fps = []
    for mol in tqdm(mols, msg):
        fp = gen.GetFingerprint(mol)
        fps.append(fp)
    return fps


def GetFingerprint(mol: Mol):
    """
    Restituisce la fingerprint della molecola passata come argomento in `mol`
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def GetFingerprintsFromMols(mols: list[Mol]) -> list:
    """
    Restituisce la lista delle fingerprint associate alle molecole della lista `mols`
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for mol in tqdm(mols, "Generando fingerprint"):
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
def GetButinaClustersFromDistanceMatrix(distance_matrix, n, dist_tresh):
    return Butina.ClusterData(
        distance_matrix, n, distThresh=dist_tresh, isDistData=True
    )


# Questa funzione espande una distance matrix da un formato Butina ad un formato pieno NxN,
# adatto ai metodi che lavorano su matrici di distanza complete.
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


def CompressDistanceMatrix(expanded_distance_matrix):
    """
    Converte una matrice delle distanze NxN nella forma compressa richiesta da Butina.
    """
    compressed_matrix = []
    n = len(expanded_distance_matrix)

    for j in range(1, n):
        for i in range(j):
            compressed_matrix.append(expanded_distance_matrix[i][j])

    return compressed_matrix


class MappingStrategy(Enum):
    """
    Strategie di riduzione della dimensionalità per generare coordinate 2D a partire da una matrice di distanze.
        - SAMMON: utilizza il Sammon mapping, che è una tecnica di riduzione della dimensionalità non lineare che cerca di preservare le piccole distanze.
        - MDS: utilizza il Multiple Dimensional Scaling, che è la tecnica standard di riduzione della dimensionalità lineare.
        - UMAP: utilizza il Uniform Manifold Approximation and Projection, che è un'ottima tecnica per la visualizzazione di cluster.
    """
    SAMMON = "sammon"
    MDS = "mds"
    UMAP = "umap"


class ClusteringStrategy(Enum):
    """
    Strategie di clustering per generare cluster a partire da coordinate 2D o da una matrice di distanze.
        - KMEANS: utilizza il clustering KMeans, che è la tecnica standard di clustering.
        - BUTINA: utilizza il clustering Butina, che è una tecnica di clustering basata sulla distanza.
        - AGGLOMERATIVE: utilizza il clustering agglomerativo, che è una tecnica di clustering gerarchico.
    """
    KMEANS = "kmeans"
    BUTINA = "butina"
    AGGLOMERATIVE = "agglomerative"


def GetClustersFromDistanceMatrix(
    mols: list[Mol],
    distance_matrix,
    cluster_count,
    random_state,
    mapping_strategy: MappingStrategy = MappingStrategy.SAMMON,
    cluster_strategy: ClusteringStrategy = ClusteringStrategy.KMEANS,
) -> list[ClusteredMol]:
    """
    Genera coordinate 2D e cluster a partire da una matrice delle distanze.

    input:
      - mols: lista di molecole su cui vogliamo generare i cluster
      - distance_matrix: matrice delle distanze completa NxN
      - cluster_count: numero di cluster da generare

    output:
      - Lista di ClusteredMol, che contiene le molecole clusterizzate e le loro coordinate 2D
    """
    clustered_mols = []
    coords_2d = []

    # Creiamo un array numpy dalla matrice di distanze
    np_distance_matrix = np.array(distance_matrix)

    if mapping_strategy == MappingStrategy.SAMMON:
        logger.info(
            "Eseguendo Sammon mapping per ridurre la matrice di distanze a 2 dimensioni"
        )
        np.random.seed(random_state)
        sammon_result = sammon(np_distance_matrix, inputdist="distance", init="")
        coords_2d = sammon_result[0]
    elif mapping_strategy == MappingStrategy.MDS:
        logger.info("Eseguendo MDS per ridurre la matrice di distanze a 2 dimensioni")
        mds = MDS(
            n_components=2,
            metric="precomputed",
            random_state=random_state,
            normalized_stress="auto",
            n_init=4,
            init="random",
        )
        coords_2d = mds.fit_transform(np_distance_matrix)
    elif mapping_strategy == MappingStrategy.UMAP:
        logger.info("Eseguendo UMAP per ridurre la matrice di distanze a 2 dimensioni")
        reducer = umap.UMAP(
            n_components=2, metric="precomputed", random_state=random_state
        )
        coords_2d = reducer.fit_transform(np_distance_matrix)
    else:
        raise ValueError(
            f"Strategia di riduzione della dimensionalità '{mapping_strategy}' non supportata"
        )

    clusters = []
    if cluster_strategy == ClusteringStrategy.KMEANS:
        logger.info("Eseguendo KMeans per generare {} cluster".format(cluster_count))
        km = KMeans(cluster_count, random_state=random_state)
        # KMeans fit predict restituisce un vettore di cluster a cui ogni molecola appartiene,
        # ad esempio [0, 0, 1, 1, 2] significa che le prime due molecole appartengono al cluster 0,
        # le successive due al cluster 1 e l'ultima al cluster 2
        clusters = km.fit_predict(coords_2d)
    elif cluster_strategy == ClusteringStrategy.BUTINA:
        logger.info("Eseguendo Butina per generare {} cluster".format(cluster_count))
        logger.warning(
            "Questa strategia ignora il numero di cluster specificato e utilizza una threshold di distanza per generare i cluster"
        )
        # Butina utilizza una threshold di distanza per generare i cluster, quindi non
        # possiamo specificare direttamente il numero di cluster che vogliamo,
        # ma possiamo giocare con la threshold per cercare di ottenere un numero di cluster simile a quello desiderato.
        dist_tresh = 0.85
        compressed_distance_matrix = CompressDistanceMatrix(distance_matrix)
        clusters = Butina.ClusterData(
            compressed_distance_matrix, len(mols), dist_tresh, isDistData=True
        )
        logger.info(
            f"Butina ha generato {len(clusters)} cluster con threshold {dist_tresh}"
        )
        # Butina restituisce una tupla di tuple, dove ogni tupla interna contiene gli indici delle molecole che appartengono a quel cluster.
        # Convertiamo la tupla di tuple in un vettore di cluster a cui ogni molecola appartiene
        cluster_vector = [0] * len(mols)
        for cluster_id, cluster in enumerate(clusters):
            for mol_index in cluster:
                cluster_vector[mol_index] = cluster_id
        clusters = cluster_vector
    elif cluster_strategy == ClusteringStrategy.AGGLOMERATIVE:
        logger.info(
            "Eseguendo Agglomerative Clustering per generare {} cluster".format(
                cluster_count
            )
        )
        agg = AgglomerativeClustering(
            n_clusters=cluster_count, metric="precomputed", linkage="average"
        )
        clusters = agg.fit_predict(np_distance_matrix)
    else:
        raise ValueError(f"Strategia di clustering '{cluster_strategy}' non supportata")

    # Ora abbiamo un vettore di cluster e un vettore di coordinate 2D per ogni molecola,
    # vogliamo combinarli in un unico contenitore che ci permetta di accedere facilmente a tutte queste informazioni.
    for i, (mol, cluster_id) in enumerate(zip(mols, clusters)):
        x, y = coords_2d[i]
        clustered_mols.append(
            ClusteredMol(mol=mol, cluster_id=int(cluster_id), x=float(x), y=float(y))
        )

    # Verifichiamno che clustered_mols e mols siano compatibili
    assert len(clustered_mols) == len(
        mols
    ), "Il numero di molecole clusterizzate non corrisponde al numero di molecole originali"
    for i in range(len(mols)):
        assert (
            clustered_mols[i].mol == mols[i]
        ), f"La molecola clusterizzata alla posizione {i} non corrisponde alla molecola originale"

    return clustered_mols


class ClusteredMol:
    """Classe che rappresenta una molecola clusterizzata, con informazioni utili per l'analisi e la visualizzazione"""

    @property
    def mol(self) -> Mol:
        """mol e' la molecola originale"""
        return self._mol

    @property
    def cluster_id(self) -> int:
        """cluster_id e' l'id del cluster a cui appartiene questa molecola"""
        return self._cluster_id

    @property
    def x(self) -> float:
        """x e' la coordinata x della molecola nello spazio 2D dell'embedding"""
        return self._x

    @property
    def y(self) -> float:
        """y e' la coordinata y della molecola nello spazio 2D dell'embedding"""
        return self._y

    def __init__(self, mol, cluster_id, x, y):
        self._mol = mol
        self._cluster_id = cluster_id
        self._x = x
        self._y = y

    def GetName(self):
        return GetMolName(self._mol)


class ClusterMCS:
    """Classe che rappresenta il risultato dell'MCS di un cluster, con informazioni utili per l'analisi e la visualizzazione"""

    @property
    def cluster_id(self) -> int:
        """cluster_id e' l'id del cluster a cui si riferisce questo MCS"""
        return self._cluster_id

    @property
    def mcs_smarts(self) -> str:
        """mcs_smarts e' la rappresentazione SMARTS della sottostruttura comune, che puo' essere usata per cercare questa sottostruttura in altre molecole"""
        return self._mcs_smarts

    @property
    def mcs_mol(self) -> Mol:
        """mcs_mol e' la rappresentazione Mol della sottostruttura comune"""
        return self._mcs_mol

    @property
    def mols_in_cluster(self) -> list[ClusteredMol]:
        """mols_in_cluster e' la lista di molecole che appartengono al cluster a cui si riferisce questo MCS"""
        return self._mols_in_cluster

    def __init__(self, cluster_id, mcs_smarts, mcs_mol, mols_in_cluster):
        self._cluster_id = cluster_id
        self._mcs_smarts = mcs_smarts
        self._mcs_mol = mcs_mol
        self._mols_in_cluster = mols_in_cluster

    def GetSize(self):
        """Restituisce il numero di molecole che appartengono al cluster a cui si riferisce questo MCS"""
        return len(self._mols_in_cluster)


def RenderColoredMolImage(
    mol: Mol, color, size: tuple[int, int] = (220, 220)
) -> np.ndarray:
    """
    Disegna la molecola come immagine RGBA con sfondo trasparente.
    Il colore viene applicato alla molecola stessa, non al frame.
    """
    mol_to_draw = Draw.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()

    options.clearBackground = False
    options.useBWAtomPalette()
    options.setSymbolColour(tuple(color[:3]))
    options.setLegendColour(tuple(color[:3]))
    options.fillHighlights = False
    options.continuousHighlight = False

    atom_ids = list(range(mol_to_draw.GetNumAtoms()))
    bond_ids = list(range(mol_to_draw.GetNumBonds()))
    atom_colors = {atom_id: tuple(color[:3]) for atom_id in atom_ids}
    bond_colors = {bond_id: tuple(color[:3]) for bond_id in bond_ids}
    atom_radii = {atom_id: 0.01 for atom_id in atom_ids}
    drawer.DrawMolecule(
        mol_to_draw,
        atom_ids,
        bond_ids,
        atom_colors,
        bond_colors,
        atom_radii,
        -1,
        "",
    )
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


def GetClusterCount(clustered_mols: list[ClusteredMol]) -> int:
    """Funzione che calcola il numero di cluster presenti in una lista di ClusteredMol"""
    cluster_ids = set([cm.cluster_id for cm in clustered_mols])
    return len(cluster_ids)


def GetClustersMCS(
    clustered_mols: list[ClusteredMol], mols: list[Mol]
) -> list[ClusterMCS]:
    """
    Funzione che genera MCSs a partire dai clusters

    input:
     - clustered_mols: una lista di ClusteredMol, che contiene le molecole clusterizzate e le loro coordinate 2D
     - mols: contenitore di molecole su cui vogliamo calcolare gli MCS, che puo' essere la lista originale di molecole o
        la lista dei murcko scaffold generici, a seconda di cosa vogliamo analizzare

    output:
     - Lista di ClusterMCS
    """
    cluster_results = []
    cluster_count = GetClusterCount(clustered_mols)
    for cluster_id in tqdm(
        range(cluster_count),
        total=cluster_count,
        desc="Calcolando MCS per ogni cluster",
    ):
        # Otteniamo gli indici delle molecole che appartengono al cluster i, filtrando la lista di ClusteredMol per cluster_id
        mol_indices_in_cluster = []
        clustered_mols_in_cluster = []
        for i, clustered_mol in enumerate(clustered_mols):
            if clustered_mol.cluster_id == cluster_id:
                mol_indices_in_cluster.append(i)
                clustered_mols_in_cluster.append(clustered_mol)

        if len(mol_indices_in_cluster) < 2:
            # salta i cluster con una sola molecola
            continue

        # trova la MCS per questo specifico cluster
        logger.debug(
            "Calcolando MCS per cluster {} con {} molecole".format(
                cluster_id, len(mol_indices_in_cluster)
            )
        )

        # otteniamo le molecole originali usando gli indici ottenuti e la lista di molecole passata come argomento
        mols_in_cluster = [mols[i] for i in mol_indices_in_cluster]
        mcs_res = rdFMCS.FindMCS(mols_in_cluster)

        # converti il risultato MCS in una molecola visualizzabile
        mcs_mol = MolFromSmarts(mcs_res.smartsString)

        # La mol ottenuto da MolFromSmarts non ha determinati valori che sono richiesti per il calcolo delle fingerprints
        # quindi, invochiamo `UpdatePropertyCache` e `FastFindRings` per forzare l'aggiornamento di questi valori.
        mcs_mol.UpdatePropertyCache(strict=False)
        FastFindRings(mcs_mol)

        # Nel ClusterMCS, salviamo anche la lista di ClusteredMol che appartengono a quel cluster,
        # in modo da poter accedere facilmente a tutte le informazioni su quelle molecole quando analizziamo o visualizziamo l'MCS
        cluster_results.append(
            ClusterMCS(
                cluster_id=cluster_id,
                mcs_smarts=mcs_res.smartsString,
                mcs_mol=mcs_mol,
                mols_in_cluster=clustered_mols_in_cluster,
            )
        )
    return cluster_results


def DrawClusterMCS(cluster_mcs: ClusterMCS, out_dir: str = "out/mcs"):
    """
    Funzione che disegna l'MCS di un cluster e lo salva in out/mcs come file PNG.
    Il nome del file e' cluster_{id}_mcs.png, dove {id} e' l'id del cluster.
    input: cluster_mcs - un oggetto ClusterMCS, ovvero il risultato della funzione GetClustersMCS per un singolo cluster
    """
    # Se la cartella out/mcs non esiste, creala
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mcs = cluster_mcs.mcs_mol
    id = cluster_mcs.cluster_id
    cluster_size = cluster_mcs.GetSize()
    legend = f"MCS Cluster {id} (size {cluster_size})"
    for clustered_mol in cluster_mcs.mols_in_cluster:
        legend += f"\n- {clustered_mol.GetName()}"

    DrawMol(mcs, out_dir, name_prefix=f"cluster_{id}_mcs", legend=legend)


def PlotClustersMCS(clusters_mcs: list[ClusterMCS], out_dir: str = "out/mcs"):
    """
    Funzione che disegna un grafico di tutte le molecole che appartengono a quel cluster,
    usando le coordinate 2D dell'embedding e colorando le molecole in base al cluster.
    Il grafico viene salvato in `out_dir` come file PNG con nome clusters_mcs.png
    """

    # Se la cartella out/mcs non esiste, creala
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    xs = []
    ys = []

    for cluster_mcs in clusters_mcs:
        for clustered_mol in cluster_mcs.mols_in_cluster:
            color = cmap(clustered_mol.cluster_id % cmap.N)

            mol_img = RenderColoredMolImage(clustered_mol.mol, color, size=(180, 180))
            image_box = OffsetImage(mol_img, zoom=0.25)
            ab = AnnotationBbox(
                image_box,
                (clustered_mol.x, clustered_mol.y),
                frameon=False,
                pad=0.0,
                zorder=2,
            )
            ax.add_artist(ab)

            xs.append(clustered_mol.x)
            ys.append(clustered_mol.y)

            ax.text(
                clustered_mol.x,
                clustered_mol.y,
                clustered_mol.GetName(),
                fontsize=8,
                ha="center",
                va="bottom",
                color=color,
                zorder=3,
            )

    x_margin = max((max(xs) - min(xs)) * 0.1, 0.25) if xs else 1.0
    y_margin = max((max(ys) - min(ys)) * 0.1, 0.25) if ys else 1.0

    if xs and ys:
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

    ax.set_title("Molecole nei cluster MCS")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/clusters_mcs.png", dpi=300, bbox_inches="tight")
    plt.close()


def DrawClustersMCS(clusters_mcs: list[ClusterMCS], out_dir: str = "out/mcs"):
    """
    Funzione che disegna gli MCS di ogni cluster e li salva nella cartella specificata come file PNG.
    Il nome del file e' cluster_{id}_mcs.png, dove {id} e' l'id del cluster.

    Inoltre, questa funzione disegna un grafico di tutte le molecole che appartengono a quel cluster,
    usando le coordinate 2D dell'embedding e colorando le molecole in base al cluster.

    - input: `clusters_mcs` - risultato della funzione GetClustersMCS
    - input: `out_dir` - percorso della cartella in cui salvare i file PNG
    """
    # Cancella la cartella out/mcs se esiste per eliminare i vecchi risultati
    if os.path.exists(out_dir):
        for file in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, file))

    for cluster_mcs in clusters_mcs:
        DrawClusterMCS(cluster_mcs, out_dir=out_dir)

    PlotClustersMCS(clusters_mcs, out_dir=out_dir)


def DebugMol(mol: Mol):
    """
    Funzione di debug che stampa informazioni utili sulla molecola, come il nome, il numero di atomi e di legami.
    """
    name = GetMolName(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    logger.debug(f"Molecola '{name}': {num_atoms} atomi, {num_bonds} legami")


def DrawMols(mols, out_dir: str = "out/mols"):
    """
    Disegna le molecole e salva il risultato in file PNG nella cartella `out_dir`
    """
    for mol in tqdm(mols, desc="Disegnando le molecole"):
        DebugMol(mol)
        DrawMol(mol, out_dir=out_dir)



def DrawMol(
    mol, out_dir: str = "out/mols", name_prefix: str = "mol", legend: str = None
):
    name = GetMolName(mol)

    if mol is None or mol.GetNumAtoms() == 0:
        logger.warning(f"Molecola '{name}' nulla o senza atomi, salto il disegno")
        return

    def _draw(mol_to_draw):
        return Draw.MolToImage(
            mol_to_draw,
            size=(600, 600),
            legend=legend if legend else name,
        )

    try:
        img = _draw(mol)
    except Exception as e:
        logger.warning(
            f"Prima conversione immagine fallita per '{name}': {e}, provo a calcolare coordinate 2D e riprovo"
        )
        try:
            rdDepictor.Compute2DCoords(mol)
            img = _draw(mol)
        except Exception as e2:
            logger.error(f"Errore durante il disegno della molecola '{name}' dopo Compute2DCoords: {e2}")
            return

    # Se la cartella out non esiste, creala
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Name potrebbe avere il carattere di separatore di cartella al suo interno `/`
    # Quindi lo dobbiamo safinicare se vogliamo usarlo come nome di file
    safe_name = name.replace("/", "_")
    # Salva l'immagine in out/mol_{name}.png
    img.save(f"{out_dir}/{name_prefix}_{safe_name}.png")


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


def GetClustersMCSFromMols(
    mols: list[Mol],
    use_scaffolds: bool = True,
    cluster_count=20,
    random_state=42,
    mapping_strategy: MappingStrategy = MappingStrategy.SAMMON,
    cluster_strategy: ClusteringStrategy = ClusteringStrategy.KMEANS,
    fingerprint_strategy: FingerprintStrategy = FingerprintStrategy.MORGAN,
) -> list[ClusterMCS]:
    """Funzione che ottiene i cluster e i relativi MCS a partire da una lista di molecole

    - input: mols - lista di molecole (formato Mol)
    - output: lista di ClusterMCS, ovvero una lista di oggetti che rappresentano i cluster e i relativi MCS, con informazioni utili per l'analisi e la visualizzazione

    La funzione esegue i seguenti passaggi:
    1. Ottiene i fingerprint per ogni molecola
    2. Ottiene la matrice di distanze a partire dai fingerprint
    3. Utilizza la strategia di clustering selezionata per ottenere i cluster a partire dalla matrice di distanze
    4. Per ogni cluster, ottiene l'MCS a partire dalle molecole che appartengono a quel cluster, e salva le informazioni in un oggetto ClusterMCS
    """

    # Otteniamo le fingerprint per ogni molecola
    fingerprints = GetFingerprintFromSDMol(mols, fingerprint_strategy)
    # Otteniamo la matrice di distanze
    compressed_distance_matrix = GetDistanceMatrixFromFingerprints(fingerprints)

    expended_distance_matrix = ExpandDistanceMatrix(
        compressed_distance_matrix, len(fingerprints)
    )

    clustered_mols = GetClustersFromDistanceMatrix(
        mols,
        expended_distance_matrix,
        cluster_count,
        random_state=random_state,
        mapping_strategy=mapping_strategy,
        cluster_strategy=cluster_strategy,
    )

    if use_scaffolds:
        # L'MCS trova la più grande sottostruttura comune tra due o più molecole.
        # Cercare l'MCS su molecole intere potrebbe dare un risultato troppo
        # piccolo o "sporcato" dalle catene laterali.
        # Usare Bemis-Murcko prima dell'MCS permette di trovare il nucleo comune degli scheletri,
        # ignorando le variazioni dei sostituenti.
        scaffolds = GetScaffoldsFromSDMol(mols)

        # Se invece vogliamo concentrarci solo sulla topologia dello scheletro, ignorando
        # anche i tipi di atomi e di legami, possiamo usare MakeScaffoldGeneric
        generic_scaffolds = MakeScaffoldsGeneric(scaffolds)

        # otteniamo gli MCS di ogni cluster
        return GetClustersMCS(clustered_mols, scaffolds)
    else:
        return GetClustersMCS(clustered_mols, mols)
