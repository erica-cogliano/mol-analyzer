# Obiettivo: filtrare il file "data/ttd_drug_disease_by_drug.csv" per rimuovere tutte 
# le molecole che non sono state approvate o che appartengono alla classe k ovarico, 
# e salvare il risultato in un nuovo file "data/ttd_approved_non_k_ovarian_drugs.csv"

# Steps:
# 1. Leggere il file "data/ttd_drug_disease_by_drug.csv" usando pandas
# 2. Filtrare le righe per mantenere solo quelle in cui:
#    - la colonna "approval categories" contiene "approved". E' importante notare che il
#      valore potrebbe essere una stringa che contiene più categorie di approvazione 
#      (e.g "approved; phase 3"), quindi dobbiamo verificare se "approved" è presente in quella stringa.
#    - e la colonna "icd11_codes" non è relativa al k ovarico, ovvero non contiene "2C73"
# 3. Salvare il risultato in un nuovo file "data/ttd_approved_non_k_ovarian_drugs.csv"

ALL_DRUGS_CSV_FILE_PATH = "data/ttd_drug_disease_by_drug.csv"
APPROVED_NON_K_OVARIAN_CSV_FILE_PATH = "data/ttd_approved_non_k_ovarian_drugs.csv"

APPROVAL_CATEGORIES_COLUMN_NAME = "approval_categories"
ICD11_CODES_COLUMN_NAME = "icd11_codes"
APPROVED_KEYWORD = "approved"
K_OVARIAN_ICD11_CODE = "2C73"

import pandas as pd


def FilterApprovedNonKOvarianDrugs(input_csv_file_path: str, output_csv_file_path: str):
    # Leggi il file CSV
    df = pd.read_csv(input_csv_file_path)
    
    # Filtra le righe
    filtered_df = df[
        # Prendi i farmaci approvati
        df[APPROVAL_CATEGORIES_COLUMN_NAME].str.contains(APPROVED_KEYWORD, case=False, na=False) &

        # Ma escludi quelli che sono relativi al k ovarico
        ~df[ICD11_CODES_COLUMN_NAME].str.contains(K_OVARIAN_ICD11_CODE, case=False, na=False)
    ]
    
    # Salva il risultato in un nuovo file CSV
    filtered_df.to_csv(output_csv_file_path, index=False)


# La particolarita' di `if __name__ == "__main__":` e' che permette di eseguire il codice solo 
# quando il file viene eseguito direttamente, e non quando viene importato come modulo in un altro file.
# In questo modo, possiamo definire la funzione `FilterApprovedNonKOvarianDrugs` in questo file, e poi 
# importarla e usarla in altri file senza eseguire il codice di filtraggio ogni volta che importiamo questo modulo.
if __name__ == "__main__":
    FilterApprovedNonKOvarianDrugs(ALL_DRUGS_CSV_FILE_PATH, APPROVED_NON_K_OVARIAN_CSV_FILE_PATH)
