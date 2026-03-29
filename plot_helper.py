import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


def PlotMatrixCSV(csv_path: str, png_path: str, title: str = None):
    """Disegna la matrice rappresentata dal file CSV utilizzando pyplot

    - csv_path: Percorso del file CSV in input
    - png_path: Percorso del'immagine PNG da creare in output
    """

    # 1. Legge il file CSV
    # index_col=0 poiche' la prima colonna contiene i nomi delle molecole
    df = pd.read_csv(csv_path, index_col=0)

    # 2. Controllo rapido del contenuto
    print(df.head())
    print(df.shape)

    # 3. Crea il grafico
    plt.figure(figsize=(16, 16))
    matrix = df.values
    img = plt.imshow(
        matrix,
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    # Barra dei colori
    plt.colorbar(img, label="Distanza")

    # 4. Personalizzazione
    plot_title = title
    if plot_title is None:
        plot_title = csv_path

    plt.title(title)

    # Se vuoi mostrare i nomi sugli assi
    plt.xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(df.index)), df.index, fontsize=8)

    # Salva immagine
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    logger.info("Immagine PNG creata in %s" % png_path)


def main():
    PlotMatrixCSV(
        "data/matrice_distanza_fmcs.csv",
        "out/matrice_distanza_fmcs.png",
        "Matrice Distanza FMCS",
    )


if __name__ == "__main__":
    main()
