import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np

# Ignora avvisi
warnings.filterwarnings('ignore')

# Funzione per formattare le etichette
def format_label(col_name):
    if col_name == 'image_cer':
        return 'Image CER'
    # Rimuovi underscore, usa Title Case
    return col_name.replace('_', ' ').title()

# Carica il dataset
file_path = 'combined_token_results.csv'

try:
    df = pd.read_csv(file_path)
    print("File CSV caricato con successo.")

    # --- Plot 1: Heatmap di Correlazione (Seaborn) ---
    print("Generazione Plot 2 (Heatmap)...")

    # Definisci le colonne da includere
    # Colonne numeriche originali meno quelle da escludere
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['position', 'token_id', 'sample_id', 'gradcam_peak', 'attention_peak']
    cols_for_heatmap = [col for col in all_numeric_cols if col not in cols_to_exclude]

    # Crea il dizionario per rinominare
    rename_dict = {col: format_label(col) for col in cols_for_heatmap}

    # Calcola matrice di correlazione sul df originale
    correlation_matrix = df[cols_for_heatmap].corr()

    # Rinomina righe e colonne della matrice
    correlation_matrix_renamed = correlation_matrix.rename(index=rename_dict, columns=rename_dict)

    # Riordina le colonne/righe per mettere 'Image CER' vicino a 'Token Loss'
    original_renamed_order = list(correlation_matrix_renamed.columns)

    # Rimuovi 'Image CER' dalla sua posizione attuale (se esiste)
    if 'Image CER' in original_renamed_order:
        original_renamed_order.remove('Image CER')

    # Inseriscilo dopo 'Token Loss' (o all'inizio se 'Token Loss' non c'è)
    try:
        token_loss_index = original_renamed_order.index('Token Loss')
        new_order = original_renamed_order[:token_loss_index+1] + ['Image CER'] + original_renamed_order[token_loss_index+1:]
    except ValueError:
        # Se 'Token Loss' non è presente, metti 'Image CER' all'inizio
        new_order = ['Image CER'] + original_renamed_order

    # Applica il nuovo ordine
    correlation_matrix_final = correlation_matrix_renamed.reindex(index=new_order, columns=new_order)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix_final,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        annot_kws={"size": 10} # Riduci dimensione font per leggibilità
    )
    plt.title(None) # Rimuovi titolo
    plt.xticks(rotation=45, ha='right') # Ruota etichette asse x per leggibilità
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('seaborn_plot_2_heatmap.png')
    print("Plot: seaborn_plot_2_heatmap.png")

except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
except Exception as e:
    print(f"Si è verificato un errore durante l'analisi: {e}")
