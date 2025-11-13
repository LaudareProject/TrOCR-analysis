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

    # Campioniamo per performance
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=1)
    else:
        df_sample = df

    # Variabili da plottare
    y_var = 'token_loss'
    x_vars = [
        'attention_entropy',
        'gradcam_entropy',
        'attention_gini',
        'gradcam_gini',
        'attention_coverage',
        'gradcam_coverage'
    ]

    y_var_formatted = format_label(y_var) # "Token Loss"

    print("Inizio generazione dei 6 scatter plot...")

    # Loop per creare ciascun grafico
    for x_var in x_vars:
        x_var_formatted = format_label(x_var)
        filename = f'seaborn_scatter_{y_var}_vs_{x_var}.png'

        plt.figure(figsize=(10, 6))

        sns.regplot(
            data=df_sample,
            x=x_var,
            y=y_var,
            scatter_kws={'alpha': 0.3, 's': 10},
            line_kws={'color': 'red'}
        )

        plt.xlabel(x_var_formatted)
        plt.ylabel(y_var_formatted)
        plt.title(None) # Rimuovi titolo
        plt.tight_layout()

        plt.savefig(filename)
        print(f"Grafico generato: {filename}")

    print("\nGenerazione dei 6 plot completata.")

except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
except Exception as e:
    print(f"Si è verificato un errore durante l'analisi: {e}")
