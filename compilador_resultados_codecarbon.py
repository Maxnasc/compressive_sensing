import pandas as pd

df_emissions_CoSaMP_com_dct = pd.read_csv("emissions_CoSaMP_com_dct.csv")
df_emissions_CoSaMP_com_dft = pd.read_csv("emissions_CoSaMP_com_dft.csv")
df_emissions_OMP_com_dct_3 = pd.read_csv("emissions_OMP_com_dct_3.csv")
df_emissions_OMP_com_dct = pd.read_csv("emissions_OMP_com_dct.csv")
df_emissions_OMP_com_dft_1 = pd.read_csv("emissions_OMP_com_dft_1.csv")
df_emissions_OMP_com_dft_2 = pd.read_csv("emissions_OMP_com_dft_2.csv")
df_emissions_OMP_com_dft_3 = pd.read_csv("emissions_OMP_com_dft_3.csv")
df_emissions_SL0_com_dct = pd.read_csv("emissions_SL0_com_dct.csv")

# A correção está aqui: usamos .tail(1) para obter um DataFrame de uma linha
df_concatenado = pd.concat(
    [
        df_emissions_CoSaMP_com_dct.tail(1),
        df_emissions_CoSaMP_com_dft.tail(1),
        df_emissions_OMP_com_dct_3.tail(1),
        df_emissions_OMP_com_dct.tail(1),
        df_emissions_OMP_com_dft_1.tail(1),
        df_emissions_OMP_com_dft_2.tail(1),
        df_emissions_OMP_com_dft_3.tail(1),
        df_emissions_SL0_com_dct.tail(1),
    ],
    ignore_index=True,  # Usado para resetar o índice
)

# Adicionando a coluna indicando qual otimizador e dicionário
otm_and_dict = [
    "CoSaMP_com_dct.csv",
    "CoSaMP_com_dft.csv",
    "OMP_com_dct_3.csv",
    "OMP_com_dct.csv",
    "OMP_com_dft_1.csv",
    "OMP_com_dft_2.csv",
    "OMP_com_dft_3.csv",
    "SL0_com_dct.csv",
]
df_concatenado.insert(loc=0, column="Abordagem", value=otm_and_dict)

# Salvando a avaliação completa
df_concatenado.to_csv('emissions_comparation.csv')

# Dropando colunas indesejadas
columns_to_stay = ['Abordagem', 'duration', 'emissions', 'emissions_rate', 'cpu_energy', 'gpu_energy', 'ram_energy', 'energy_consumed']
df_f = df_concatenado[columns_to_stay]
df_f.to_csv('emissions_comparation_filtered.csv')

print(df_concatenado)
