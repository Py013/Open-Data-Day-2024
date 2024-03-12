import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


kpis = pd.read_excel('./idicadores_educacionais.xlsx')

kpis.head(10)


def plot_analysis(df, colunas):
    for coluna in colunas:
        if coluna in df.columns and pd.api.types.is_numeric_dtype(df[coluna]):
            fig, axs = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 3, 3, 1]})

            # Gráfico de dispersão
            axs[0].scatter(df.index, df[coluna])
            axs[0].set_title(f'Dispersão, Histograma e Boxplot de {coluna}')
            
            # Histograma
            sns.histplot(df[coluna], kde=True, ax=axs[1])            
            # Boxplot
            sns.boxplot(x=df[coluna], ax=axs[2])

            stats = df[coluna].describe().round(2)
            axs[3].axis('tight')
            axs[3].axis('off')
            table = axs[3].table(cellText=stats.values.reshape(-1, 1), 
                                 rowLabels=stats.index, 
                                 loc='center',
                                 colWidths=[0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            plt.tight_layout()
            print(f"{coluna}:")
            plt.show()
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame ou não é numérica.")

columns = ['afd', 'dsu', 'had','ied', 'ird', 'tdi', 'tx_rend_abandono', 'tx_rend_aprovacao','tx_rend_reprovacao']
plot_analysis(kpis, columns)


correlacao = kpis[columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de Correlação dos Indicadores Educacionais')
plt.show()


municipios_unicos = kpis[['municipio_id', 'municipio_name']].drop_duplicates().reset_index(drop=True)
dfs_for_city = {}
for index, row in kpis[['municipio_id', 'municipio_name']].drop_duplicates().iterrows():
    cidade_nome = row['municipio_name'].upper().replace(' ', '_')
    df_nome = f'kpi_{cidade_nome}'
    dfs_for_city[df_nome] = kpis[kpis['municipio_id'] == row['municipio_id']]
df_cities = list(dfs_for_city.keys())
df_cities



def plot_city_correlations(dfs, columns):
    for df_name, df in dfs.items():
        cidade_nome = df_name.replace('kpi_', '').replace('_', ' ').title()
        correlacao = df[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlacao, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'Mapa de Correlação dos Indicadores Educacionais - {cidade_nome}')
        plt.show()
plot_city_correlations(dfs_for_city, columns)




kpis_clean = kpis.dropna(subset=['tx_rend_aprovacao'])
X = kpis_clean.select_dtypes(include=['number']).drop(columns=['tx_rend_aprovacao'])
X_filled = X.fillna(X.mean())
y = kpis_clean['tx_rend_aprovacao']
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r_squared = model.score(X_test, y_test)
print(f"R²: {r_squared}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Valores Reais vs. Valores Previstos')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Linha de perfeita previsão
plt.show()