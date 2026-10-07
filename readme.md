# Análise de Componentes Principais (PCA) dos Distritos de São Paulo

Este projeto realiza um estudo de **Análise de Componentes Principais (PCA)** e **Análise Fatorial** com dados socioeconômicos e demográficos de 96 distritos do município de São Paulo.

O objetivo principal é extrair fatores sintéticos que sintetizem a qualidade de vida, o nível socioeconômico e a vulnerabilidade social dos distritos paulistas.

---

## 📌 Principais Etapas da Análise

1. **Pré-processamento e Padronização**:
   - Padronização das variáveis via **Z-Score** (`StandardScaler`) devido às diferentes escalas de grandeza.

2. **Testes de Adequação da Análise Fatorial**:
   - **Kaiser-Meyer-Olkin (KMO)**: Resultado geral de $\approx 0{,}83$, indicando alta adequação da amostragem para a análise fatorial.
   - **Teste de Esfericidade de Bartlett**: Com $p < 0{,}05$ ($p \approx 9{,}27 \times 10^{-136}$), confirmou-se que a matriz de correlação é significativamente diferente da matriz identidade.

3. **Extração de Fatores e Critério de Kaiser**:
   - **Fator 1 (F1)**: Explica **$55{,}07\%$** da variância total. Fortemente associado a variáveis como *renda*, *escolaridade*, *idade* e *quota*. Representa a dimensão de **Desenvolvimento Socioeconômico**.
   - **Fator 2 (F2)**: Explica **$12{,}94\%$** da variância total. Fortemente correlacionado com *densidade populacional* e *favelização*. Representa a dimensão de **Vulnerabilidade e Densidade Urbana**.
   - **Variância Explicada Acumulada**: Os dois primeiros fatores explicam **$68{,}00\%$** da informação original.

4. **Visualização de Resultados**:
   - Mapeamento e dispersão gráfica dos 96 distritos com base nos scores fatoriais de $F1$ e $F2$.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas** e **NumPy**: Manipulação e tratamento de dados.
- **Scikit-Learn**: Padronização dos dados e execução do algoritmo PCA.
- **Factor Analyzer**: Testes estatísticos de KMO e Bartlett.
- **Plotly Express**: Criação dos gráficos interativos dos scores fatoriais.

---

## 📈 Código para Gerar os Scores Fatoriais

```python
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Seleção das variáveis socioeconômicas
colunas_numericas = [
    'renda',
    'quota',
    'escolaridade',
    'idade',
    'mortalidade',
    'txcresc',
    'causasext',
    'favel',
    'denspop',
]

# 2. Padronização por Z-Score
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(distritos_sp[colunas_numericas])

# 3. Análise de Componentes Principais
pca = PCA(n_components=len(colunas_numericas))
scores_fatoriais = pca.fit_transform(dados_padronizados)

# Dataframe com os scores dos 2 primeiros fatores
df_scores = pd.DataFrame(scores_fatoriais[:, :2], columns=['F1', 'F2'])
df_scores['distritos'] = distritos_sp['distritos']

# 4. Gráfico de dispersão F1 x F2
fig = px.scatter(
    df_scores,
    x='F1',
    y='F2',
    text='distritos',
    title='Mapeamento dos Distritos de SP (F1 vs F2)',
    hover_data=['distritos'],
)

fig.update_traces(textposition='top center', marker=dict(size=8, color='navy'))
fig.update_layout(
    xaxis=dict(title='Fator 1 (Desenvolvimento / Renda / Escolaridade)'),
    yaxis=dict(title='Fator 2 (Densidade Populacional / Vulnerabilidade)'),
    height=700,
)

fig.show()
```