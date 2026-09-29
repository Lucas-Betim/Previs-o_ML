# Classificação de Score de Crédito com Machine Learning

## Sobre o projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de classificar o **score de crédito de clientes** com base em informações financeiras e comportamentais.

A base utilizada possui aproximadamente **100 mil registros** e contém dados como idade, profissão, salário anual, quantidade de contas e cartões, histórico de crédito, atrasos, empréstimos, comportamento de pagamento e outras informações financeiras.

O problema foi tratado como uma tarefa de **classificação supervisionada**, em que o modelo deve prever uma das categorias de score de crédito disponíveis na base.

---

## Objetivo

Criar e comparar modelos de classificação capazes de prever automaticamente o score de crédito de um cliente a partir de suas características.

As principais etapas foram:

- importação e análise da base de dados;
- preparação das variáveis;
- transformação de dados categóricos;
- separação entre dados de treino e teste;
- treinamento de modelos de Machine Learning;
- comparação dos resultados;
- previsão para novos clientes.

---

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Random Forest
- K-Nearest Neighbors (KNN)
- Jupyter Notebook

---

# Base de dados

O dataset possui aproximadamente **100.000 registros e 25 colunas**.

Entre as variáveis utilizadas estão:

- idade;
- profissão;
- salário anual;
- quantidade de contas;
- quantidade de cartões;
- juros de empréstimos;
- número de empréstimos;
- dias de atraso;
- quantidade de pagamentos atrasados;
- verificações de crédito;
- dívida total;
- taxa de utilização de crédito;
- histórico de crédito;
- investimento mensal;
- comportamento de pagamento;
- saldo final do mês;
- tipos de empréstimos.

A variável que será prevista é:

```text
score_credito
```

As categorias presentes na base incluem classificações como:

```text
Good
Standard
Poor
```



---

# Etapas do projeto

## 1. Importação da base

A base de dados é carregada utilizando Pandas:

```python
import pandas as pd

tabela = pd.read_csv("clientes.csv")
```

Em seguida, são analisadas as informações da tabela e os tipos de cada coluna.

---

## 2. Tratamento das variáveis categóricas

Algumas colunas possuem valores em texto e precisam ser convertidas para valores numéricos antes de serem utilizadas pelos modelos.

Foi utilizado `LabelEncoder` para as colunas:

- `profissao`;
- `mix_credito`;
- `comportamento_pagamento`.

Exemplo:

```python
from sklearn.preprocessing import LabelEncoder

codificador_profissao = LabelEncoder()
tabela["profissao"] = codificador_profissao.fit_transform(
    tabela["profissao"]
)
```

O mesmo processo foi aplicado às demais variáveis categóricas.

---

## 3. Definição das variáveis de entrada e saída

A variável que será prevista é armazenada em `y`:

```python
y = tabela["score_credito"]
```

As demais variáveis utilizadas pelo modelo ficam em `x`:

```python
x = tabela.drop(
    columns=["score_credito", "id_cliente"]
)
```

A coluna `id_cliente` é removida porque funciona apenas como identificador e não representa uma característica relevante para a previsão.

---

## 4. Divisão entre treino e teste

A base é dividida em dados de treino e teste utilizando `train_test_split`.

```python
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2
)
```

Nesse projeto:

- **80%** dos dados são utilizados para treinamento;
- **20%** são utilizados para avaliação dos modelos.



---

# Modelos utilizados

Foram comparados dois algoritmos de classificação.

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

modelo_rf = RandomForestClassifier()
modelo_rf.fit(x_treino, y_treino)
```

## K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

modelo_knn = KNeighborsClassifier()
modelo_knn.fit(x_treino, y_treino)
```



---

# Avaliação dos modelos

Após o treinamento, os modelos foram utilizados para prever os dados de teste.

```python
previsao_rf = modelo_rf.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)
```

A métrica utilizada para comparação foi a **acurácia**:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_teste, previsao_rf)
accuracy_score(y_teste, previsao_knn)
```

Os resultados obtidos foram aproximadamente:

| Modelo | Acurácia |
|---|---:|
| Random Forest | 82,88% |
| KNN | 74,87% |

O **Random Forest** apresentou o melhor resultado entre os dois modelos testados.

---

# Previsão de novos clientes

Após a escolha do modelo com melhor desempenho, uma nova base contendo clientes sem classificação foi importada.

```python
novos_clientes = pd.read_csv(
    "novos_clientes.csv"
)
```

As variáveis categóricas também foram transformadas e, em seguida, o modelo Random Forest foi utilizado para realizar as previsões:

```python
previsao = modelo_rf.predict(
    novos_clientes
)
```

Dessa forma, o modelo consegue receber os dados de novos clientes e gerar automaticamente uma classificação de score de crédito.

---

# Conceitos praticados

Este projeto trabalha conceitos como:

- Machine Learning supervisionado;
- classificação;
- preparação de dados;
- transformação de variáveis categóricas;
- separação entre treino e teste;
- treinamento de modelos;
- avaliação de desempenho;
- comparação de algoritmos;
- realização de novas previsões.

---

# Possíveis melhorias futuras

Algumas possíveis evoluções do projeto:

- utilizar outras métricas além da acurácia;
- gerar matriz de confusão;
- avaliar Precision, Recall e F1-Score;
- realizar validação cruzada;
- testar outros algoritmos;
- aplicar ajuste de hiperparâmetros;
- analisar importância das variáveis;
- utilizar pipelines do Scikit-learn;
- salvar o modelo treinado para utilização futura.

---

#  Estrutura sugerida

```text
credit-score-ml/
│
├── projeto.ipynb
├── clientes.csv
├── novos_clientes.csv
├── requirements.txt
└── README.md
```

---

# Tecnologias e conceitos

**Python • Pandas • Scikit-learn • Random Forest • KNN • Machine Learning • Classificação • Análise de Dados**

---

## Conclusão

Neste projeto foi desenvolvido um fluxo completo básico de Machine Learning para classificação de score de crédito.

Foram realizadas etapas de preparação dos dados, treinamento, comparação entre modelos e previsão de novos registros.

Entre os algoritmos testados, o **Random Forest apresentou maior acurácia**, alcançando aproximadamente **82,9%** no conjunto de teste.
