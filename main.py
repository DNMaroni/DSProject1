import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('StudentsPerformance.csv')

df_selected = data[['parental level of education', 'math score', 'reading score', 'writing score']]

column_translation = {
    "bachelor's degree": "bacharelado",
    'some college': 'superior incompleto',
    "master's degree": "mestrado",
    "associate's degree": "técnico",
    'high school': 'ensino médio',
    'some high school': 'ensino médio incompleto'
}

df_selected['parental level of education'] = df_selected['parental level of education'].map(column_translation)

df_selected['average_score'] = df_selected[['math score', 'reading score', 'writing score']].mean(axis=1)

df_encoded = pd.get_dummies(df_selected['parental level of education'], prefix='parental level of education')

df_encoded['average_score'] = df_selected['average_score']

X = df_encoded.drop('average_score', axis=1)
y = df_encoded['average_score']

model = LinearRegression()

model.fit(X, y)

education_levels = df_selected['parental level of education'].unique()
df_predictions = pd.DataFrame({'parental level of education': education_levels})
df_predictions = pd.concat([df_predictions, pd.get_dummies(df_predictions['parental level of education'], prefix='parental level of education')], axis=1)
df_predictions['average_score'] = model.predict(df_predictions.drop('parental level of education', axis=1))

plt.figure(figsize=(10, 6))
plt.bar(df_predictions['parental level of education'], df_predictions['average_score'])
plt.xlabel('Nível de educação dos pais')
plt.ylabel('Média de nota dos filhos')
plt.title('Relação entre o nível de educação dos pais com a média de notas dos filhos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
