import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Gender survival rate
gender_survival = df.groupby('Sex')['Survived'].mean()

# Age group survival rate
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 60, 100], labels=['Child', 'Adult', 'Senior'])
age_survival = df.groupby('Age_Group')['Survived'].mean()

# Create a bar plot for gender survival rate
plt.figure(figsize=(12, 5))

# Plot 1: Gender Survival Rate
plt.subplot(1, 2, 1)
sns.barplot(x=gender_survival.index, y=gender_survival.values, palette='pastel')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')

# Plot 2: Age Group Survival Rate
plt.subplot(1, 2, 2)
sns.barplot(x=age_survival.index, y=age_survival.values, palette='muted')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')

# Display plots
plt.tight_layout()
plt.show()
