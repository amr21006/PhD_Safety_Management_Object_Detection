import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv(“survey_responses.csv”) # Replace “survey_responses.csv” with your actual file name

# Data Preprocessing (Encoding Categorical Variables)
for col in ['decision_making', 'ppe', 'safety_inspection', 'satisfaction', 'alarm_system_implementation', 'confidence', 'benefits', 'accidents', 'negative_impact', 'cost_effectiveness', 'efficiency', 'features', 'technologies', 'integration_importance', 'challenges', 'feedback']:
    df[col] = df[col].astype(“category”).cat.codes

df[“involved_in_safety_decisions”] = df[“Are you involved in making decisions about safety management on construction sites?   ”].map({"Strongly Agree":1, "Agree":.75, “Neutral”:.5, “Disagree”:.25, “Strongly Disagree”:0})
df[“satisfaction_with_safety_practices”] = df[“How satisfied are you with your current safety practices?\n\nSpecifically, we are referring to measures related to hazard identification, and use of safety equipment.”].map({"Very satisfied":1, "Somewhat satisfied":.75, “Neutral”:.5, “Somewhat dissatisfied”:.25, “Very dissatisfied”:0})
df[“likelihood_of_implementing_alarm_system”] = df[“How likely are you to consider implementing real time alarm systems on your construction site in the future?”].map({"Extremely likely":1, "Very likely":75, “Moderately likely”:.5, “Somewhat likely”:.25, “Not at all likely”:0})
df[“experience_in_years”] = df[“How many years of experience do you have in the construction industry ”].map({"0-5 years":5, "6-15 years":10, “16-25 years”:20, “More than 25 years”:25})
df[“confidence_in_AI”] = df[“How confident are you in the ability of real time alarm systems to increase safety on construction sites?”].map({"Extremely confident":1, "Very confident":.75, “Moderately confident”:.5, “Somewhat confident”:.25, “Not at all confident”:0})
df[“accident_occure”] = df[“If real time monitoring systems are not utilized on construction sites, how likely are accidents or near-misses to occur?”].map({"Extremely likely":1, "Very likely":.75, “Moderately likely”:.5, “Somewhat likely”:.25, “Not at all likely”:0})
df[“importance_of_real time_AI”] = df[“How important is the ability to integrate with existing safety systems for a real-time alarm system in construction? (Select one)”].map({"Extremely important":1, "Very important":.75, “Moderately important”:.5, “Somewhat important”:.25, “Not at all important”:0})

# --- Section 1: Chi-Square Tests ---
categorical_vars = [“job_titles”, “experience”, “company_size”, “education','decision_making’, “ppe”, “safety_inspection”, “satisfaction','alarm_system_implementation’, “confidence”, “benefits”, “accidents','negative_impact’, “cost_effectiveness”, “efficiency”, “features','technologies’, “integration_importance”, “challenges”, “feedback”]
chi2_results = pd.DataFrame(index=categorical_vars, columns=categorical_vars)

for v1 in categorical_vars:
    for v2 in categorical_vars:
        contingency_table = pd.crosstab(df[v1], df[v2])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results.loc[v1, v2] = p

plt.figure(figsize=(15, 15))
sns.heatmap(chi2_results.astype(float), annot=True, cmap=’coolwarm’, fmt=".4f")
plt.title(“Chi-square Test Results”)
plt.xticks(rotation=45, ha=’right’) # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()



# --- Section 2: Correlation Analysis ---
corr_matrix = df.corr(numeric_only=True) # Specify numeric_only for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap=’coolwarm’, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()



# --- Section 3: Multivariate Analysis ---

# Logistic Regression (Predicting Likelihood of Implementing Alarm System)
X = df[['experience_in_years', 'satisfaction_with_safety_practices', 'involved_in_safety_decisions']]
y = df['likelihood_of_implementing_alarm_system'].apply(lambda x: 1 if x > 0.5 else 0)


oversampler = RandomOverSampler(random_state=42) # Addressing class imbalance if needed
X_resampled, y_resampled = oversampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
 
