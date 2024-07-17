import sklearn
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def add_noise(dataframe, noise_level=0.01):

    noisy_dataframe = dataframe.copy()
    numeric_columns = dataframe.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        noise = np.random.normal(0, noise_level, len(dataframe))
        noisy_dataframe[column] += noise

    return noisy_dataframe

def calculate_percentage_similarity(df1, df2, tolerance=1e-6):
    # if df1.shape != df2.shape:
    #     raise ValueError("Dataframes must have the same shape for comparison.")
    matching_elements = np.sum(np.isclose(df1, df2, atol=tolerance))

    total_elements = df1.size
    percentage_similarity = (matching_elements / total_elements) * 100

    return percentage_similarity

csv_path = 'DNP3_Intrusion_Detection_Dataset/Training_Testing_Balanced_CSV_Files/Custom_DNP3_Parser/Custom_DNP3_Parser_Training_Balanced.csv'

df = pd.read_csv(csv_path,sep=',')

numeric_columns = df.select_dtypes(include=np.number)
df_numeric = df[numeric_columns.columns]

df_train, df_test = train_test_split(df_numeric, test_size=0.2, random_state=42)

knnimputer = KNNImputer(n_neighbors=2, weights="uniform")
knnimputer.fit(df_train)

meanimp = SimpleImputer(missing_values=np.nan, strategy='mean')
meanimp.fit(df_train)

modeimp= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
modeimp.fit(df_train)

medianimp= SimpleImputer(missing_values=np.nan, strategy='median')
medianimp.fit(df_train)

itrimp = IterativeImputer(max_iter=10, random_state=0)
itrimp.fit(df_train)

probability = 0.1
mask = np.random.rand(*df_test.shape) < probability
df_masked = df_test.mask(mask)

knn_per_sim = calculate_percentage_similarity(df_test,pd.DataFrame(knnimputer.transform(df_masked)))
print(f"Percentage Similarity knn: {knn_per_sim:.2f}%")

imp_per_sim = calculate_percentage_similarity(df_test,pd.DataFrame(itrimp.transform(df_masked)))
print(f"Percentage Similarity linear regression: {imp_per_sim:.2f}%")

mean_per_sim= calculate_percentage_similarity(df_test,pd.DataFrame(meanimp.transform(df_masked)))
print(f"Percentage Similarity mean: {mean_per_sim:.2f}%")

median_per_sim = calculate_percentage_similarity(df_test,pd.DataFrame(medianimp.transform(df_masked)))
print(f"Percentage Similarity median: {median_per_sim:.2f}%")

mode_per_sim = calculate_percentage_similarity(df_test,pd.DataFrame(modeimp.transform(df_masked)))
print(f"Percentage Similarity mode: {mode_per_sim:.2f}%")
