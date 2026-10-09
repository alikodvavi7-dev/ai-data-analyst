import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, silhouette_score
from google import genai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key = api_key)


st.title("AI Data Analyst")
st.write("Upload any CSV and get AI Powered analysis of your data")

uploaded_file = st.file_uploader(
    "Upload your CSV File",
    type = ["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(5))

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        st.dataframe(missing_values)
    else:
        st.success("No missing values found")

    duplicates = df.duplicated().sum()
    st.subheader("Duplicated rows")
    st.write(f"{duplicates} duplicate rows found")

    numeric_columns = df.select_dtypes(include= "number").columns.tolist()
    catogorical_columns = df.select_dtypes(include = ["object", "category"]).columns.tolist()

    st.subheader("Column Types")
    st.write("Numeric Columns:", numeric_columns)
    st.write("Non-Numeric Columns:", catogorical_columns)

    clean_df = df.copy()
    clean_df = clean_df.drop_duplicates()
    for column in numeric_columns:
        clean_df[column] = clean_df[column].fillna(
            clean_df[column].median()
        )
    for column in catogorical_columns:
        clean_df[column] = clean_df[column].fillna("unknown")

    removed_duplicates = len(df) - len(clean_df)
    st.write("Cleaning Summary")
    st.write(f"{removed_duplicates} duplicates were removed")
    st.write("Missing numerical values filled with median.")
    st.write("Missing categorical values filled with 'Unknown'.")

    numeric_summary = clean_df.describe()

    if len(numeric_columns) > 1:
        corr_matrix = clean_df[numeric_columns].corr()
        corr_info = corr_matrix.to_string()
    else:
        corr_info = "Not enough numerical columns for correlation analysis"

    categorical_summary = ""
    for column in catogorical_columns:
        categorical_summary += f"\n{column}:\n"
        categorical_summary += clean_df[column].value_counts().head(10).to_string()
        categorical_summary += "\n"

    dataset_info = f"""
    Rows : {clean_df.shape[0]}
    Columns : {clean_df.shape[1]}
    Column Names : {list(clean_df.columns)}
    data types : {clean_df.dtypes.to_string()}
    numerical summary : {numeric_summary.to_string()}
    categorical summary : {categorical_summary}
    correlation summary : {corr_info}
    """
    
    prompt = f"""
    You are an expert data analyst. analyse the following dataset information 
    {dataset_info}
    Give 5 useful insights on this dataset
    Focus on:
    Important Patterns
    unsual values
    interesting differences
    possible relationships between variables
    useful business or behavioral insights
    also explain the correlations briefly
    
    keep the explanation clear and concise
    do not make claims that cannot be supported by the provided info
    """
        
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing your dataset"):
            try:
                    response = client.models.generate_content(
                    model = "gemini-2.5-flash",
                    contents = prompt
                )
                    st.subheader("AI Insights")
    
                    st.write(response.text)
            except Exception as e:
                    st.error("AI is temporarily unavailable. Please try later.")
                
            
    

    st.subheader("Machine Learning Analysis")
    type = st.selectbox("Choose the analysis type", ["Regression", "Classification", "K-Means Clustering"])

    if type == "Regression":
        st.write("Select the numerical variable you want to predict")
        target_column = st.selectbox("Target variable", numeric_columns)
        available_features = [column for column in numeric_columns if column != target_column]
        feature_column = st.multiselect(
            "Select the variables used to predict the target",
            available_features,
            default = available_features
        )
        
        x = clean_df[feature_column]
        y = clean_df[target_column]

        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=24)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.subheader("Regression Results")
        st.write("R2 Score: ", round(r2,3))
        st.write("Mean Absolute Error: ", round(mae,3))
        cofficients = pd.DataFrame({
            "Feature" : feature_column,
            "Cofficient" : model.coef_
        })
        st.write("Feature Effects")
        st.dataframe(cofficients)
        st.write("Intercept: ", round(model.intercept_, 3))

        regression_info = f"""
Target variable: {target_column}
R2 score : {r2}
mean absolute error : {mae}
regression cofficients : {cofficients.to_string(index = False)}
intercept : {model.intercept_}
Training Rows: {len(x_train)}
Testing Rows: {len(x_test)}

"""
        regression_prompt = f"""
You are a data analyst.

Interpret these regression results:

{regression_info}

Explain:
- how well the model performed
- what the R2 score means
- what the MAE means
- which features have the strongest relationships with the target
- whether those relationships are positive or negative

Use simple language.
Do not claim that the relationships are causal.
"""
        if st.button("Generate Regression Insights"):
                 with st.spinner("Analyzing...."):
                     try:
                        reg_response = client.models.generate_content(
                        model = "gemini-2.5-flash",
                        contents = regression_prompt
                    )
                        st.subheader("AI Regression Insights")
        
                        st.write(reg_response.text)
                     except Exception as e:
                        st.error("AI is temporarily unavailable. Please try later.")


    if type == "Classification":
         st.write("Select the categorical variable you want to predict")
         target_column = st.selectbox("Classification target", catogorical_columns)
         feature_column = st.multiselect("Select the variables used to predict the target", numeric_columns, default = numeric_columns)
         x = clean_df[feature_column]
         y = clean_df[target_column]
         x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 42)
         models = {
              "Logistic Regression" : LogisticRegression(max_iter= 1000),
              "Decision Tree" : DecisionTreeClassifier(random_state=24),
              "Random Forest" : RandomForestClassifier(random_state=42)
         }

         param_grids = {
              "Logistic Regression" : {"C": [0.1,1,10]},
              "Decision Tree" : {"max_depth": [None, 5,10], "min_samples_split": [2,5]},
              "Random Forest" : {"n_estimators": [50,100,200], "max_depth": [None, 5,10]}
         }



         results = []
         for model_name, model in models.items():
              grid_search = GridSearchCV(model, param_grids[model_name], cv = 5, scoring = "f1_weighted")
              grid_search.fit(x_train,y_train)
              best_model = grid_search.best_estimator_
              
              y_pred = best_model.predict(x_test)
              test_f1 = f1_score(y_test, y_pred, average = "weighted", zero_division= 0)
              accuracy = accuracy_score(y_test, y_pred)
              precision = precision_score(y_test, y_pred, average = "weighted", zero_division= 0)
              recall = recall_score(y_test, y_pred, average = "weighted", zero_division= 0)
              results.append({
                   "model" : model_name,
                   "best parameters" : grid_search.best_params_,
                   "best cv f1" : grid_search.best_score_,
                   "test f1" : test_f1,
                   "accuracy" : accuracy,
                   "precision" : precision,
                   "recall" : recall
              })

         results_df = pd.DataFrame(results)
         st.subheader("Classification Results")
         st.dataframe(results_df)
         best_model_row = results_df.loc[results_df["test f1"].idxmax()]
         best_model_name = best_model_row["model"]
         best_f1 = best_model_row["test f1"]

         st.success(
              f"Best Model: {best_model_name} | F1 Score: {best_f1}"
         )

         classification_info = f"""
Classification model results:
{results_df.to_string(index = False)}
best model : {best_model_name}
best f1 : {best_f1}
target variable: {target_column}
features used : {feature_column}
"""
         classification_prompt = f"""
you are a data analyst, interpret these classification results 
{classification_info}

explain 
1. which model performed best
2. what the accuracy, precision, recall and f1 scores mean in this data/classification
3. how the models compare
4. whether the best model performed well overall
5. any limitations

Use simple language
"""
         if st.button("Generate Classification Insights"):
                          with st.spinner("Analyzing...."):
                              try:
                                 class_response = client.models.generate_content(
                                 model = "gemini-2.5-flash",
                                 contents = classification_prompt
                             )
                                 st.subheader("AI Classification Insights")
                 
                                 st.write(class_response.text)
                              except Exception as e:
                                 st.error("AI is temporarily unavailable. Please try later.")


    if type == "K-Means Clustering":
         st.write("Select the numerical variables you want to use for clustering")
         cluster_features = st.multiselect("Variables for clustering", numeric_columns, default = numeric_columns)
         x_cluster = clean_df[cluster_features]
         scaler = StandardScaler()
         scaled_features = scaler.fit_transform(x_cluster)
         n_clusters = st.slider(
              "Number of clusters",
              min_value = 2,
              max_value = 10,
              value = 3
         )
         kmeans = KMeans(n_clusters= n_clusters, random_state=42)
         cluster_labels = kmeans.fit_predict(scaled_features)
         cluster_df = clean_df.copy()
         cluster_df["Cluster"] = cluster_labels
         st.subheader("Cluster Sizes")
         cluster_sizes = cluster_df["Cluster"].value_counts().sort_index()
         st.write(cluster_sizes)
         profiles = cluster_df.groupby("Cluster")[cluster_features].mean()
         st.subheader("Cluster Profiles")
         st.dataframe(profiles)

         silhouette = silhouette_score(scaled_features, cluster_labels)
         st.write("Clustering Performance")
         st.write("Silhouette Score: ", round(silhouette,3))

         k_values = range(2,10)
         silhouette_scores = []
         for k in k_values:
              test_kmeans = KMeans(n_clusters=k, random_state=42)
              test_labels = test_kmeans.fit_predict(scaled_features)
              score = silhouette_score(scaled_features, test_labels)
              silhouette_scores.append(score)
         best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]
         best_silhouette = max(silhouette_scores)
         st.write("Recommended number of Clusters: ", best_k)
         st.write("Best Silhouette Score: ", best_silhouette)

         clustering_info = f"""
number of clusters selected : {n_clusters}
silhouette score : {silhouette}
cluster sizes : {cluster_sizes.to_string()}
cluster profiles : {profiles.to_string()}
recommended number of clusters : {best_k}
best silhouette score : {best_silhouette}

"""
         clustering_prompt = f"""
You are a data analyst. Interperet these clustering results {clustering_info}
Explain:
1. what makes each cluster different
2. which variables are the most important in distinguishing the clusters
3. whether the selected number of clusters appear reasonable
4. how it comnpares with the recommended number of clusters 
5. any useful patterns in the group?

Explain in simple words

"""
         if st.button("Generate Clustering Insights"):
                                   with st.spinner("Analyzing...."):
                                       try:
                                          clustering_response = client.models.generate_content(
                                          model = "gemini-2.5-flash",
                                          contents = clustering_prompt
                                      )
                                          st.subheader("AI Clustering Insights")
                          
                                          st.write(clustering_response.text)
                                       except Exception as e:
                                          st.error("AI is temporarily unavailable. Please try later.")
         


         
                   
              




         





    