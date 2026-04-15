import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

# We keep your original mapping and columns here
academic_level_mapping = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4}
categorical_columns = ['Gender', 'Major', 'Top Location Preference']
numeric_columns = ['Application Score', 'Resume Score', 'Video Score']

def process_features(df):
    """Your original feature engineering logic."""
    df = df.copy()
    
    if 'Academic Level' in df.columns:
        df = df.rename(columns={'Academic Level': 'AcademicLevel'})
        
    if 'Language' in df.columns:
        df['HasLanguage'] = ~df['Language'].astype(str).str.strip().str.upper().isin(['NO'])
    elif 'Can you speak a language other than English?' in df.columns:
        df['HasLanguage'] = ~df['Can you speak a language other than English?'].astype(str).str.strip().str.upper().isin(['NO'])
    else:
        df['HasLanguage'] = 0
    df['HasLanguage'] = df['HasLanguage'].astype(int)
        
    if 'AcademicLevel' in df.columns:
        df['AcademicLevel'] = df['AcademicLevel'].astype(str).str.strip().str.title().map(academic_level_mapping).fillna(0)
    else:
        df['AcademicLevel'] = 0
        
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = 0
            
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')
            
    return df

def run_placements(train_df, test_df, capacity_df, min_training_examples_per_location=10):
    """
    This is the 'Wrapper'. It takes the data from the website, 
    runs your model, and returns the results.
    """
    # 1. Prepare Capacity
    capacity_df['Placement Location'] = capacity_df['Placement Location'].astype(str).str.strip()
    capacity_dict = dict(zip(capacity_df['Placement Location'], capacity_df['Capacity']))

    # 2. Filter Training Data
    placement_counts = train_df['Placement Decision'].value_counts()
    valid_placements = placement_counts[placement_counts > min_training_examples_per_location].index
    train_df = train_df[train_df['Placement Decision'].isin(valid_placements)]

    # 3. Process Features
    X_train_raw = process_features(train_df)
    y_train = train_df['Placement Decision']
    X_test_raw = process_features(test_df)

    feature_cols = categorical_columns + ['AcademicLevel', 'HasLanguage'] + numeric_columns
    X_train = X_train_raw[feature_cols]
    X_test = X_test_raw[feature_cols]

    # 4. Train Model
    passthrough_cols = ['AcademicLevel', 'HasLanguage'] + numeric_columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
            ('ordinal_num', 'passthrough', passthrough_cols)
        ],
        remainder='drop'
    )

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)

    # 5. Predict
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    results_df = test_df.copy()
    sorted_indices = np.argsort(y_proba, axis=1)[:, ::-1]

    top2_list, top3_list = [], []
    for i in range(len(results_df)):
        top2_list.append(classes[sorted_indices[i, 1]] if len(classes) > 1 else None)
        top3_list.append(classes[sorted_indices[i, 2]] if len(classes) > 2 else None)

    results_df['Predicted_Placement'] = y_pred
    results_df['Top2_Class'] = top2_list
    results_df['Top3_Class'] = top3_list

    if 'Total Score' in results_df.columns:
        results_df = results_df.sort_values(by='Total Score', ascending=False)

    # 6. Enforce Capacity
    cap_dict_norm = {k.strip(): v for k, v in capacity_dict.items()}
    assigned_counts = defaultdict(int)
    final_placements = []

    for idx, row in results_df.iterrows():
        candidates = [row['Predicted_Placement'], row['Top2_Class'], row['Top3_Class']]
        assigned = False
        for choice in candidates:
            if pd.isna(choice): continue
            choice_clean = str(choice).strip()
            limit = cap_dict_norm.get(choice_clean, 999)
            
            if assigned_counts[choice_clean] < limit:
                assigned_counts[choice_clean] += 1
                final_placements.append(choice_clean)
                assigned = True
                break
        if not assigned:
            final_placements.append("Unassigned")
            assigned_counts["Unassigned"] += 1

    results_df['Final_Placement'] = final_placements

    return results_df, assigned_counts