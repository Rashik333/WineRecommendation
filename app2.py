import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import sys
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
import shap

food_data = pd.read_csv('food_data.csv')
wine_data = pd.read_csv('wine_data.csv')

food_data = food_data.sort_values(by='Food')

def calculate_food_properties(selected_food_items, food_data):
    selected_food_data = food_data[food_data['Food'].isin(selected_food_items)]
    avg_properties = selected_food_data.drop('Food', axis=1).mean()
    return avg_properties
    
def suggest_wines(food_properties, wine_data):
    wine_properties = wine_data.drop('Wine', axis=1)
    similarities = cosine_similarity([food_properties], wine_properties)[0]
    
    congruent_wines = []
    contrasting_wines = []
    for i, similarity in enumerate(similarities):
        wine_name = wine_data['Wine'][i]
        if similarity >= 0.5:
            congruent_wines.append((wine_name, similarity))
        else:
            contrasting_wines.append((wine_name, similarity))
    
    congruent_wines = sorted(congruent_wines, key=lambda x: x[1], reverse=True)
    contrasting_wines = sorted(contrasting_wines, key=lambda x: x[1], reverse=True)
    
    return congruent_wines, contrasting_wines
       
def calculate_food_wine_similarity(selected_food_items, food_data, wine_data, predicted_ratings):
    selected_food_data = food_data[food_data['Food'].isin(selected_food_items)]
    avg_properties = selected_food_data.drop('Food', axis=1).mean()

    wine_properties = wine_data.drop('Wine', axis=1)
    similarities = cosine_similarity([avg_properties], wine_properties)
    
    wine_names = wine_data['Wine'].tolist()
    wine_recommendations = list(zip(wine_names, similarities[0]))
    wine_recommendations = sorted(wine_recommendations, key=lambda x: x[1], reverse=True)
    
    return wine_recommendations 

def train_recommendation_model(food_data, wine_data):
    user_food_matrix = pd.DataFrame(index=food_data['Food'], columns=wine_data['Wine'])
    
    np.random.seed(0)
    user_food_matrix.values[:] = np.random.rand(*user_food_matrix.shape)
    
    nmf_model = NMF(n_components=2, init='random', random_state=0)
    W = nmf_model.fit_transform(user_food_matrix)
    H = nmf_model.components_
    
    predicted_ratings = np.dot(W, H)
    
    return predicted_ratings

def explain_with_lime(selected_food_items, food_data, wine_data, predicted_ratings):
    explainer = LimeTabularExplainer(
        training_data=food_data.drop('Food', axis=1).values,
        mode="classification",
        training_labels=predicted_ratings.argmax(axis=1),
        discretize_continuous=False
    )
    
    wine_to_explain = wine_data['Wine'].iloc[0]
    wine_properties = wine_data[wine_data['Wine'] == wine_to_explain].drop('Wine', axis=1).values[0]
    
    explanation = explainer.explain_instance(
        data_row=wine_properties,
        predict_fn=lambda x: x
    )
    
    return explanation.as_list()

st.title("Wine and Food Pairing Recommendation")
st.image("1.webp", width=100)
st.write("Discover the perfect wine for your favorite dishes!")
st.write("Simply select your favorite food items, and we'll recommend wines that pair well with them.")

st.subheader("How the Recommendation Works:")
st.write("Our recommendation system uses a basic Matrix Factorization model to predict user-food ratings.")
st.write("Then, it calculates the cosine similarity between the selected food items and wine properties.")
st.write("Wines with higher cosine similarity scores are considered more suitable for your selected food items.")

st.markdown(
    """
    <style>
    .stApp {
        color: #8B0000;
        background: #FFA07A;
        font-size: 18px;
    </style>
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Logout"):
    sys.exit()

food_items = st.multiselect("Select food items:", food_data['Food'].tolist())

if not food_items:
    st.warning("Please select at least one food item.")
else:
    if st.button("Recommend"):
        st.subheader("Recommended Wines:")
        st.write("Based on your selected food items, here are some recommended wines:")

        predicted_ratings = train_recommendation_model(food_data, wine_data)
        food_properties = calculate_food_properties(food_items, food_data)        
        wine_recommendations = calculate_food_wine_similarity(food_items, food_data, wine_data, predicted_ratings)
        congruent_wines, contrasting_wines = suggest_wines(food_properties, wine_data)
        
        st.subheader("Wine Recommendations:")
        for wine, recommendation in wine_recommendations[:4]:
            st.write(f"{wine}: Recommendation Score: {recommendation:.2f}")
            
        st.subheader("Congruent Wine Recommendations:")
        for wine, similarity in congruent_wines[:4]:
            st.write(f"{wine}: Similarity Score: {similarity:.2f}")

        st.subheader("Contrasting Wine Recommendations:")
        for wine, similarity in contrasting_wines[:4]:
            st.write(f"{wine}: Similarity Score: {similarity:.2f}")

        st.subheader("Radar Charts for Top 4 Recommended Wines:")
        figs = []
        for wine, _ in congruent_wines[:4]:
            wine_properties = wine_data[wine_data['Wine'] == wine].drop('Wine', axis=1).squeeze()
            fig = px.line_polar(wine_properties, r=wine_properties.values, theta=wine_properties.index, line_close=True)
            fig.update_traces(fill='toself')
            fig.update_layout(title=f'Radar Chart for {wine}')
            figs.append(fig)

        for fig in figs:
            st.plotly_chart(fig)
            
        lime_explanations = explain_with_lime(food_items, food_data, wine_data, predicted_ratings)
        st.subheader("LIME Explanations:")
        st.write(lime_explanations)

