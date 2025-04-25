from flask import Flask, render_template, request
import google.generativeai as genai
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA1Zte9GOrMWZ5NcAO3K7p2MlJ62y_1hs8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# client = genai.Client(api_key="AIzaSyA1Zte9GOrMWZ5NcAO3K7p2MlJ62y_1hs8")

app = Flask(__name__)

# Initialize the model
model = genai.GenerativeModel("models/gemini-2.0-flash")
# model="gemini-2.0-flash"

# Function to generate recommendations
def generate_recommendation(user_age,user_height,user_weight,user_gender,dietary_preferences, fitness_goals, lifestyle_factors, dietary_restrictions,
                            health_conditions,):
    prompt = f"""
    Can you suggest a comprehensive plan that includes diet and workout options for better fitness?
    for this user:
    age:{user_age},
    height:{user_height}cm,
    weight:{user_weight}kg,
    gender:{user_gender},
    dietary preferences: {dietary_preferences},
    fitness goals: {fitness_goals},
    lifestyle factors: {lifestyle_factors},
    dietary restrictions: {dietary_restrictions},
    health conditions: {health_conditions},
 

    Based on the above user’s dietary preferences, fitness goals, lifestyle factors, dietary restrictions, and health conditions provided, create a customized plan that includes:

    Diet Recommendations: RETURN LIST
    5 specific diet types suited to their preferences and goals.

    Workout Options: RETURN LIST
    5 workout recommendations that align with their fitness level and goals.

    Meal Suggestions: RETURN LIST
    5 breakfast ideas.

    5 dinner options.

    


    Additional Recommendations: RETURN LIST
    Any useful snacks, supplements, or hydration tips tailored to their profile.


    Nutrients Value: RETURN LIST
    total calories, protein, carbohydrates, fats, vitamins, that the user should consume per breakfast and dinner.




    """

    response = model.generate_content(prompt)
    return response.text if response else "No response from the model."

@app.route('/', methods=['GET', 'POST'])
def home():
       return render_template('homepage.html')



@app.route('/diet-and-workout')
def index():
    return render_template('index.html', recommendations=None)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == "POST":
        # Collect form data
        user_age = request.form['user_age']
        user_height = request.form['user_height'] 
        user_weight = request.form['user_weight']
        user_gender = request.form['user_gender']
        dietary_preferences = request.form['dietary_preferences']
        fitness_goals = request.form['fitness_goals'] or "none"
        lifestyle_factors = request.form['lifestyle_factors'] or "none"
        dietary_restrictions = request.form['dietary_restrictions'] or "none"
        health_conditions = request.form['health_conditions'] or "none"
        

        # user_query = request.form['user_query']

        # print(user_age,user_height,user_weight,user_gender,dietary_preferences, fitness_goals, lifestyle_factors, dietary_restrictions, health_conditions)

        # Generate recommendations using the model
        recommendations_text = generate_recommendation(
           user_age,user_height,user_weight,user_gender,dietary_preferences, fitness_goals, lifestyle_factors, dietary_restrictions, health_conditions,
        )

        # Parse the results for display
        recommendations = {
            "diet_types": [],
            "workouts": [],
            "breakfasts": [],
            "dinners": [],
            "additional_tips": [],
            "nutrient_value": []
        }

        # print("text : ", recommendations_text)

        # Split and map responses based on keywords
        current_section = None
        for line in recommendations_text.splitlines():
            if "Diet Recommendations:" in line:
                current_section = "diet_types"
            elif "Workout Options:" in line:
                current_section = "workouts"
            elif "Meal Suggestions:" in line:
                current_section = "breakfasts"
            elif "Dinner Options:" in line:
                current_section = "dinners"
            elif "Additional Recommendations:" in line:
                current_section = "additional_tips"
            elif "Nutrients Value:" in line:
                 current_section = "nutrient_value"
            elif line.strip() and current_section:
                recommendations[current_section].append(line.strip())

        print("dict : ", recommendations)
        return render_template('recommendation.html', recommendations=recommendations)
    
    # @app.route('/', methods=['GET', 'POST'])
    # def home():
    #    return render_template('homepage.html')



        # Absolute path example
data = pd.read_csv("updated_recipes_with_youtube_links.csv")


# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    input_features_scaled = scaler.transform([input_features[:7]])
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url' , 'youtube_link']].head(5)

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text
    
@app.route('/recipe-recommendation')
def recipe_recommendation():
    return render_template('recipe.html', recommendations=None)

@app.route('/recipe', methods=[ 'POST'])
def recipe ():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template('recipe.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
    return render_template('recipe.html', recommendations=[])
    



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
