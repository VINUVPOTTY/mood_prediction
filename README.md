📝 How the Mood Prediction App Works (Full Explanation)
1. User Interface & Experience:
The app is built with Streamlit, providing a clean, modern, and interactive web interface. Users are greeted with a visually appealing background and a glowing, animated heading for the app title. All input fields are clearly labeled and styled for ease of use.

2. Input Collection:
Users are prompted to:

Select a Sub Mood (e.g., Happy, Tired, etc.) from a dropdown.
Select the Weekday (e.g., Monday, Tuesday, etc.).
Choose one or more Activities (e.g., Reading, Exercise, etc.) from a multi-select list.
3. Data Preprocessing:
When the user clicks "Predict Mood":

The app encodes the selected sub-mood and weekday using pre-trained label encoders.
Activities are transformed into a binary vector using a multi-label binarizer, matching the format used during model training.
All features are combined into a single row DataFrame and scaled using the same scaler as the training phase.
4. Machine Learning Prediction:

The processed input is fed into a pre-trained machine learning model (Random Forest or similar, loaded via joblib).
The model predicts the user's mood based on the input features.
5. Output & Feedback:

The predicted mood is displayed prominently, accompanied by a relevant emoji for instant visual feedback.
The app background can be customized (or extended) to reflect the predicted mood, enhancing the user experience.
6. Technical Stack:

Frontend/UI: Streamlit, HTML, CSS (for custom styling and animations)
Backend/ML: scikit-learn (for model and encoders), pandas, numpy, joblib (for loading models and transformers)
Deployment: Streamlit Cloud (or similar platform)
7. Code Structure Highlights:

All models and encoders are loaded at startup for fast predictions.
Session state is used to manage background images and UI state.
The code is modular, making it easy to extend with new moods, activities, or model improvements.
8. Customization & Extensibility:

You can easily add more moods, activities, or change the UI theme.
The app can be adapted for other prediction tasks by swapping out the model and encoders.
