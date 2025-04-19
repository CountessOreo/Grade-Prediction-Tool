from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.keras')

# Mapping index to letter grade
grade_mapping = {
    0: 'A',  # GPA >= 3.5
    1: 'B',  # 3.0 <= GPA < 3.5
    2: 'C',  # 2.5 <= GPA < 3.0
    3: 'D',  # 2.0 <= GPA < 2.5
    4: 'F'   # GPA < 2.0
}

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        study_time = float(request.form['study_time_weekly'])
        absences = float(request.form['absences'])
        tutoring = 1 if request.form['tutoring_status'] == 'Yes' else 0
        parental = int(request.form['parental_involvement'])
        extracurricular = 1 if request.form['extracurricular'] == 'Yes' else 0
        sport = 1 if request.form['sport'] == 'Yes' else 0
        music = 1 if request.form['music'] == 'Yes' else 0

        # Prepare the input data for the model
        input_data = np.array([[study_time, absences, tutoring, parental, extracurricular, sport, music]])
        
        # Predict probabilities for each class
        result = model.predict(input_data)
        
        # Get the predicted class index and map it to a grade
        predicted_class = int(np.argmax(result, axis=1)[0])
        prediction = grade_mapping.get(predicted_class, "Unknown")

    return render_template('dashboard.html', results=prediction)

if __name__ == '__main__':
    app.run(debug=True)