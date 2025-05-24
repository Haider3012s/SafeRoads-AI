from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import json

app = Flask(__name__)

# 15 features based on your dataset
features = [
    'location_easting_osgr', 'location_northing_osgr', 'police_force',
    'legacy_collision_severity', 'day_of_week', 'local_authority_ons_district',
    'local_authority_highway', 'first_road_number', 'speed_limit',
    'junction_detail', 'second_road_number', 'pedestrian_crossing_human_control',
    'pedestrian_crossing_physical_facilities', 'special_conditions_at_site',
    'carriageway_hazards'
]

# Load model
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(f)) for f in features]
            input_array = np.array(input_data).reshape(1, -1)
            prediction = int(model.predict(input_array)[0])
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', features=features, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
