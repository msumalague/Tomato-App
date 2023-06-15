import datetime
import pickle

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from imageai.Detection.Custom import CustomObjectDetection
from imageai.Detection.Custom import CustomVideoObjectDetection
import os

app = Flask("Tomatect")
CORS(app)

execution_path = os.getcwd()
prediction = CustomObjectDetection()
prediction.setModelTypeAsYOLOv3()
prediction.setModelPath(os.path.join(execution_path, "model.pt"))
prediction.setJsonPath(os.path.join(execution_path, "json/detection_config.json"))
prediction.loadModel()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "model.pt"))
video_detector.setJsonPath(os.path.join(execution_path, "json/detection_config.json"))
video_detector.loadModel()

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(execution_path, image.filename)
    image.save(image_path)

    detections = prediction.detectObjectsFromImage(input_image=image_path,
                                                   output_image_path=os.path.join(execution_path, "output.jpg"))

    results = []
    for detection in detections:
        result = {
            "name": detection["name"],
            "percentage_probability": detection["percentage_probability"]
        }
        results.append(result)

    return jsonify(results), 200

@app.route("/live_detect", methods=["POST"])
def live_detect():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    video_path = os.path.join(execution_path, video.filename)
    video.save(video_path)

    def forFrame(frame_number, output_array, output_count):
        results = []
        for detection in output_array:
            result = {
                "name": detection["name"],
                "percentage_probability": detection["percentage_probability"]
            }
            results.append(result)

        print(results)  # Modify as per your requirement

    video_detector.detectObjectsFromVideo(input_file_path=video_path,
                                          frames_per_second=20,
                                          frame_detection_interval=1,
                                          per_frame_function=forFrame,
                                          output_file_path=os.path.join(execution_path, "output_video.avi"),
                                          minimum_percentage_probability=30)

    return "Live detection completed"

@app.route("/survival_rate", methods=['GET'])
def survival_rate():
    result = request.args
    data = [
        float(result["area_planted"]),
        float(result["area_harvested"]),
        float(result["temp_max"]),
        float(result["month"]),
        float(result["temp_min"]),
        float(result["rel_humidity"]),
        float(result["rainfall"]),
        float(result["temp_mean"]),
        float(result["production_kg"])
    ]
    prediction = model.predict([data])[0]
    return jsonify({'survival_rate': float(prediction)})

# API endpoint to get weather data
def get_weather_data():
    url = "https://weatherapi-com.p.rapidapi.com/forecast.json"
    query_params = {
        "q": "Batangas City",
        "days": 1
    }
    headers = {
        "X-RapidAPI-Key": "436466a530msh4a39bdacd9279bdp11efa1jsn0312c5e3c07d",
        "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
    }
    response = requests.get(url, params=query_params, headers=headers)
    response.raise_for_status()
    return response.json()


# Parse weather data
def parse_weather_data():
    weather_data = get_weather_data()
    current_condition = weather_data['current']
    forecast_day = weather_data['forecast']['forecastday'][0]

    rainfall = forecast_day['day']['totalprecip_mm']
    temp_max = forecast_day['day']['maxtemp_c']
    temp_min = forecast_day['day']['mintemp_c']
    temp_mean = current_condition['temp_c']
    rel_humidity = current_condition['humidity']

    return rainfall, temp_max, temp_min, temp_mean, rel_humidity


@app.route("/survival_rate", methods=['GET'])
def predict():
    area_planted = request.args.get("area_planted")
    area_harvested = request.args.get("area_harvested")
    production_kg = request.args.get("production_kg")

    # Check if all required parameters are present
    if not all([area_planted, area_harvested, production_kg]):
        return jsonify({'error': 'Missing required query parameters'}), 400

    try:
        area_planted = float(area_planted)
        area_harvested = float(area_harvested)
        production_kg = float(production_kg)
    except ValueError:
        return jsonify({'error': 'Invalid query parameters (must be numeric)'}), 400

    # Get the current month as a number
    current_month = datetime.now().month

    # Get weather data
    rainfall, temp_max, temp_min, temp_mean, rel_humidity = parse_weather_data()

    data = [
        area_planted,
        area_harvested,
        temp_max,
        current_month,
        temp_min,
        rel_humidity,
        rainfall,
        temp_mean,
        production_kg
    ]
    prediction = model.predict([data])[0]

    return jsonify({'survival_rate': float(prediction)})

if __name__ == "__main__":
    app.run()

#http://localhost:5000/predict/?area_planted=10.5&area_harvested=8.2&production_kg=1500.0

