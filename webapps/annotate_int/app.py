from flask import Flask, render_template, request, jsonify
import json
from glob import glob

app = Flask(__name__)

all_points = []
current_image_index = 0
image_paths = []
markers_dict = {}

# Load image paths from JSON file at startup
try:
    with open('markers_init.json', 'r') as f:
        markers_dict = json.load(f)
except:
    markers_dict = {}

image_paths = glob('static/img/*.jpg')
image_paths = [path.replace('static/img\\', '') for path in image_paths]
print(len(image_paths))
print(image_paths[0])

# Route for the home page
@app.route('/')
def index():
    if image_paths:
        initial_image_path = image_paths[current_image_index]
        return render_template('index.html', initial_image_path=initial_image_path)
    return 'No images found', 404

# Route to handle mouse click coordinates
@app.route('/log_click', methods=['POST'])
def log_click():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    print(f"({x}, {y})")  # You can log this to a file or database
    all_points.append((x, y))
    print(all_points)
    return jsonify(success=True)

# Route to get the next image URL
@app.route('/next_image', methods=['GET'])
def next_image():
    global current_image_index
    current_image_index +=1
    next_image_path = image_paths[current_image_index]
    return jsonify(next_image_url=next_image_path)

# Route to save coordinates
@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    global all_points

    # Identify the current subject and camera
    current_image_path = image_paths[current_image_index]

    markers_dict[current_image_path] = all_points

    # Save the updated markers_dict to the markers.json file
    with open('markers_init.json', 'w') as f:
        json.dump(markers_dict, f, indent=4)

    # Clear the all_points list
    all_points = []

    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
