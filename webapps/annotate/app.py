from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

all_points = []
current_image_index = 0
image_paths = []
markers_dict = {}

# Load image paths from JSON file at startup
with open('/home/chris/Desktop/Masters/Stereo Paper/code/data_process/markers.json', 'r') as f:
    markers_dict = json.load(f)

# Extract all image paths into a list
for subject in markers_dict:
    for camera in markers_dict[subject]:
        if 'points' not in markers_dict[subject][camera]:
            image_paths.append(markers_dict[subject][camera]['path'])

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
    current_image_index = (current_image_index + 1) % len(image_paths)
    next_image_path = image_paths[current_image_index]
    return jsonify(next_image_url=next_image_path)

# Route to save coordinates
@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    global all_points

    # Identify the current subject and camera
    current_image_path = image_paths[current_image_index]
    path_parts = current_image_path.split('/')
    subject = path_parts[-3]
    camera = path_parts[-4]

    # Save coordinates to the original markers_dict
    if subject in markers_dict and camera in markers_dict[subject]:
        markers_dict[subject][camera]['points'] = all_points

        # Save the updated markers_dict to the markers.json file
        with open('/home/chris/Desktop/Masters/Stereo Paper/code/data_process/markers.json', 'w') as f:
            json.dump(markers_dict, f, indent=4)

        # Clear the all_points list
        all_points = []

        return jsonify(success=True)
    return jsonify(success=False, error="Invalid subject or camera")

if __name__ == '__main__':
    app.run(debug=True)
