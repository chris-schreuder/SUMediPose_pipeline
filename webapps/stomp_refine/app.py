from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import json
import os
import pandas as pd

app = Flask(__name__)

root_path = 'path/to/your/data/'

# Load data
try:
    with open(f'{root_path}data/stomp_refine.json', 'r') as f:
        stomp_data = json.load(f)
except FileNotFoundError:
    stomp_data = {}

combinations_done = list(stomp_data.keys())

df = pd.read_csv(f'{root_path}data/stomps_C4.csv')
df = df[~df['combination'].isin(combinations_done)]

df['first_idx'] = pd.to_numeric(df['first_idx'], errors='coerce').astype('Int64')
df['last_idx'] = pd.to_numeric(df['last_idx'], errors='coerce').astype('Int64')

combinations = df['combination'].tolist()
first_idxs = df['first_idx'].tolist()
last_idxs = df['last_idx'].tolist()

combination_idx = 0

combination = combinations[combination_idx]
first_idx = first_idxs[combination_idx]
last_idx = last_idxs[combination_idx]

# Route for the home page
@app.route('/')
def index():
    global combination
    global first_idx
    global last_idx
    cam = combination[1]
    subject = combination.split('S')[-1].split('A')[0]
    path_first = f'C{cam}/S{subject}/{combination}/{combination}_{str(first_idx).zfill(4)}.jpg'
    path_last = f'C{cam}/S{subject}/{combination}/{combination}_{str(last_idx).zfill(4)}.jpg'
    return render_template('index.html', initial_image_path=path_first, final_image_path=path_last)

# Route to serve images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join(f'{root_path}data/frames', filename))

# Route to get the next image URL for the first image
@app.route('/next_image_first', methods=['GET'])
def next_image_first():
    global first_idx
    global combination
    first_idx += 1
    cam = combination[1]
    subject = combination.split('S')[-1].split('A')[0]
    path_first = f'C{cam}/S{subject}/{combination}/{combination}_{str(first_idx).zfill(4)}.jpg'
    return jsonify(next_image_url=f'/images/{path_first}')

@app.route('/prev_image_first', methods=['GET'])
def prev_image_first():
    global first_idx
    global combination
    first_idx -= 1
    cam = combination[1]
    subject = combination.split('S')[-1].split('A')[0]
    path_first = f'C{cam}/S{subject}/{combination}/{combination}_{str(first_idx).zfill(4)}.jpg'
    return jsonify(next_image_url=f'/images/{path_first}')

# Route to get the next image URL for the last image
@app.route('/next_image_last', methods=['GET'])
def next_image_last():
    global last_idx
    global combination
    last_idx += 1
    cam = combination[1]
    subject = combination.split('S')[-1].split('A')[0]
    path_last = f'C{cam}/S{subject}/{combination}/{combination}_{str(last_idx).zfill(4)}.jpg'
    return jsonify(next_image_url=f'/images/{path_last}')

@app.route('/prev_image_last', methods=['GET'])
def prev_image_last():
    global last_idx
    global combination
    last_idx -= 1
    cam = combination[1]
    subject = combination.split('S')[-1].split('A')[0]
    path_last = f'C{cam}/S{subject}/{combination}/{combination}_{str(last_idx).zfill(4)}.jpg'
    return jsonify(next_image_url=f'/images/{path_last}')

# Route to move to the next combination
@app.route('/next_combination', methods=['GET'])
def next_combination():
    global combination_idx
    global combination
    global first_idx
    global last_idx
    combination_idx += 1
    if combination_idx < len(combinations):
        combination = combinations[combination_idx]
        first_idx = first_idxs[combination_idx]
        last_idx = last_idxs[combination_idx]
        cam = combination[1]
        subject = combination.split('S')[-1].split('A')[0]
        path_first = f'C{cam}/S{subject}/{combination}/{combination}_{str(first_idx).zfill(4)}.jpg'
        path_last = f'C{cam}/S{subject}/{combination}/{combination}_{str(last_idx).zfill(4)}.jpg'
        return jsonify(initial_image_path=f'/images/{path_first}', final_image_path=f'/images/{path_last}')
    else:
        return jsonify(error="No more combinations")

# Route to save coordinates
@app.route('/save_idx', methods=['POST'])
def save_idx():
    global combination
    stomp_data[combination] = {
        'first_idx': first_idx,
        'last_idx': last_idx
    }
    # Save the updated stomp_data to the JSON file
    with open(f'{root_path}data/stomp_refine.json', 'w') as f:
        json.dump(stomp_data, f, indent=4)
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
