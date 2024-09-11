from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import cv2
import numpy as np

from ultralytics import YOLO
import math

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
model = YOLO('yolov8n_plus_training.pt')


@app.route('/upload', methods=['POST'])
def upload_image():
    pixel_to_cm_ratio = request.form.get('pixel_to_cm_ratio', type=float, default=1.0)

    # Check for Base64-encoded image in form data
    if 'image' in request.form:
        image_data = request.form['image']

        # Check if the image data is Base64
        if image_data.startswith("data:image"):
            try:
                # Decode the Base64 string
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)

                # Convert the decoded bytes to a numpy array and then to OpenCV format
                image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Decode to OpenCV format

                # Process the image with YOLO
                return process_image(image, pixel_to_cm_ratio, quality=50)

            except Exception as e:
                print(f"Error decoding and processing image: {e}")
                return jsonify({'error': 'Failed to decode and process image'}), 500
        else:
            return jsonify({'error': 'Invalid image format'}), 400

    # Check for file upload in request.files
    elif 'image' in request.files:
        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Read the image in memory
            image_bytes = image.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Decode to OpenCV format

            # Process the image with YOLO
            return process_image(image, pixel_to_cm_ratio, quality=50)

        except Exception as e:
            print(f"Error reading and processing image: {e}")
            return jsonify({'error': 'Failed to process image'}), 500
    else:
        return jsonify({'error': 'No image data provided'}), 400

@app.route('/calibrate', methods=['POST'])
def calibrate_image():
    if 'image' in request.form:
        image_data = request.form['image']

        # Check if the image data is Base64
        if image_data.startswith("data:image"):
            try:
                # Decode the Base64 string
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)

                # Save the decoded image to the upload folder
                image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)

                # Process the image with YOLO
                return process_image(image_path)

            except Exception as e:
                print(f"Error decoding and saving image: {e}")
                return jsonify({'error': 'Failed to decode and save image'}), 500
        else:
            return jsonify({'error': 'Invalid image format'}), 400

        # Check for file upload in request.files
    elif 'image' in request.files:
        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            image = cv2.imread(image)
            results = model(image)
            blocks = results[0].boxes  # List of bounding boxes detected
            if len(blocks) == 0:
                return jsonify({'error': 'No block detected'}), 40
            block = blocks[0]  # Take the first detected block
            x1, y1, x2, y2 = map(int, block.xyxy[0])  # Extract bounding box coordinates

            length_pixels = x2 - x1
            return jsonify({
                'length_pixels': length_pixels
            }), 200

        except Exception as e:
            print(f"Error saving image file: {e}")
            return jsonify({'error': f'Failed to save image file :{e}'}), 500
    else:
        return jsonify({'error': 'No image data provided'}), 400

@app.after_request
def apply_keep_alive(response):
    response.headers["Connection"] = "keep-alive"
    return response

def process_image(image, pixel_to_cm_ratio=1, quality=50):
    print('Predicting!')
    # Perform prediction using YOLOv8
    results = model(image)

    # Initialize counters and data structures
    total_blocks = 0
    g1_blocks = 0
    non_g1_blocks = 0
    block_details = []

    # Process and display the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes

        for i, box in enumerate(boxes):
            total_blocks += 1
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            length_pixels = x2 - x1
            height_pixels = y2 - y1

            # Convert pixel measurements to cm
            truncated_ratio = math.floor(pixel_to_cm_ratio * 100) / 100
            length_cm = length_pixels * truncated_ratio
            print(truncated_ratio, length_pixels)
            print('length cm :', length_cm)
            height_cm = height_pixels * truncated_ratio
            width_cm = 5  # Assuming a fixed height for now or another method for height determination

            # Calculate volume
            volume_cm3 = length_cm * width_cm * height_cm

            # Check for red circle inside the block
            g1 = check_for_red_circle(image[y1:y2, x1:x2])  # Check for red circle within the bounding box

            if g1:
                g1_blocks += 1
            else:
                non_g1_blocks += 1

            # Create block detail dictionary
            block_details.append({
                'Name': f'Block {total_blocks}',
                'g1': g1,
                'length': round(length_cm, 2),
                'height': round(height_cm, 2),
                'width': round(width_cm, 2),
                'volume': round(volume_cm3, 2)
            })

            # Draw bounding boxes and annotations
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green bounding box with thickness 2
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2

            cv2.putText(image, f'{length_cm:.2f} cm', (center_x - 30, y1 - 10),  # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Red text

            # Display height_cm in the middle of the left and right sides of the bounding box
            cv2.putText(image, f'{height_cm:.2f} cm', (x2 + 5, center_y),  # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Red text

    # Encode the processed image back to a Base64 string
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return JSON response with Base64 encoded processed image
    payload = jsonify({
        'total_blocks': total_blocks,
        'g1_blocks': g1_blocks,
        'non_g1_blocks': non_g1_blocks,
        'block_details': block_details,
        'processed_image': f"data:image/jpeg;base64,{processed_image_base64}"
    })

    print(payload)
    return payload

def check_for_red_circle(block_image):
    # Convert to HSV color space to detect red color
    hsv = cv2.cvtColor(block_image, cv2.COLOR_BGR2HSV)

    # Define range for detecting red color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color detection
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)

    # Count non-zero pixels in the mask to determine the presence of red
    if np.count_nonzero(mask) > 0:
        return True
    else:
        return False


if __name__ == '__main__':
    # Make server accessible from outside by setting host to 0.0.0.0
    app.run(debug=True, host='0.0.0.0', port=5000)
