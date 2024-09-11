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

    # Check for file upload in request.files
    if 'image' in request.files:
        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Save the uploaded image to a temporary location
            temp_image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(temp_image_path)

            # Now read the image using OpenCV
            image_np = cv2.imread(temp_image_path)

            # Process the image with YOLO
            return process_image(image_np, pixel_to_cm_ratio, quality=50)

        except Exception as e:
            print(f"Error reading and processing image: {e}")
            return jsonify({'error': f'Failed to process image: {e}'}), 500
    else:
        return jsonify({'error': 'No image data provided'}), 400


@app.route('/calibrate', methods=['POST'])
def calibrate_image():
    # Check for file upload in request.files
    if 'image' in request.files:
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        try:
            # Save the uploaded image to a temporary location
            temp_image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(temp_image_path)

            # Now read the image using OpenCV
            image_np = cv2.imread(temp_image_path)  # Read the image from the file path

            if image_np is None:
                return jsonify({'error': 'Failed to read image file'}), 400

            # Perform object detection with the loaded model
            results = model(image_np)  # Use the numpy array directly
            blocks = results[0].boxes  # List of bounding boxes detected

            if len(blocks) == 0:
                return jsonify({'error': 'No block detected'}), 400

            # Take the first detected block
            block = blocks[0]
            x1, y1, x2, y2 = map(int, block.xyxy[0])  # Extract bounding box coordinates

            length_pixels = x2 - x1
            return jsonify({
                'length_pixels': length_pixels
            }), 200

        except Exception as e:
            print(f"Error saving image file: {e}")
            return jsonify({'error': f'Failed to process image file: {e}'}), 500
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
