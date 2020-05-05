import io
import os
import sys
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import time
import logging
import tempfile
import random
import string

import basnet

logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)


# Simple probe.
@app.route('/', methods=['GET'])
def hello():
    return 'Hello BASNet!'

# Route http posts to this method
@app.route('/', methods=['POST'])
def run():
    start = time.time()

    # Convert string of image data to uint8
    if 'data' not in request.files:
        return jsonify({'error': 'missing file param `data`'}), 400
    data = request.files['data'].read()
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    # Convert string data to PIL Image
    img = Image.open(io.BytesIO(data))

    # Resize image to 256 (BASNet compliant), and crop it
    img.thumbnail((256, 256))
    box = (0, 0, 256, 256)
    cropped_image = img.crop(box)

    # Process Image
    res = basnet.run(np.array(cropped_image))

    # Create mask file, load it in memory and remove file
    maskfilename = randomString(8) + ".png"
    res.save(maskfilename)
    mask = Image.open(maskfilename).convert("L")
    os.remove(maskfilename)

    empty = Image.new("RGBA", cropped_image.size, 0)
    newImg = Image.composite(cropped_image, empty, mask)

    # Save to buffer
    buff = io.BytesIO()
    newImg.save(buff, 'PNG')
    buff.seek(0)

    # Print stats
    logging.info(f'Completed in {time.time() - start:.2f}s')

    # Return data
    return send_file(buff, mimetype='image/png')

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
