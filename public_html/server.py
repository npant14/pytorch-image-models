from flask import Flask, render_template, jsonify, send_from_directory
import os
import random

app = Flask(__name__, template_folder="templates")

# Define base directories for images
BASE_PATH = "/gpfs/data/tserre/npant1/ILSVRC/train/"
MASK_PATH = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/EVF-SAM/ghverify_imagenet/"

@app.route("/")
def index():
    """Render the HTML page."""
    return render_template("index.html")

@app.route("/get_images/<class_id>")
def get_images(class_id):
    """API endpoint to get image lists for a given class."""
    class_dir = os.path.join(BASE_PATH, class_id)
    mask_dir = os.path.join(MASK_PATH, class_id)

    if not os.path.exists(class_dir) or not os.path.exists(mask_dir):
        return jsonify({"error": "Class not found"}), 404

    # Get valid image-mask pairs
    image_files = [
        f for f in os.listdir(class_dir)
        if f.endswith(".JPEG") and os.path.exists(os.path.join(mask_dir, f.replace(".JPEG", "_vis.png")))
    ]

    if not image_files:
        return jsonify([])

    # Randomly sample up to 20 images
    sampled_images = random.sample(image_files, min(20, len(image_files)))

    # Prepare the JSON response
    image_data = [
        {
            "original": f"/train/{class_id}/{img}",
            "mask": f"/masks/{class_id}/{img.replace('.JPEG', '_vis.png')}",
            "thumbnail": f"/masks/{class_id}/{img.replace('.JPEG', '_vis.png')}"  # Display mask as thumbnail
        }
        for img in sampled_images
    ]

    return jsonify(image_data)

# Serve original images
@app.route("/train/<path:filename>")
def serve_train_images(filename):
    return send_from_directory(BASE_PATH, filename)

# Serve mask images
@app.route("/masks/<path:filename>")
def serve_mask_images(filename):
    return send_from_directory(MASK_PATH, filename)



if __name__ == "__main__":
    app.run(host="172.20.209.24", port=8512, debug=True)
