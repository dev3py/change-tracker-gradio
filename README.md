# FastAPI YOLOv11 Object Detection Application

This project is a FastAPI application that utilizes a custom YOLOv11 model for object detection. The model has been trained on a specific dataset and deployed using Roboflow. This README provides instructions on how to set up and run the application.

## Project Structure

```bash
root/
├── pothole_detection_app.py   # Your Gradio script
├── test/
│   ├── image1.jpg
│   ├── image2.png
│   ├── image3.jpeg
│   └── ...                    # Example test images
├── app/
│   ├── main.py                # Main FastAPI app entry point
│   ├── models/
│   │   └── yolo11_custom_model.pt # YOLOv11 model weights
│   └── inference.py           # Inference logic for the YOLOv11 model
├── requirements.txt           # Dependencies for the FastAPI application
└── README.md                  # Project documentation
```

## Setup Instructions

1.  **Clone the repository:**
    
    ```bash
    git clone https://github.com/dev3py/bike-no-helmet-detection.git
    cd bike-no-helmet-detection
    ```
    
2.  **Create a virtual environment on Ubuntu (optional but recommended):**
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    
    **Create a virtual environment on Windows (optional but recommended):**
    
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    
3.  **Install the required dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Download the YOLOv11 model weights:**
    
    Ensure that the `yolo11_custom_model.pt` file is placed in the `app/models` directory.
    

Running the Application
-----------------------

To start the FastAPI application, run the following command:

```bash
python pothole+detection_app.py
```

The application will be accessible at `http://127.0.0.1:7866`.

Docker Build
------------

```bash
docker build -t atvds .
```

Docker Run the Build
--------------------

```bash
docker run -p 7866:7866 --name atvds-container -d atvds
```

Usage
-----

### Pothole Detection App

This is a Gradio-based application for detecting potholes in uploaded images. It uses a pre-trained detection model hosted on Roboflow via the `inference_sdk` API. The app dynamically loads test images from a local folder (`test`) and provides bounding boxes around detected potholes in bright orange.

### Features

*   Upload images to detect potholes using a pre-trained model.
*   Dynamically load test images from the `test` folder as examples.
*   Displays detected potholes with bounding boxes and confidence scores.
*   User-friendly interface powered by Gradio.

Installation
------------

1.  **Clone the Repository:**
    
    ```bash
    git clone https://github.com/your-repo/pothole-detection-app.git
    cd pothole-detection-app
    ```
    
2.  **Install Dependencies:**
    
    Install the required Python packages using `pip`:
    
    ```bash
    pip install gradio inference-sdk pillow watchfiles
    ```
    
3.  **Prepare the Test Folder:**
    
    *   Place example images in the `test` folder (located in the root directory).
    *   Supported image formats: `.jpg`, `.png`, `.jpeg`.
4.  **Run the App:**
    
    ```bash
    python pothole_detection_app.py
    ```
    

Folder Structure
----------------

```bash
root/
├── pothole_detection_app.py  # Main application script
├── test/                     # Folder containing test images
│   ├── example1.jpg
│   ├── example2.png
│   ├── ...
├── README.md                 # Project documentation
```

How to Use
----------

1.  **Launch the App:**
    
    Run the following command in the terminal:
    
    ```bash
    python pothole_detection_app.py


    
    ```
    
    Gradio will start a local server and provide a URL (e.g., `http://127.0.0.1:7866`).
    
2.  **Upload or Select Test Images:**
    
    *   Drag and drop an image into the upload box.
    *   OR, click on one of the preloaded example images (loaded from the `test` folder).
3.  **View Results:**
    
    *   The app will process the image and display it with potholes highlighted in bright orange.
    *   Confidence scores for each detected pothole are shown near the bounding boxes.

API Details
-----------

This app uses the `inference_sdk` to interact with the Roboflow detection API.

*   **API Base URL**: `https://detect.roboflow.com`
*   **Model ID**: `pothole-detection-bqu6s/9`
*   **API Key**: Replace `qHlOlDBUZ2xAR6Lhika7` in the script if using a different key.

Customization
-------------

*   **Change the Test Folder**: Modify the `test_folder` variable in the script to point to a different directory.
*   **Update Bounding Box Color**: Change the `"orange"` color in the `detect_objects` function to any desired color.
*   **Add Example Images**: Place additional images in the `test` folder to make them available as examples.

Dependencies
------------

*   [Gradio](https://gradio.app): User interface for the app.
*   [inference\_sdk](https://pypi.org/project/inference-sdk/): SDK for accessing the Roboflow inference API.
*   [Pillow](https://python-pillow.org/): Image manipulation library for drawing bounding boxes.

Troubleshooting
---------------

*   **No Bounding Boxes Detected**: Ensure the uploaded image contains visible potholes and meets the model’s criteria.
*   **API Key Issues**: Verify your API key is valid and matches the one in your Roboflow account.
*   **Test Folder Not Found**: Ensure a folder named `test` exists in the root directory with at least one image file.

License
-------

This project is licensed under the MIT License. See the `LICENSE` file for details.

Acknowledgments
---------------

*   [Roboflow](https://roboflow.com): For providing the detection API.
*   [Gradio](https://gradio.app): For the intuitive and easy-to-use user interface framework.

Enjoy using the Pothole Detection App! 😊 Let me know if you have further questions or need assistance. +++