import cv2
import numpy as np
import gradio as gr


# ----------------- Helper Functions (OpenCV processing) ----------------- #
def align_images(old_img, new_img):
    """
    Align the old image to the new image using feature matching.
    Both images are assumed to be in BGR.
    """
    # Convert to grayscale
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(old_gray, None)
    kp2, des2 = orb.detectAndCompute(new_gray, None)

    # Match features using brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix using RANSAC
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the old image to align with the new image (using cubic interpolation)
    aligned_old = cv2.warpPerspective(
        old_img, M, (new_img.shape[1], new_img.shape[0]), flags=cv2.INTER_CUBIC
    )
    return aligned_old


def detect_added_elements(old_img, new_img, min_contour_area=100):
    """
    Detect elements that are present in new_img (BGR) but not in old_img (BGR).
    Returns a binary mask for the added regions.
    """
    # First, align the old image to the new image
    old_aligned = align_images(old_img, new_img)

    # Compute absolute difference between new image and aligned old image
    diff = cv2.absdiff(new_img, old_aligned)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to capture areas with significant differences (new content)
    _, thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Clean the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter small regions
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask


def detect_deleted_elements(old_img, new_img, min_contour_area=100):
    """
    Detect elements that were present in old_img but have been removed in new_img.
    Returns a tuple: (binary mask for deleted regions, aligned old image)
    """
    # Align old image to new image
    aligned_old = align_images(old_img, new_img)

    # Convert both images to grayscale
    old_gray = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # For deleted elements: subtract the old from the new (so white shows removed parts)
    diff = cv2.subtract(new_gray, old_gray)

    # Threshold to obtain binary mask of deleted areas
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Clean the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter by area
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask, aligned_old


def highlight_additions_on_white_cv2(old_img, new_img):
    """
    Process the added elements: extract them from the new image and overlay on a white background.
    The images are assumed to be in BGR.
    Returns the resulting image (BGR).
    """
    added_mask = detect_added_elements(old_img, new_img)
    added_elements = cv2.bitwise_and(new_img, new_img, mask=added_mask)
    white_background = np.full(new_img.shape, 255, dtype=np.uint8)
    white_background[added_mask == 255] = added_elements[added_mask == 255]
    return white_background


def highlight_deletions_on_white_cv2(old_img, new_img):
    """
    Process the deleted elements: extract them from the aligned old image and overlay on a white background.
    The images are assumed to be in BGR.
    Returns the resulting image (BGR).
    """
    deleted_mask, aligned_old = detect_deleted_elements(old_img, new_img)
    deleted_elements = cv2.bitwise_and(aligned_old, aligned_old, mask=deleted_mask)
    white_background = np.full(aligned_old.shape, 255, dtype=np.uint8)
    white_background[deleted_mask == 255] = deleted_elements[deleted_mask == 255]
    return white_background


# ----------------- Gradio Interface Function ----------------- #
def process_images(old_img, new_img):
    """
    Expects two images (old and new) as input in RGB format.
    Converts them to BGR, processes the added and deleted elements,
    and returns two images (converted back to RGB) for display.
    """
    if old_img is None or new_img is None:
        return None, None

    # Convert from RGB (Gradio default) to BGR (OpenCV default)
    old_bgr = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR)
    new_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    # Process for added and deleted elements
    added_result_bgr = highlight_additions_on_white_cv2(old_bgr, new_bgr)
    deleted_result_bgr = highlight_deletions_on_white_cv2(old_bgr, new_bgr)

    # Convert results back to RGB for display in Gradio
    added_result = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
    deleted_result = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)

    return added_result, deleted_result


# ----------------- Build the Gradio Interface ----------------- #
with gr.Blocks() as demo:
    gr.Markdown("# Compare Images: Added & Deleted Elements")
    gr.Markdown(
        """
        Upload the **Old Image** and the **New Image** (both as technical drawings or similar)
        and click **Process** to see:
        
        - **Added**: New elements that appear in the new image.
        - **Deleted**: Elements that were removed from the old image.
        """
    )

    with gr.Row():
        old_image = gr.Image(label="Old Image", type="numpy")
        new_image = gr.Image(label="New Image", type="numpy")

    process_button = gr.Button("Process")

    with gr.Tabs():
        with gr.Tab("Added"):
            added_output = gr.Image(label="Added Elements")
        with gr.Tab("Deleted"):
            deleted_output = gr.Image(label="Deleted Elements")

    process_button.click(
        fn=process_images,
        inputs=[old_image, new_image],
        outputs=[added_output, deleted_output],
    )

    # Launch the app
    demo.launch(
        show_api=False,  # Hides the "Use via API" link in the footer
        share=False,  # Do not allow sharing the link publicly
        pwa=True,  # Enables PWA support
        favicon_path="./favicon.ico",  # Correctly set the favicon path
        server_port=7866,  # Specify the port number
    )
