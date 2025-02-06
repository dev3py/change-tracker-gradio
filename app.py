import cv2
import numpy as np
import gradio as gr
import tempfile
import imageio


# ----------------- Helper Functions (OpenCV Processing) ----------------- #
def align_images(old_img, new_img):
    """
    Align the old image to the new image using feature matching.
    Both images are assumed to be in BGR.
    """
    # Convert images to grayscale
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(old_gray, None)
    kp2, des2 = orb.detectAndCompute(new_gray, None)

    # Match features using brute-force matching with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matching keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix using RANSAC
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
    # Align the old image to the new image
    old_aligned = align_images(old_img, new_img)

    # Compute absolute difference between the new image and aligned old image
    diff = cv2.absdiff(new_img, old_aligned)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to capture areas with significant differences (new content)
    _, thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Clean the mask using morphological operations
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
    # Align the old image to the new image
    aligned_old = align_images(old_img, new_img)

    # Convert both images to grayscale
    old_gray = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # For deleted elements, subtract the old from the new (white where removed)
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


def highlight_additions_on_background_cv2(old_img, new_img, style="Only-Changed"):
    # """
    # For non-animated outputs:
    #   - If style is "Only-Changed": return a white background with only the added regions (from new image) visible.
    #   - If style is "Greyscale": return a Greyscale version of the new image with the added regions in full color.
    #   - If style is "Specific-Colors": return a Greyscale background with added regions forced to the user-specified color.
    # """
    added_mask = detect_added_elements(old_img, new_img)

    if style == "Only-Changed":
        background = np.full(new_img.shape, 255, dtype=np.uint8)
        result = background.copy()
        result[added_mask == 255] = new_img[added_mask == 255]
    elif style == "Greyscale":
        # Convert the entire new image to Greyscale, then back to BGR
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        # Overlay the added regions in full color
        result[added_mask == 255] = new_img[added_mask == 255]
    elif style == "Specific-Colors":
        # Convert the entire new image to Greyscale, then back to BGR
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        # Force the added regions to the user-specified color (handled in process_images)
        # (The color will be applied in process_images.)
        # Here we simply mark the regions.
        result[added_mask == 255] = result[
            added_mask == 255
        ]  # Placeholder; will be overridden.
    else:
        result = None
    return result, added_mask


def highlight_deletions_on_background_cv2(old_img, new_img, style="Only-Changed"):
    """
    For non-animated outputs:
      - If style is "Only-Changed": return a white background with only the deleted regions (from aligned old image) visible.
      - If style is "Greyscale": return a Greyscale version of the aligned old image with the deleted regions in full color.
      - If style is "Specific-Colors": return a Greyscale background with deleted regions forced to the user-specified color.
    """
    deleted_mask, aligned_old = detect_deleted_elements(old_img, new_img)

    if style == "Only-Changed":
        background = np.full(aligned_old.shape, 255, dtype=np.uint8)
        result = background.copy()
        result[deleted_mask == 255] = aligned_old[deleted_mask == 255]
    elif style == "Greyscale":
        gray = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        result[deleted_mask == 255] = aligned_old[deleted_mask == 255]
    elif style == "Specific-Colors":
        gray = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        # Force the deleted regions to the user-specified color (handled in process_images)
        # (The color will be applied in process_images.)
        # Here we simply mark the regions.
        result[deleted_mask == 255] = result[
            deleted_mask == 255
        ]  # Placeholder; will be overridden.
    else:
        result = None
    return result, deleted_mask, aligned_old


def hex_to_bgr(color_string):
    """
    Convert a color string to a BGR numpy array.
    Supports hexadecimal (e.g., "#FF0000"), rgb (e.g., "rgb(255, 0, 0)"), and rgba (e.g., "rgba(0, 38.143060151558984, 255, 1)") formats.
    """
    print("color_string:", color_string)
    color_string = color_string.strip()
    if color_string.startswith("#"):
        # Hexadecimal format.
        hex_color = color_string.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)
    elif color_string.lower().startswith("rgba"):
        # RGBA format: ignore the alpha channel.
        inside = color_string[color_string.find("(") + 1 : color_string.find(")")]
        parts = inside.split(",")
        if len(parts) < 3:
            raise ValueError("Invalid rgba color format.")
        r = int(float(parts[0].strip()))
        g = int(float(parts[1].strip()))
        b = int(float(parts[2].strip()))
        return np.array([b, g, r], dtype=np.uint8)
    elif color_string.lower().startswith("rgb"):
        # RGB format.
        inside = color_string[color_string.find("(") + 1 : color_string.find(")")]
        parts = inside.split(",")
        if len(parts) < 3:
            raise ValueError("Invalid rgb color format.")
        r = int(float(parts[0].strip()))
        g = int(float(parts[1].strip()))
        b = int(float(parts[2].strip()))
        return np.array([b, g, r], dtype=np.uint8)
    else:
        # Fallback: try treating it as a hex string.
        hex_color = color_string.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)


# ----------------- Gradio Interface Function ----------------- #
def process_images(old_img, new_img, output_style, added_color, removed_color):
    """
    Expects two images (old and new) in RGB format, an output style, and two color values.
    Converts them to BGR for processing, then returns either:
      - For "Animation": two animated GIF file paths (one for added and one for deleted)
      - For static outputs ("Only-Changed", "Greyscale", "Specific-Colors"):
          two static images (converted back to RGB) for display.
    """
    if old_img is None or new_img is None:
        return None, None

    # Convert from RGB (Gradio default) to BGR (OpenCV default)
    old_bgr = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR)
    new_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    if output_style == "Animation":
        # ----- Generate Animation for "Added" Regions -----
        added_mask = detect_added_elements(old_bgr, new_bgr)
        # Base: Greyscale version of new image
        gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
        gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
        num_frames = 10  # Increase the number of frames for a longer Animation
        added_frames = []
        for i in range(num_frames):
            frame = gray_new_bgr.copy()
            # Blink: overlay the colored regions every other frame
            if i % 2 == 0:
                frame[added_mask == 255] = new_bgr[added_mask == 255]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            added_frames.append(frame_rgb)

        # ----- Generate Animation for "Deleted" Regions -----
        deleted_mask, aligned_old = detect_deleted_elements(old_bgr, new_bgr)
        gray_old = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_old_bgr = cv2.cvtColor(gray_old, cv2.COLOR_GRAY2BGR)
        deleted_frames = []
        for i in range(num_frames):
            frame = gray_old_bgr.copy()
            if i % 2 == 0:
                frame[deleted_mask == 255] = aligned_old[deleted_mask == 255]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            deleted_frames.append(frame_rgb)

        # Save Animations as temporary GIF files with infinite looping.
        temp_added = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        temp_deleted = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        temp_added.close()
        temp_deleted.close()
        # Use loop=0 for infinite loops, adjust duration as needed.
        imageio.mimsave(temp_added.name, added_frames, duration=0.5, loop=0)
        imageio.mimsave(temp_deleted.name, deleted_frames, duration=0.5, loop=0)
        return temp_added.name, temp_deleted.name

    elif output_style == "Specific-Colors":
        # For Specific-Colors, use a Greyscale background with added regions forced to the user-specified color and deleted regions forced to the user-specified color.
        # Process Added Regions:
        added_mask = detect_added_elements(old_bgr, new_bgr)
        gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
        gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
        user_added_color = hex_to_bgr(added_color)  # default red (#FF0000)
        added_result_bgr = gray_new_bgr.copy()
        added_result_bgr[added_mask == 255] = user_added_color

        # Process Deleted Regions:
        deleted_mask, aligned_old = detect_deleted_elements(old_bgr, new_bgr)
        gray_old = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_old_bgr = cv2.cvtColor(gray_old, cv2.COLOR_GRAY2BGR)
        user_removed_color = hex_to_bgr(removed_color)  # default yellow (#FFFF00)
        deleted_result_bgr = gray_old_bgr.copy()
        deleted_result_bgr[deleted_mask == 255] = user_removed_color

        added_result = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
        deleted_result = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)
        return added_result, deleted_result

    else:
        # For "Only-Changed" and "Greyscale" modes.
        added_result_bgr, _ = highlight_additions_on_background_cv2(
            old_bgr, new_bgr, style=output_style
        )
        deleted_result_bgr, _, _ = highlight_deletions_on_background_cv2(
            old_bgr, new_bgr, style=output_style
        )
        added_result = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
        deleted_result = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)
        return added_result, deleted_result


# ----------------- Build the Gradio Interface ----------------- #
with gr.Blocks(
    title="Change Tracker",
    theme=gr.themes.Monochrome(),
    css="footer{display:none !important}",
) as demo:

    with gr.Row():
        gr.Markdown("# Change Tracker")
        with gr.Group():
            toggle_dark = gr.Button(value="Toggle Dark Mode")

    # JavaScript to toggle the 'dark' class on the body
    toggle_dark.click(
        None,
        js="""
            () => {
                document.body.classList.toggle('dark');
            }
            """,
    )
    gr.Markdown(
        """
            Upload the **Previous Image** and the **Latest Image**  and select an output style.
            
            - **Only-Changed**: A white background with only the changed regions visible.
            - **Greyscale**: A Greyscale version of the image with the changed regions in full color.
            - **Animation**: Similar to Greyscale, but the changed regions blink to attract attention.
            - **Specific-Colors**: A Greyscale background with added regions in a user-specified color and removed regions in a user-specified color.
            """
    )

    with gr.Row():
        old_image = gr.Image(label="Previous", type="numpy")
        new_image = gr.Image(label="Latest", type="numpy")

    # Add examples that will populate the "Old Image" and "New Image" fields when clicked.
    gr.Examples(
        examples=[
            ["examples/old.png", "examples/new.png"],
            ["examples/old4.png", "examples/new4.png"],
        ],
        inputs=[old_image, new_image],
        label="Example Images",
    )

    # Radio button for output style selection (four options now)
    output_style = gr.Radio(
        choices=["Only-Changed", "Greyscale", "Animation", "Specific-Colors"],
        label="Output Style",
        value="Animation",
    )

    # Color pickers for Specific-Colors mode; initially hidden.
    with gr.Row():
        added_color_picker = gr.ColorPicker(
            label="Added Color", value="#FF0000", visible=False
        )
        removed_color_picker = gr.ColorPicker(
            label="Removed Color", value="#FFFF00", visible=False
        )

    # Toggle the visibility of the color pickers based on the output style.
    def toggle_color_pickers(style):
        if style == "Specific-Colors":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    output_style.change(
        fn=toggle_color_pickers,
        inputs=output_style,
        outputs=[added_color_picker, removed_color_picker],
    )

    process_button = gr.Button("Process")

    with gr.Tabs():
        with gr.Tab("Added"):
            added_output = gr.Image(label="Added (Output)")
        with gr.Tab("Removed"):
            deleted_output = gr.Image(label="Removed (Output)")

    process_button.click(
        fn=process_images,
        inputs=[
            old_image,
            new_image,
            output_style,
            added_color_picker,
            removed_color_picker,
        ],
        outputs=[added_output, deleted_output],
    )

    # Footer
    gr.HTML(
        "<div style='bottom: 0; width: 100%; text-align: center; padding: 10px;'>Powered by Gollab</div>"
    )

    # Launch the app
    demo.launch(
        show_api=False,  # Hides the "Use via API" link in the footer
        share=False,  # Do not allow sharing the link publicly
        pwa=True,  # Enables PWA support
        favicon_path="./favicon.ico",  # Correctly set the favicon path
        server_port=7866,  # Specify the port number
        # auth=[("gollabadmin", "nipl18"), ("admin", "gollab18")],
    )
