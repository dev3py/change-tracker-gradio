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


def detect_color_changed_elements(old_img, new_img, threshold):
    """
    Detect pixels where the color difference between new_img and old_img exceeds the threshold.
    Only considers pixels where both images are not nearly white.
    Returns a binary mask.
    """
    diff = np.linalg.norm(
        new_img.astype(np.float32) - old_img.astype(np.float32), axis=2
    )
    mask = diff > threshold
    not_white_old = np.all(old_img < 240, axis=2)
    not_white_new = np.all(new_img < 240, axis=2)
    combined = mask & not_white_old & not_white_new
    return (combined.astype(np.uint8)) * 255


def highlight_additions_on_background_cv2(old_img, new_img, style="Only-Changed"):
    # For non-animated outputs:
    #   - If style is "Only-Changed": return a white background with only the added regions (from new image) visible.
    #   - If style is "Greyscale": return a Greyscale version of the new image with the added regions in full color.
    #   - If style is "Specific-Colors": return a Greyscale background with added regions forced to the user-specified color.
    added_mask = detect_added_elements(old_img, new_img)

    if style == "Only-Changed":
        background = np.full(new_img.shape, 255, dtype=np.uint8)
        result = background.copy()
        result[added_mask == 255] = new_img[added_mask == 255]
    elif style == "Greyscale":
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        result[added_mask == 255] = new_img[added_mask == 255]
    elif style == "Specific-Colors":
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = gray_bgr.copy()
        # Placeholder; actual color override is applied in process_images.
        result[added_mask == 255] = result[added_mask == 255]
    else:
        result = None
    return result, added_mask


def highlight_deletions_on_background_cv2(old_img, new_img, style="Only-Changed"):
    # For non-animated outputs:
    #   - If style is "Only-Changed": return a white background with only the deleted regions (from aligned old image) visible.
    #   - If style is "Greyscale": return a Greyscale version of the aligned old image with the deleted regions in full color.
    #   - If style is "Specific-Colors": return a Greyscale background with deleted regions forced to the user-specified color.
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
        # Placeholder; actual color override is applied in process_images.
        result[deleted_mask == 255] = result[deleted_mask == 255]
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
        hex_color = color_string.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)
    elif color_string.lower().startswith("rgba"):
        inside = color_string[color_string.find("(") + 1 : color_string.find(")")]
        parts = inside.split(",")
        if len(parts) < 3:
            raise ValueError("Invalid rgba color format.")
        r = int(float(parts[0].strip()))
        g = int(float(parts[1].strip()))
        b = int(float(parts[2].strip()))
        return np.array([b, g, r], dtype=np.uint8)
    elif color_string.lower().startswith("rgb"):
        inside = color_string[color_string.find("(") + 1 : color_string.find(")")]
        parts = inside.split(",")
        if len(parts) < 3:
            raise ValueError("Invalid rgb color format.")
        r = int(float(parts[0].strip()))
        g = int(float(parts[1].strip()))
        b = int(float(parts[2].strip()))
        return np.array([b, g, r], dtype=np.uint8)
    else:
        hex_color = color_string.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)


def detect_color_changed_elements(old_img, new_img, threshold):
    """
    Detect pixels where the color difference between new_img and old_img exceeds the threshold.
    Returns a binary mask.
    """
    diff = np.linalg.norm(
        new_img.astype(np.float32) - old_img.astype(np.float32), axis=2
    )
    mask = diff > threshold
    not_white_old = np.all(old_img < 240, axis=2)
    not_white_new = np.all(new_img < 240, axis=2)
    combined = mask & not_white_old & not_white_new
    return (combined.astype(np.uint8)) * 255


# ----------------- Gradio Interface Function ----------------- #
def process_images(
    old_img,
    new_img,
    output_style,
    added_color,
    removed_color,
    changed_color,
    color_threshold,
):
    """
    Expects two images (old and new) in RGB format, an output style, and three color values plus a threshold.
    Converts them to BGR for processing, then returns three outputs:
      - Added output (result for added regions)
      - Removed output (result for removed regions)
      - Color-Changed output (result for pixels with significant color changes)

    The Color-Changed output is always computed using the aligned old image (to match new image dimensions)
    so that the subtraction works correctly. If the selected output style is "Color-Changed",
    the user-specified changed color is used; otherwise, a default red (#FF0000) is used.
    """
    if old_img is None or new_img is None:
        return None, None, None

    # Convert input images from RGB to BGR.
    old_bgr = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR)
    new_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    # Branch for processing Added and Removed outputs based on output_style.
    if output_style == "Animation":
        # Process Added output as an animated GIF.
        added_mask = detect_added_elements(old_bgr, new_bgr)
        gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
        gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
        num_frames = 10
        added_frames = []
        for i in range(num_frames):
            frame = gray_new_bgr.copy()
            if i % 2 == 0:
                frame[added_mask == 255] = new_bgr[added_mask == 255]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            added_frames.append(frame_rgb)
        # Process Removed output as an animated GIF.
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
        temp_added = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        temp_deleted = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        temp_added.close()
        temp_deleted.close()
        imageio.mimsave(temp_added.name, added_frames, duration=0.5, loop=0)
        imageio.mimsave(temp_deleted.name, deleted_frames, duration=0.5, loop=0)
        added_out = temp_added.name
        deleted_out = temp_deleted.name

    elif output_style == "Specific-Colors":
        # Use user-specified colors for Added and Removed outputs.
        added_mask = detect_added_elements(old_bgr, new_bgr)
        gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
        gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
        user_added_color = hex_to_bgr(added_color)
        added_result_bgr = gray_new_bgr.copy()
        added_result_bgr[added_mask == 255] = user_added_color

        deleted_mask, aligned_old = detect_deleted_elements(old_bgr, new_bgr)
        gray_old = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_old_bgr = cv2.cvtColor(gray_old, cv2.COLOR_GRAY2BGR)
        user_removed_color = hex_to_bgr(removed_color)
        deleted_result_bgr = gray_old_bgr.copy()
        deleted_result_bgr[deleted_mask == 255] = user_removed_color

        added_out = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
        deleted_out = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)

    elif output_style == "Color-Changed":
        # For Color-Changed style, process Added and Removed outputs like Specific-Colors.
        added_mask = detect_added_elements(old_bgr, new_bgr)
        gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
        gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
        user_added_color = hex_to_bgr(added_color)
        added_result_bgr = gray_new_bgr.copy()
        added_result_bgr[added_mask == 255] = user_added_color

        deleted_mask, aligned_old = detect_deleted_elements(old_bgr, new_bgr)
        gray_old = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY)
        gray_old_bgr = cv2.cvtColor(gray_old, cv2.COLOR_GRAY2BGR)
        user_removed_color = hex_to_bgr(removed_color)
        deleted_result_bgr = gray_old_bgr.copy()
        deleted_result_bgr[deleted_mask == 255] = user_removed_color

        added_out = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
        deleted_out = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)

    else:
        # For "Only-Changed", "Greyscale", etc.
        added_result_bgr, _ = highlight_additions_on_background_cv2(
            old_bgr, new_bgr, style=output_style
        )
        deleted_result_bgr, _, _ = highlight_deletions_on_background_cv2(
            old_bgr, new_bgr, style=output_style
        )
        added_out = cv2.cvtColor(added_result_bgr, cv2.COLOR_BGR2RGB)
        deleted_out = cv2.cvtColor(deleted_result_bgr, cv2.COLOR_BGR2RGB)

    # Always compute the Color-Changed output.
    # Always align the old image to new image for computing the color difference.
    aligned_old_for_color = align_images(old_bgr, new_bgr)
    changed_mask = detect_color_changed_elements(
        aligned_old_for_color, new_bgr, color_threshold
    )
    gray_new = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
    gray_new_bgr = cv2.cvtColor(gray_new, cv2.COLOR_GRAY2BGR)
    # Use the user-specified changed color if the output style is "Color-Changed",
    # otherwise default to red (#FF0000).
    if output_style == "Color-Changed":
        user_changed_color = hex_to_bgr(changed_color)
    else:
        user_changed_color = hex_to_bgr("#FF0000")
    color_changed_result_bgr = gray_new_bgr.copy()
    color_changed_result_bgr[changed_mask == 255] = user_changed_color
    color_changed_result = cv2.cvtColor(color_changed_result_bgr, cv2.COLOR_BGR2RGB)

    return added_out, deleted_out, color_changed_result


# ----------------- Build the Gradio Interface ----------------- #
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""
with gr.Blocks(
    title="Change Tracker",
    theme=gr.themes.Monochrome().set(
        loader_color="#FF0000",
        # slider_color="#FF0000",
    ),
    js=js_func,
    css="footer{display:none !important}",
) as demo:

    with gr.Row():
        gr.Markdown("# Change Tracker")
        with gr.Group():
            toggle_dark = gr.Button(value="Toggle Dark Mode")

    toggle_dark.click(
        None,
        js="""() => { document.body.classList.toggle('dark'); }""",
    )
    gr.Markdown(
        """
        Upload the **Previous Image** and the **Latest Image** and select an output style.
        
        - **Only-Changed**: A white background with only the changed regions visible.
        - **Greyscale**: A Greyscale version of the image with the changed regions in full color.
        - **Animation**: Similar to Greyscale, but the changed regions blink to attract attention.
        - **Specific-Colors**: A Greyscale background with added regions in a user-specified color and removed regions in a user-specified color.
        - **Color-Changed**: A Greyscale background with elements whose color has changed over a threshold shown in a user-specified color.
        """
    )

    with gr.Row():
        old_image = gr.Image(label="Previous", type="numpy")
        new_image = gr.Image(label="Latest", type="numpy")

    gr.Examples(
        examples=[
            ["examples/old.png", "examples/new.png"],
            ["examples/old4.png", "examples/new4.png"],
        ],
        inputs=[old_image, new_image],
        label="Example Images",
    )

    output_style = gr.Radio(
        choices=[
            "Only-Changed",
            "Greyscale",
            "Animation",
            "Specific-Colors",
            "Color-Changed",
        ],
        label="Output Style",
        value="Animation",
    )

    with gr.Row():
        added_color_picker = gr.ColorPicker(
            label="Added Color", value="#FF0000", visible=False
        )
        removed_color_picker = gr.ColorPicker(
            label="Removed Color", value="#FFFF00", visible=False
        )
        changed_color_picker = gr.ColorPicker(
            label="Changed Color", value="#FF0000", visible=False
        )
        color_threshold_slider = gr.Slider(
            label="Color Change Threshold",
            minimum=0,
            maximum=255,
            value=50,
            step=1,
            visible=False,
        )

    def toggle_color_pickers(style):
        if style == "Specific-Colors":
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif style == "Color-Changed":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    output_style.change(
        fn=toggle_color_pickers,
        inputs=output_style,
        outputs=[
            added_color_picker,
            removed_color_picker,
            changed_color_picker,
            color_threshold_slider,
        ],
    )

    process_button = gr.Button("Process")

    with gr.Tabs():
        with gr.Tab("Added"):
            added_output = gr.Image(label="Added (Output)")
        with gr.Tab("Removed"):
            deleted_output = gr.Image(label="Removed (Output)")
        with gr.Tab("Color-Changed"):
            color_changed_output = gr.Image(label="Color-Changed (Output)")

    process_button.click(
        fn=process_images,
        inputs=[
            old_image,
            new_image,
            output_style,
            added_color_picker,
            removed_color_picker,
            changed_color_picker,
            color_threshold_slider,
        ],
        outputs=[added_output, deleted_output, color_changed_output],
    )

    gr.HTML(
        "<div style='bottom: 0; width: 100%; text-align: center; padding: 10px;'>Powered by Gollab</div>"
    )

    demo.launch(
        show_api=False,
        share=False,
        pwa=True,
        favicon_path="./favicon.ico",
        server_port=7866,
    )
