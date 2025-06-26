import gradio as gr
import cv2
import torch
import logging
import os
import numpy as np
from src.utils import Utils as utils
from src.utils import Processing as processing
from config import Config
from src.swint_model import SwinTClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(top_classes=3):
    """
    Main function to run the classification UI
    :param top_classes: Number of top classes to display
    :return: None
    """
    # Define configuration
    config = Config()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model...")
    model = SwinTClassifier(
        num_classes=config.num_classes,
        transfer_learning=(config.transfer_learning == 1),
    ).to(device)
    model_path = os.path.join(config.saved_model_dir, "SwinT.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")

    # Define image classification function
    def classify_image(inp, enable_gradcam):
        # Convert PIL image to numpy array
        inp_np = np.array(inp)
        inp_resized = cv2.resize(inp_np, (config.image_size, config.image_size))
        inp_processed = processing.norm_image(inp_resized)
        inp_tensor = (
            torch.tensor(inp_processed.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            prediction = model(inp_tensor).softmax(dim=1).cpu().numpy().flatten()
        confidences = {
            config.labels[i]: float(prediction[i]) for i in range(len(config.labels))
        }

        if enable_gradcam:
            explained_image = utils.gradcam_explain_instance(model, inp, device)
            return confidences, explained_image
        else:
            return confidences, None

    title = "Construction Period Classifier"
    description = (
        "Upload an image to classify its construction period. "
        "Use the checkbox to enable Grad-CAM explanation."
    )

    # Create the interface with an additional checkbox for Grad-CAM
    demo = gr.Interface(
        fn=classify_image,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Checkbox(label="Enable Grad-CAM Explanation", value=False),
        ],
        outputs=[
            gr.Label(label="Top Classes", num_top_classes=top_classes),
            gr.Image(label="Model Interpretation", width=750, height=500),
        ],
        title=title,
        description=description,
    )

    # Launch the Gradio demo
    try:
        demo.launch(inbrowser=True, debug=True)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")


if __name__ == "__main__":
    main()
