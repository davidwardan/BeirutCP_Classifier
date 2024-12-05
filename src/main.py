import gradio as gr
import cv2
import torch
import logging
from src.utils import utils, processing
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(top_classes=3, lime=False, shap=False):
    """
    Main function to run the classification UI
    :param top_classes: Number of top classes to display
    :param lime: Boolean to enable LIME explanation
    :param shap: Boolean to enable SHAP explanation
    :return: None
    """
    # Define configuration
    config = Config()

    # Load model
    logger.info("Loading model....")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.saved_model_dir != "":
        model = torch.load(config.saved_model_dir, map_location=device)
    else:
        model = torch.load("saved_models/SwinT", map_location=device)
    model.eval()

    # Check labels
    if not config.labels:
        raise ValueError("Config labels are not defined. Please set `config.labels`.")

    # Define image classification function
    def classify_image(inp):
        inp = cv2.resize(inp, (config.image_size, config.image_size))
        inp = processing.norm_image(inp)
        inp = (
            torch.tensor(inp.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            prediction = model(inp).softmax(dim=1).cpu().numpy().flatten()
        confidences = {
            config.labels[i]: float(prediction[i]) for i in range(len(config.labels))
        }

        if lime:
            explained_image = utils.Lime_explain_instance(
                model,
                inp.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                num_samples=1000,
                num_features=10,
            )
            return confidences, explained_image
        elif shap:
            explained_image = utils.shapley_explain_instance(
                model, inp.cpu(), labels=config.labels, evals=1000, top_labels=32
            )
            return confidences, explained_image
        else:
            return confidences

    # Define Gradio demo
    title = "Construction Period Classifier"
    description = "Upload an image to classify its construction period. Use LIME or SHAP for model interpretability."
    if lime or shap:
        demo = gr.Interface(
            fn=classify_image,
            inputs=gr.Image(label="Input Image"),
            outputs=[
                gr.Label(label="Top 3 classes", num_top_classes=top_classes),
                gr.Image(label="Model Interpretation", width=750, height=500),
            ],
            title=title,
            description=description,
        )
    else:
        demo = gr.Interface(
            fn=classify_image,
            inputs=gr.Image(label="Input Image"),
            outputs=gr.Label(num_top_classes=top_classes),
            title=title,
            description=description,
        )

    # Launch demo
    try:
        demo.launch(inbrowser=True, debug=True)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")


if __name__ == "__main__":
    main()
