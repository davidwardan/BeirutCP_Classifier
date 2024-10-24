import gradio as gr
import cv2
import keras

from src.utils import utils
from src.utils import processing
from config import Config


def main(top_classes=3, lime=False, shap=False):
    """
    Main function to run the classification UI
    :param top_classes: Number of top classes to display
    :param lime: Boolean to enable LIME explanation
    :param shap: Boolean to enable SHAP explanation
    :return: None

    Note: The function is not returning anything, it is launching the UI
    If lime or shap are set to True, the UI will display the explanation image
    If lime and shao are set to True, the UI will display the LIME explanation image
    """

    # define configuration
    config = Config()

    # load saved model
    print("loading model....")
    if config.saved_model_dir != "":
        model = keras.models.load_model(config.saved_model_dir)
    else:
        model = keras.saving.load_model('saved_models/SwinT')

    # define interface
    def classify_image(inp):
        # inp = inp[..., ::-1]  # BGR to RGB
        inp = cv2.resize(inp, (config.image_size, config.image_size))
        inp = inp.reshape(1, config.image_size, config.image_size, 3)
        inp = processing.norm_image(inp)
        prediction = model.predict(inp).flatten()
        confidences = {config.labels[i]: float(prediction[i]) for i in range(len(config.labels))}
        if lime:
            # TODO: Solve the issue with the image being displayed with burnt colors
            explained_image = utils.Lime_explain_instance(model, inp.reshape(config.image_size, config.image_size, 3),
                                                          num_samples=1000, num_features=10)
            return confidences, explained_image
        elif shap:
            explained_image = utils.shapley_explain_instance(model, inp, labels=config.labels,
                                                             evals=1000, top_labels=32)
            return confidences, explained_image
        else:
            return confidences

    # define demo
    if lime or shap:
        demo = gr.Interface(
            fn=classify_image,
            inputs=gr.Image(label="Input Image"),
            outputs=[gr.Label(label='Top 3 classes', num_top_classes=top_classes),
                     gr.Image(label="Model Interpretation", width=750, height=500)],
            title="Construction Period Classifier",
            description="Upload an image to classify",
        )
    else:
        demo = gr.Interface(
            fn=classify_image,
            inputs=gr.Image(label="Input Image"),
            outputs=gr.Label(num_top_classes=top_classes),
            title="Construction Period Classifier",
            description="Upload an image to classify",
        )

    # launch demo
    demo.launch(inbrowser=True, debug=True)


if __name__ == "__main__":
    main()
