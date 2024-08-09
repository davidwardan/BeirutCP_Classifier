import gradio as gr
import cv2
import keras

from utils import utils
from config import Config


def main(top_classes=3, explain=False):
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
        inp = utils.norm_image(inp)
        prediction = model.predict(inp).flatten()
        confidences = {config.labels[i]: float(prediction[i]) for i in range(len(config.labels))}
        if explain:
            # TODO: Solve the issue with the image being displayed with burnt colors
            explained_image = utils.Lime_explain_instance(model, inp.reshape(config.image_size, config.image_size, 3),
                                                          num_samples=1000, num_features=10)
            return confidences, explained_image
        else:
            return confidences

    # define demo
    if explain:
        demo = gr.Interface(
            fn=classify_image,
            inputs=gr.Image(label="Input Image"),
            outputs=[gr.Label(label='Top 3 classes', num_top_classes=top_classes),
                     gr.Image(label="Model Interpretation")],
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
    main(explain=True)
