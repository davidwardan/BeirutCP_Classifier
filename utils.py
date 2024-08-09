import numpy as np
import matplotlib.pyplot as plt


class utils:
    def __init__(self, in_dir: str):
        self.in_dir = in_dir

    @staticmethod
    def load_data(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=True)

    @staticmethod
    def norm_image(x: np.ndarray) -> np.ndarray:
        return x / 255.0

    @staticmethod
    def denorm_image(x: np.ndarray) -> np.ndarray:
        return x * 255.0

    @staticmethod
    def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        return np.eye(num_classes)[y]

    # TODO: Set the size of the image in function of n and img_per_row (ensures that the images are not too small)
    @staticmethod
    def plot_n_images(images: np.ndarray, n: int, img_per_row: int, save_path: str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 5))
        for i in range(n):
            plt.subplot(int(n / img_per_row), img_per_row, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def Lime_explain_instance(model, image, num_samples: int, num_features: int, save_path: str = None):
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image.astype('double'), model.predict,
                                          top_labels=3, hide_color=0, num_samples=num_samples)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,
                                                    num_features=num_features, hide_rest=False)
        # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        # if save_path is not None:
        #     plt.savefig(save_path, bbox_inches='tight')
        # else:
        #     plt.show()

        # return image with mask
        return mark_boundaries(temp / 2 + 0.5, mask)
