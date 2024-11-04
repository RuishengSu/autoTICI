import logging
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


class LandmarkDetector:
    def __init__(self, view):
        model_path = hf_hub_download(
            repo_id="RuishengSu/autoTICI",
            filename="models/landmark_detection/{}_combined.h5".format(view)
        )
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.eval()

    def detect_landmarks(self, img):
        """Assuming input image is normalized to 0-1"""
        downsample_factor = img.shape[0] / 256
        img = cv.resize(img, (256, 256))[np.newaxis, :, :, np.newaxis]
        heatmaps = self.model.predict(img, verbose=0)
        ICA_heatmap = heatmaps[0, :, :, 0, 0]
        M1_heatmap = heatmaps[0, :, :, 1, 0]

        ICA_coords = np.unravel_index(ICA_heatmap.argmax(), ICA_heatmap.shape, order='F')
        ICA_coords = tuple(int(i * downsample_factor) for i in ICA_coords)
        M1_coords = np.unravel_index(M1_heatmap.argmax(), M1_heatmap.shape, order='F')
        M1_coords = tuple(int(i * downsample_factor) for i in M1_coords)

        logger.info("Predicted location of ICA: {}, M1: {}".format(ICA_coords, M1_coords))
        return ICA_coords, M1_coords


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    landmark_detector = LandmarkDetector('ap')
    output = landmark_detector.detect_landmarks(np.zeros((1024, 1024)))
    logger.info(output)
    logger.info("Done")
