import logging
import sys
import cv2 as cv
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import scipy.ndimage as ndimage
from scipy.ndimage import binary_erosion, binary_dilation

logger = logging.getLogger(__name__)


class RegionSegmentor:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="RuishengSu/autoTICI",
            filename="models/region_segmentation/segmentation_model.pth"
        )
        # Load pytorch model (nnUNet)
        self.model = torch.load(model_path)
        self.model.eval()

    def segment(self, minip_img, occ):
        logger.info("Predicting vascular territory on post-EVT MINIP.")
        test_image_array = np.expand_dims(minip_img, axis=0)
        test_image_tensor = torch.tensor(test_image_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(test_image_tensor)
        output_probs = torch.softmax(output, dim=1)
        predicted_mask = torch.argmax(output_probs, dim=1)
        # Create ICA mask by adding together ACA (label 1) and MCA (label 2)
        ica_mask = (predicted_mask == 1) | (predicted_mask == 2)
        # Create MCA mask directly from label 2 of nnUNet
        mca_mask = (predicted_mask == 1)

        '''Post-processing segmentation'''
        ica_mask = ica_mask.squeeze().cpu().numpy().astype(np.uint8)
        mca_mask = mca_mask.squeeze().cpu().numpy().astype(np.uint8)
        # Select right mask based on input
        if 'ICA' in occ:
            region_mask = ica_mask
        elif 'M1' in occ or 'M2' in occ:
            region_mask = mca_mask
        else:
            raise ValueError('Unknown occlusion location')

        # Perform morphological operations erosion, connected component analysis, dilation
        region_mask = binary_erosion(region_mask, iterations=5)
        labeled_mask, num_features = ndimage.label(region_mask)
        if num_features > 0:
            largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
        else:
            largest_component = np.zeros_like(region_mask, dtype=bool)
        region_mask = binary_dilation(largest_component, iterations=5)
        region_mask = region_mask.astype(np.uint8) * 255

        return region_mask


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    region_segmentator = RegionSegmentor()
    output = region_segmentator.segment(np.zeros((1024, 1024)), "ICA")
    logger.info(output)
    logger.info("Done")
