import argparse
import logging
import sys
import cv2 as cv
import numpy as np
import pydicom
import torch
from torchsummary import summary
from torchvision.transforms import transforms
from huggingface_hub import hf_hub_download
from itertools import combinations_with_replacement

from phase_classification import models

logger = logging.getLogger(__name__)


class DSAPhaseClassifier:
    def __init__(self, num_classes=4, img_size=224):
        model_path = hf_hub_download(
            repo_id="RuishengSu/autoTICI",
            filename="models/phase_classification/multi_frame_best.model"
        )
        self.num_classes = num_classes
        self.model = models.ResNet18(num_classes=self.num_classes, feature_extract=False, use_pretrained=False)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        summary(self.model, (3, img_size, img_size))
        self.model.load_state_dict(torch.load(model_path))
        self.validation_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.model.eval()
        self.transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def label_fit(self, label_softmax_prob_sequence):
        lbl_length = label_softmax_prob_sequence.shape[0]
        """Generate all possible valid label sequences"""
        combs = []
        for comb in combinations_with_replacement(range(self.num_classes + 1), lbl_length):
            valid = True
            comb = list(comb)
            for i in range(1, len(comb)):
                if not (comb[i - 1] <= comb[i] <= comb[i - 1] + 1):
                    valid = False
                    break
            if valid:
                comb = [0 if l == 4 else l for l in comb]
                combs.append(comb)

        """If input label list is a valid, return"""
        _, label_sequence = torch.max(label_softmax_prob_sequence, 1)
        label_sequence = label_sequence.tolist()
        if label_sequence in combs:
            logger.info("label sequence: {}".format(label_sequence))
            return label_sequence

        """"Otherwise, find the closest valid label sequence"""
        min_loss = None
        logical_label_sequence = label_sequence
        for comb in combs:
            loss = self.validation_loss_fn(label_softmax_prob_sequence, torch.tensor(comb).to(self.device)).item()
            if (min_loss is None) or (min_loss > loss):
                min_loss = loss
                logical_label_sequence = comb
        logger.info("label sequence: {}".format(logical_label_sequence))
        return logical_label_sequence

    def get_phase_borders(self, sequence_phases):
        try:
            arterial_phase_first_frame = sequence_phases.index(1)
            arterial_phase_last_frame = len(sequence_phases) - sequence_phases[::-1].index(1) - 1
        except ValueError:
            arterial_phase_first_frame, arterial_phase_last_frame = None, None

        try:
            parenchymal_phase_last_frame = len(sequence_phases) - sequence_phases[::-1].index(2) - 1
        except ValueError:
            parenchymal_phase_last_frame = None
        return arterial_phase_first_frame, arterial_phase_last_frame, parenchymal_phase_last_frame

    def predict_sequence_phases(self, dsa):
        """
        :dsa: It can be a multi-frame dicom file path or a 2D+t sequence in 3D array format.
        Time axis is expected to be first channel.
        """
        if isinstance(dsa, str):
            ds = pydicom.dcmread(dsa, defer_size="1 KB", stop_before_pixels=False, force=True)
            dsa = ds.pixel_array
            if dsa.ndim == 2:
                dsa = np.expand_dims(dsa, axis=0)

        sequence_as_model_input = []
        sequence_length = dsa.shape[0]
        for idx_frame in range(sequence_length):
            current_frame = dsa[idx_frame]
            previous_frame = dsa[max(0, idx_frame - 1)]
            next_frame = dsa[min(sequence_length - 1, idx_frame + 1)]

            image_input = np.dstack([previous_frame, current_frame, next_frame])

            '''Normalize image if input is not uint8'''
            if not isinstance(image_input[0, 0, 0], np.uint8):
                image_input = cv.normalize(image_input, None, 0, 255, cv.NORM_MINMAX)
                image_input = image_input.astype(np.uint8)

            image_input = self.transformations(image_input)
            sequence_as_model_input.append(image_input)
        sequence_as_model_input = torch.stack(sequence_as_model_input)

        sequence_as_model_input = sequence_as_model_input.to(self.device)
        with torch.no_grad():
            outputs = self.model(sequence_as_model_input)
        predicted_sequence_phases = self.label_fit(outputs)
        artery_first, artery_last, parenchymal_last = self.get_phase_borders(predicted_sequence_phases)

        return artery_first, artery_last, parenchymal_last, predicted_sequence_phases


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='Phase classification of DSA frames.')
    parser.add_argument('input', type=str, help='Input file dicom path')
    return parser.parse_args()


if __name__ == '__main__':
    '''configure logging'''
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    '''get input args'''
    args = parse_args()

    '''phase classification'''
    pc = DSAPhaseClassifier()
    output = pc.predict_sequence_phases(args.input)
    logger.info(output)
    logger.info("Done")
