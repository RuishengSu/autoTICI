import os
import sys
import argparse
from scipy.spatial import distance
from registration.prepost_registration import register_to_postEVT
from registration.transformation import warp_sequence
from registration.motion_correction import mc_sequence
from phase_classification.phase_predict import DSAPhaseClassifier
from landmark_detection import LandmarkDetector
from region_segmention import RegionSegmentor
from utils.utils import *

logger = logging.getLogger(__name__)


class AutoTICI:
    def __init__(self, landmark_preregistration_enabled, view):
        self.region_segmentor = RegionSegmentor()
        self.phase_classifier = DSAPhaseClassifier()
        if landmark_preregistration_enabled:
            self.landmark_detector = LandmarkDetector(view)

    def remove_venous_phase(self, sequence_array):
        if sequence_array.shape[0] == 1:
            return sequence_array

        arterial_phase_first_frame, arterial_phase_last_frame, parenchymal_phase_last_frame, _ = self.phase_classifier.predict_sequence_phases(
            sequence_array)
        logger.info("Predicted phase borders: [first_arterial, last_arterial, last_parenchymal] = [{}, {}, {}]".format(
            arterial_phase_first_frame, arterial_phase_last_frame, parenchymal_phase_last_frame))
        if arterial_phase_first_frame is None: arterial_phase_first_frame, arterial_phase_last_frame = 0, 0
        if parenchymal_phase_last_frame is None: parenchymal_phase_last_frame = sequence_array.shape[0] - 1

        return sequence_array[arterial_phase_first_frame:parenchymal_phase_last_frame + 1]

    def predict(self, preEVT, postEVT, occ, motion_correction_enabled=False, landmark_preregistration_enabled=False,
                 view='unknown', output_dirpath="./results"):

        logger.info("=" * 30 + "Configurations" + "=" * 30)
        logger.info("Input PreEVT: {}".format(preEVT))
        logger.info("Input PostEVT: {}".format(postEVT))
        logger.info("Apply motion correction on input DSA: {}".format(motion_correction_enabled))
        logger.info("Occlusion location: {}".format(occ))
        logger.info("-" * 70)

        '''Get preEVT and postEVT sequence, all resized to 1024x1024. type: uint8'''
        preEVT_seq, preEVT_pixel_spacing = read_sequence(preEVT)
        postEVT_seq, postEVT_pixel_spacing = read_sequence(postEVT)
        # Normalize sequence. Output image type: np.uint8
        preEVT_seq = normalize(preEVT_seq)
        postEVT_seq = normalize(postEVT_seq)
        preEVT_seq, preEVT_pixel_spacing = resize_to_1024(preEVT_seq, preEVT_pixel_spacing)
        postEVT_seq, postEVT_pixel_spacing = resize_to_1024(postEVT_seq, postEVT_pixel_spacing)
        if not (np.isnan(preEVT_pixel_spacing) or np.isnan(postEVT_pixel_spacing)):
            preEVT_seq, preEVT_pixel_spacing = resize_to_target_spacing(preEVT_seq, preEVT_pixel_spacing,
                                                                        target_spacing=postEVT_pixel_spacing)
        logger.info("preEVT spacing: {}, postEVT spacing: {}".format(preEVT_pixel_spacing, postEVT_pixel_spacing))
        logger.info('PreEVT ({:.0f}+/-{:.0f}, {}), PostEVT ({:.0f}+/-{:.0f}, {})'
                    ''.format(np.mean(preEVT_seq), np.std(preEVT_seq), preEVT_seq.dtype,
                              np.mean(postEVT_seq), np.std(postEVT_seq), postEVT_seq.dtype))

        col_label = cv.putText(img=np.zeros((100, 1124)), text="Original", org=(350, 70),
                               fontFace=3, fontScale=3, color=255, thickness=5)
        row_label_preEVT = cv.rotate(cv.putText(img=np.zeros((100, 1024)), text="PreEVT", org=(350, 90),
                                                fontFace=3, fontScale=3, color=255, thickness=5),
                                     cv.ROTATE_90_COUNTERCLOCKWISE)
        row_label_postEVT = cv.rotate(cv.putText(img=np.zeros((100, 1024)), text="PostEVT", org=(350, 90),
                                                 fontFace=3, fontScale=3, color=255, thickness=5),
                                      cv.ROTATE_90_COUNTERCLOCKWISE)
        vis = np.concatenate([col_label, np.concatenate([row_label_preEVT, minip(preEVT_seq)], axis=1),
                              np.concatenate([row_label_postEVT, minip(postEVT_seq)], axis=1)])
        vis = np.dstack([vis, vis, vis])

        '''Remove black border and text from image'''
        preEVT_seq = remove_text_and_border(preEVT_seq)
        postEVT_seq = remove_text_and_border(postEVT_seq)

        '''Motion Correction. Output image type: uint8'''
        if motion_correction_enabled:
            try:
                preEVT_seq = mc_sequence(preEVT_seq)
                postEVT_seq = mc_sequence(postEVT_seq)
            except:
                logger.error("Motion correction corrupted!")
            col_label = cv.putText(img=np.zeros((100, 1024, 3)), text="MC", org=(450, 70),
                                   fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
            vis = np.concatenate([vis, np.concatenate(
                [col_label, np.dstack([minip(preEVT_seq) for _ in range(3)]),
                 np.dstack([minip(postEVT_seq) for _ in range(3)])])], axis=1)

        '''Registering preEVT to postEVT'''
        if landmark_preregistration_enabled and view in ['ap', 'lateral']:
            landmark_detector = LandmarkDetector(view)
            preEVT_ICA_coords, preEVT_M1_coords = landmark_detector.detect_landmarks(minip(normalize_0_1(preEVT_seq)),
                                                                                     view)
            postEVT_ICA_coords, postEVT_M1_coords = landmark_detector.detect_landmarks(
                minip(normalize_0_1(postEVT_seq)), view)

            col_label = cv.putText(img=np.zeros((100, 1024, 3)), text="Landmark", org=(50, 70),
                                   fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
            vis_preEVT_landmark = np.dstack([minip(preEVT_seq) for _ in range(3)])
            vis_preEVT_landmark = cv.circle(vis_preEVT_landmark, preEVT_ICA_coords, 10, (0, 0, 255), -1)
            vis_preEVT_landmark = cv.circle(vis_preEVT_landmark, preEVT_M1_coords, 10, (255, 0, 0), 3)
            vis_postEVT_landmark = np.dstack([minip(postEVT_seq) for _ in range(3)])
            vis_postEVT_landmark = cv.circle(vis_postEVT_landmark, postEVT_ICA_coords, 10, (0, 0, 255), -1)
            vis_postEVT_landmark = cv.circle(vis_postEVT_landmark, postEVT_M1_coords, 10, (255, 0, 0), 3)

            vis = np.concatenate([vis, np.concatenate(
                [col_label, vis_preEVT_landmark, vis_postEVT_landmark])], axis=1)

            ICA_translation_matrix = np.float32([[1, 0, postEVT_ICA_coords[0] - preEVT_ICA_coords[0]],
                                                 [0, 1, postEVT_ICA_coords[1] - preEVT_ICA_coords[1]]])
            if distance.euclidean(preEVT_ICA_coords, postEVT_ICA_coords) < 250:
                preEVT_seq = warp_sequence(preEVT_seq, ICA_translation_matrix)

            col_label = cv.putText(img=np.zeros((100, 1024, 3)), text="Translated", org=(50, 70),
                                   fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
            vis = np.concatenate([vis, np.concatenate(
                [col_label, np.dstack([minip(preEVT_seq) for _ in range(3)]),
                 np.dstack([minip(postEVT_seq) for _ in range(3)])])], axis=1)

        '''Create segmentation on post EVT MinIP using nnUNet'''
        region_segmentor = RegionSegmentor()
        TDT_mask = region_segmentor.segment(minip(postEVT_seq), occ)

        """Register preEVT to postEVT"""
        logger.info("Registering preEVT to postEVT MinIP.")
        preEVT_seq, postEVT_seq = register_to_postEVT(preEVT_seq, postEVT_seq)
        '''Create contour points for TDT'''
        minip_preEVT = minip(normalize(preEVT_seq))
        minip_postEVT = minip(normalize(postEVT_seq))
        minip_preEVT = np.dstack([minip_preEVT, minip_preEVT, minip_preEVT])
        minip_postEVT = np.dstack([minip_postEVT, minip_postEVT, minip_postEVT])
        TDT_contour_points, _ = cv.findContours(TDT_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cv.drawContours(minip_preEVT, TDT_contour_points, -1, (255, 255, 255), thickness=8)
        cv.drawContours(minip_postEVT, TDT_contour_points, -1, (255, 255, 255), thickness=8)
        '''Visualization'''
        label_image = cv.putText(img=np.zeros((100, 1024, 3)), text="Reg+Seg", org=(250, 70),
                                 fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
        vis = np.concatenate([vis, np.concatenate([label_image, minip_preEVT, minip_postEVT])], axis=1)

        '''Remove venous phase frame from sequence using the trained phase classification model.'''
        logger.info("Removing venous phases")
        preEVT_seq = self.remove_venous_phase(preEVT_seq)
        postEVT_seq = self.remove_venous_phase(postEVT_seq)
        '''Visualization'''
        col_label = cv.putText(img=np.zeros((100, 1024, 3)), text="Venous removal", org=(150, 70),
                               fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
        vis = np.concatenate([vis, np.concatenate(
            [col_label, np.dstack([minip(preEVT_seq) for _ in range(3)]),
             np.dstack([minip(postEVT_seq) for _ in range(3)])])], axis=1)

        '''TICI score quantification'''
        autoTICI, vis_pipeline = segment_minip(preEVT_seq.astype(np.float32), postEVT_seq.astype(np.float32),
                                               TDT_mask)
        logger.info("autoTICI score: {}".format(autoTICI))

        '''Visualization of the full pipeline'''
        vis = np.concatenate([vis, vis_pipeline], axis=1)
        save_fig(os.path.join(output_dirpath, 'pipeline.png'), vis)


def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='autoTICI: automatic brain reperfusion scoring (eTICI) for ischemic stroke')
    parser.add_argument("preEVT", type=str, help="Input dicom file of preEVT DSA.")
    parser.add_argument("postEVT", type=str, help="Input dicom file of postEVT DSA. Make sure pre and post have the same ap/lateral view.")
    parser.add_argument("occ", type=str, choices=['ICA', 'M1', 'M2'], help="Occlusion location: [ICA, M1, M2].")
    parser.add_argument("-m", "--motion_correction_enabled", action='store_true', help="Whether to perform motion correction of each DSA series.")
    parser.add_argument("-l", "--landmark_preregistration_enabled", action='store_true', help="Whether to perform landmark-based pre-registration between pre- and post-EVT DSA.")
    parser.add_argument("-v", "--view", default='ap', type=str, choices=['ap', 'lateral'], help="DSA view: [ap, lateral], only needed for landmark-based detection.")
    parser.add_argument("-o", "--output_dirpath", default="./output", type=str, help="Output directory path.")

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args()
    autoTICI = AutoTICI(args.landmark_preregistration_enabled, args.view)
    autoTICI.predict(**vars(args))
    logger.info("Done")
    print("Done")
