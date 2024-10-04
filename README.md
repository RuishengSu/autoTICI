# autoTICI

---
license: other
license_name: non-commercial-mit
license_link: LICENSE
---


### autoTICI

#### Steps to run autoTICI
1. Create a conda environment
    `conda env create -n new_env_name -f environment.yml`
2. Install packages
    `pip install -r requirements.txt`
3. Run autoTICI
`python autoTICI "path/to/preEVT/dicom" "path/to/preEVT/dicom" occ_loc[ICA/M1/M2]`

usage: autoTICI.py [-h] [-m] [-l] [-v {ap,lateral}] [-o OUTPUT_DIRPATH] preEVT postEVT {ICA,M1,M2}

autoTICI: automatic brain reperfusion scoring (eTICI) for ischemic stroke

positional arguments:
  preEVT                Input dicom file of preEVT DSA.
  postEVT               Input dicom file of postEVT DSA. Make sure pre and post have the same ap/lateral view.
  {ICA,M1,M2}           Occlusion location: [ICA, M1, M2].

optional arguments:
  -h, --help            show this help message and exit
  -m, --motion_correction_enabled
                        Whether to perform motion correction of each DSA series.
  -l, --landmark_preregistration_enabled
                        Whether to perform landmark-based pre-registration between pre- and post-EVT DSA.
  -v {ap,lateral}, --view {ap,lateral}
                        DSA view: [ap, lateral], only needed for landmark-based detection.
  -o OUTPUT_DIRPATH, --output_dirpath OUTPUT_DIRPATH
                        Output directory path.


Please **cite** the following articles if you find it useful:
1. Su, R., Cornelissen, S. A., Van der Sluijs, M., Van Es, A. C., Van Zwam, W. H., Dippel, D. W., ... & van Walsum, T. (2021). autoTICI: automatic brain tissue reperfusion scoring on 2D DSA images of acute ischemic stroke patients. IEEE transactions on medical imaging, 40(9), 2380-2391.
2. van der Sluijs, P. M., Su, R., Cornelissen, S., van Es, A. C., a Nijeholt, G. J. L., van Doormaal, P. J., ... & van der Lugt, A. (2024). Assessment of automated TICI scoring during endovascular treatment in patients with an ischemic stroke. Journal of NeuroInterventional Surgery. 
