# Using the 3D dice score module from https://github.com/ancestor-mithril/dice-score-3d

# To run the 3D metrics, the volumes must be stored in `volumes/{model}/ce`
from dice_score_3d import dice_metrics
import numpy as np

ROOT = None
assert ROOT is not None, "Replace the root path with the path to the project folder"

models = ["baseline_enet", "deeplabv3plus", "enet", "unet", "unetpp"]

model_result = []
for model in models:
    total = np.zeros(5)
    best = 0
    worst = 1
    for patient in [1, 2, 13, 16, 21, 22, 28, 30, 35, 39]:
        results = dice_metrics(f"{ROOT}data/segthor_train/train/Patient_{patient:02d}/GT.nii.gz", f"{ROOT}volumes/finalv2/{model}/Patient_{patient:02d}.nii.gz", output_path=None, indices={"Esophagus": 63, "Heart": 126, "Trachea": 189, "Aorta": 252})["Mean"]
        total += [results["Esophagus"], results["Heart"], results["Trachea"], results["Aorta"], results["Mean"]]
        if results["Mean"] > best:
            best = results["Mean"]
            best_patient = patient
        if results["Mean"] < worst:
            worst = results["Mean"]
            worst_patient = patient
    model_result.append([total/10, best_patient, worst_patient])

for model, results in zip(models, model_result):
    print(model, results)