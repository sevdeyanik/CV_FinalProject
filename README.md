# CV_FinalProject
Computer Vision Final Project 2025 Group 9

# CycleGAN for Monet Style Transfer

This project applies CycleGAN for unpaired image-to-image translation between real-world photographs and Monet-style paintings. It explores different preprocessing strategies and architectural configurations to improve translation quality, efficiency, and training stability.

---

## 📁 Project Structure

```
├── checkpoints/                    # Saved model weights
├── datasets/                      # Monet2Photo dataset
│   └── monet2photo/
├── generated_images/             # Generated outputs for each model
│   ├── Model_1_Photo_To_Monet/
│   └── ...
├── models/                       # Model architectures and dataset classes
│   ├── Model.py
│   ├── Model6.py
│   ├── MonetPhotoDataset.py
│   └── MonetPhotoDatasetGridcropping.py
├── trains/                       # Training scripts
│   ├── train.py
│   ├── train_model1.py
│   └── train_model2.py
├── experiments/                  # Evaluation and inference scripts
│   ├── TestSimple.py
│   ├── TestAllModelsMonetToPhoto.py
│   ├── TestAllModelsPhotoToMonet.py
│   └── TestAllModelsFIDAndSSIM.py
└── report/                       # Final report
```

---
---

## 🔍 Objective

- Translate real photos ↔ Monet paintings using CycleGAN
- Experiment with:
  - Different image resolutions
  - Random and grid cropping
  - Model simplification (discriminator removal)
  - Downsampling + random cropping
- Evaluate results using **FID** and **SSIM** metrics

---

## 🛠️ Setup

```bash
pip install torch torchvision clean-fid scikit-image matplotlib
```
---

## 🖼️ Visual Examples

See `report/` for side-by-side comparison figures:
- Monet → Photo
- Photo → Monet

---

## 📚 References

- CycleGAN: [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)
- Johnson et al.: [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)

---

## ✍️ Authors

- **Sevde Yanik** — Data & Computer Science
- **Sarp Tan Gecim** — Physics

University of Heidelberg — Computer Vision Project (Group 09)

