# CV_FinalProject
Computer Vision Final Project 2025 Group 9 - Sevde Yanik, Sarp Tan Gecim

# CycleGAN for Monet Style Transfer

This project applies CycleGAN for unpaired image-to-image translation between real-world photographs and Monet-style paintings. It explores different preprocessing strategies and architectural configurations to improve translation quality, efficiency, and training stability.

---

## 📁 Project Structure

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
