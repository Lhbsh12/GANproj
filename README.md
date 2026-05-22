# Wine Fraud GAN Project

## Project Title
Exploring GAN Variants for Balancing Imbalanced Datasets: Wine Fraud Detection

## Dataset
Hugging Face dataset: `gusdelact/wine_fraud`

## What this project does
This project uses a tabular wine fraud dataset to study class imbalance. It trains two GAN models only on the minority class:

1. Vanilla GAN
2. Wasserstein GAN with Gradient Penalty (WGAN-GP)

The synthetic minority samples are then used to balance the training dataset. A consistent MLP classifier is trained and evaluated under three scenarios:

1. Original imbalanced dataset
2. Dataset balanced using Vanilla GAN
3. Dataset balanced using WGAN-GP

## Metrics
The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrix

## How to run in Google Colab
1. Upload `Wine_Fraud_GAN_Project_Colab.ipynb` to Google Colab.
2. Runtime > Change runtime type > GPU.
3. Runtime > Run all.
4. At the end, download `wine_fraud_gan_final_outputs.zip` from the Colab file browser.

## Files generated after running
- `wine_fraud_gan_outputs/report/Wine_Fraud_GAN_Project_Report.pdf`
- `wine_fraud_gan_outputs/data/metrics_summary.csv`
- `wine_fraud_gan_outputs/figures/*.png`
- `wine_fraud_gan_outputs/models/*.keras`
- `wine_fraud_gan_final_outputs.zip`

## Notes
The test set is never augmented. Synthetic data is used only in the training set, which makes the evaluation fair.
