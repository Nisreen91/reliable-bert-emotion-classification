# Reliability-Aware BERT-Based Emotion Classification

This repository contains the implementation for the study:

**Reliability-Aware BERT-Based Emotion Classification via Probability Calibration and Selective Prediction**

## Overview

This study investigates reliability-aware emotion classification using a fine-tuned BERT model. Beyond conventional predictive performance, the framework evaluates probability calibration, predictive uncertainty, and selective prediction.

The pipeline consists of:

- BERT-based emotion classification
- Temperature scaling for probability calibration
- Uncertainty estimation
- Selective prediction using uncertainty-based rejection
- Risk–coverage analysis

## Dataset

The experiments use a six-class emotion classification dataset containing the following emotion categories:

| Label | Emotion |
|------:|---------|
| 0 | 😢 Sadness |
| 1 | 😆 Joy |
| 2 | 😍 Love |
| 3 | 😡 Anger |
| 4 | 😰 Fear |
| 5 | 😮 Surprise |

The dataset is divided into stratified training, validation, and test sets.

## Model

The classification model is based on **BERT (bert-base-uncased)**.

The main architecture consists of:

`Input Text → Tokenization → BERT Encoder → [CLS] Representation → Linear Classification Head → Softmax Probabilities`

## Probability Calibration

Post-hoc **temperature scaling** is applied using the validation set to improve the reliability of the predicted probabilities.

Calibration performance is evaluated using **Expected Calibration Error (ECE)**.

## Uncertainty Estimation

Predictive uncertainty is quantified from the calibrated probability distribution using:

- Maximum softmax confidence
- Predictive entropy
- Probability margin
- Variation ratio

These measures are used to distinguish confident predictions from uncertain cases.

## Selective Prediction

The framework implements uncertainty-based selective prediction.

Instead of forcing the classifier to make a prediction for every sample, uncertain predictions can be rejected or deferred.

Performance is evaluated using:

- Coverage
- Selective accuracy
- Selective risk
- Risk–coverage analysis

Uncertainty thresholds are selected using the validation set and subsequently evaluated on the held-out test set.

## Code

The main implementation is provided in:

`reliability_aware_emotion_classification.py`

The script contains the complete experimental pipeline, including model training, calibration, uncertainty estimation, and selective prediction.

## Requirements

The implementation uses Python and common machine-learning/NLP libraries, including:

- PyTorch
- Transformers
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Reproducibility

The repository is provided to support reproducibility of the experiments and reliability analyses reported in the paper.

## Citation

If you use this code in your research, please cite the associated paper.

Citation information will be added after publication.

## License

This project is released under the MIT License.
