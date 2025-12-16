# Data

See the [starting_kit notebook](https://github.com/frcaud/xray_detector/blob/main/starting_kit.ipynb) for data exploration, baseline models, and a domain adaptation example.

## From S-Curves to Tabular Data

The raw data for this challenge originates from threshold scans performed on hybrid pixel detectors. As described in the introduction, each threshold scan produces an S-curve for every pixel in the detector. These S-curves represent the cumulative distribution of photon counts as a function of the applied threshold voltage (here in DAC units).

To transform this raw detector data into a format suitable for machine learning, we extract the S-curve measurements for each pixel. Specifically, for each pixel, we take the photon counts at 251 distinct threshold values, creating a fixed-length feature vector. This transformation converts the detector calibration problem into a tabular machine learning task where:

- **Each row** represents a single pixel's S-curve measurement
- **Each column** corresponds to one of the 251 applied threshold points
- **The target variable** is the inflection point of the S-curve, which corresponds to the optimal threshold for that pixel at the given beam energy

This representation allows machine learning models to learn patterns in the S-curve shapes and predict the inflection point directly from the raw threshold scan data, without requiring explicit curve fitting procedures.

## Data Splits

The competition dataset is divided into several subsets to support both standard supervised learning and domain adaptation scenarios:

**Phase 1**

- **Training set (`train.csv`, `train_labels.csv`)**: This is the **source domain** training data, consisting of S-curve measurements (251 features) and their corresponding inflection point labels. This data comes from a specific experimental configuration at **10keV** beam energy.

- **Domain Adaptation set (`train_DA.csv`)**: This dataset contains S-curve measurements from a different domain at **12keV** beam energy, but **without labels** (the **target domain**). The challenge is to leverage this unlabeled data to improve model generalization to target domain.

- **Test set (`test.csv`)**: The test set contains S-curve measurements from the target domain (12keV) configuration. Participants must predict the inflection points for these examples, and the predictions will be evaluated against held-out ground truth labels.

**Phase 2**

In the final phase of the competition, source domain will be data at **12keV** and target domain will be data at **18keV**. The names of the datasets are the same that those of phase 1 to be able to use the same exact submission script. Identical names but different data !

The domain shift between these datasets reflects real-world scenarios where detectors are calibrated under different experimental conditions, beam energies, or detector states. A model that can effectively adapt to these domain changes will be more robust and practical for deployment in synchrotron facilities.

## The Domain Adaptation Task

This competition presents a **domain adaptation** challenge. The core problem is that the training data and test data come from different distributions (different domains). While we have labeled training data from one domain, we also have unlabeled data from an intermediate domain (`train_DA.csv`), and we need to make accurate predictions on a test domain.

The domain adaptation task requires participants to:
1. Learn from the labeled training data
2. Leverage the unlabeled domain adaptation data (`train_DA.csv`) to adapt the model to new domain characteristics
3. Generalize well to the test domain


## Baseline Model: Random Forest with CORAL Domain Adaptation

As a starting point, we provide a baseline model that demonstrates how to utilize the domain adaptation data (`X_adapt`) in your submission. This baseline combines a **Random Forest Regressor** with **CORAL (Correlation Alignment)** domain adaptation from the [Skada library](https://scikit-adaptation.github.io).

The baseline model serves as a reference implementation that:

- Takes the 251-dimensional S-curve features as input
- Predicts the inflection point (threshold value) as a continuous regression target
- **Performs domain adaptation** using the CORAL method to align the feature distributions between the source domain (training data) and target domain (domain adaptation and test data)

The baseline model is implemented as follows:

- **Base Regressor**: Random Forest Regressor
- **Domain Adaptation Method**: CORAL (Correlation Alignment) adapter from Skada
- **Pipeline**: Uses `make_da_pipeline()` to combine the CORAL adapter with the Random Forest regressor
- **Training Procedure**:
  1. Concatenates the source domain data `X_train` with the target domain data (`X_adapt`), and concatenate source labels `y_train` with a vector of length X_adapt.shape[0] full of NaN labels.
  2. Creates domain labels to distinguish source (0) and target (-1) samples
  3. Fits the domain adaptation pipeline, which aligns the feature distributions before training the regressor
- **Prediction**: Makes predictions on `X_test` after domain adaptation

This baseline demonstrates a practical approach to domain adaptation that participants can build upon.

Participants are encouraged to explore the full range of domain adaptation methods available in Skada or implement custom domain adaptation strategies that can better leverage the unlabeled `X_adapt` data to improve generalization to the test domain.