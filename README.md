# 🍷 Wine Quality Prediction (Machine Learning Project)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ammarrazaa/Wine-quality-prediction/blob/main/Wine_Quality_prediction.ipynb)

Predict whether a wine is **Low**, **Medium** or **High** quality from its chemical properties, and compare how **training time** and **accuracy** change between two runs of each model.

> Machine Learning course project, Fall 2026
> The University of Lahore | BSCS | Instructor: Ms. Tahseen Fatima

---

## 📌 Project Idea

three models training on a tabular Kaggle dataset. Each model is trained **twice**:

| Run | Setting |
|-----|---------|
| **Run 1** | Normal CPU run (1 core / 1 thread) |
| **Run 2** | Faster setting (all CPU cores, or GPU for XGBoost if available) |

Then we compare **training time**, **accuracy** and **F1 score**.

## 📊 Dataset

- **Name:** Wine Quality (red + white Vinho Verde wines)
- **Source:** [Kaggle: Wine Quality Data Set (Red & White Wine)](https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine)
- **Size:** 6,497 rows, 13 columns (4,898 white and 1,599 red wines)
- **Features:** wine type, fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol
- **Target:** quality score (3 to 9), grouped into 3 classes:
  - **Low** = 3, 4, 5
  - **Medium** = 6
  - **High** = 7, 8, 9

## 🧠 Models Used

| Model | Why we used it |
|-------|----------------|
| **Logistic Regression** | Simple, fast baseline |
| **Random Forest** | Strong on tabular data, trains trees in parallel on many cores |
| **XGBoost** | Accurate gradient boosting model, can run on GPU |

## ⚙️ Preprocessing Steps

1. Checked for missing values (none found)
2. Encoded `type` (white = 0, red = 1)
3. Converted quality score into 3 classes
4. 80/20 train-test split with stratification
5. Scaled features with `StandardScaler` (Logistic Regression only)

## 📈 Results

Run on Google Colab (2 CPU cores, no GPU available in this session).

| Model | Run | Setting | Time (s) | Accuracy (%) | F1 (%) |
|-------|-----|---------|----------|--------------|--------|
| Logistic Regression | Run 1 | CPU (1 core) | 0.425 | 55.85 | 55.32 |
| Logistic Regression | Run 2 | CPU (repeat run) | 0.289 | 55.85 | 55.32 |
| Random Forest | Run 1 | CPU (1 core) | 5.315 | **71.54** | 71.46 |
| Random Forest | Run 2 | CPU (all 2 cores) | 4.226 | **71.54** | 71.46 |
| XGBoost | Run 1 | CPU (1 thread) | 1.784 | 70.85 | 70.85 |
| XGBoost | Run 2 | CPU (all 2 cores) | 1.874 | 70.85 | 70.85 |

**Random Forest F1 by class:** Low 76.4, Medium 70.4, High 64.6

### 🔍 Key Findings

- **Random Forest was the most accurate** (71.54%), with XGBoost close behind (70.85%). Both beat the Logistic Regression baseline (55.85%) by a large margin, because wine quality is not a simple straight-line problem.
- **Accuracy does not change between Run 1 and Run 2.** Same model, same data. Only the hardware setting changes speed.
- **Random Forest got a 1.26x speed-up** using 2 cores instead of 1.
- **XGBoost showed no speed-up** here. With a dataset this small, start-up overhead cancels the gain.
- **Alcohol is the most important feature** for predicting quality. The "High" quality class is the hardest to predict.

## ⚠️ Limitations

- The dataset is small, so GPU/parallel speed-ups are not very visible.
- No hyper-parameter tuning was done.

## 🚀 How to Run

**Option 1: Google Colab (easiest)**
1. Click the **Open in Colab** badge at the top.
2. Upload `wine-quality-white-and-red.csv` to the Colab files panel (or use the file in the `data/` folder).
3. For GPU: `Runtime → Change runtime type → T4 GPU`.
4. Click `Runtime → Run all`.

**Option 2: Locally**
```bash
git clone https://github.com/YOUR-USERNAME/REPO-NAME.git
cd REPO-NAME
pip install -r requirements.txt
jupyter notebook Wine_Quality_prediction.ipynb
```
> Note: the notebook reads the CSV from `/content/wine-quality-white-and-red.csv` (Colab path). Change this path when running locally.

## 📁 Repository Structure

```
├── Wine_Quality_prediction.ipynb        # Main notebook (code + outputs)
├── Wine_Quality_ML_Project_Report.pdf   # Full project report
├── results_table.csv                    # Performance table
├── data/
│   └── wine-quality-white-and-red.csv   # Dataset
├── charts/                              # Saved graphs
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Hyper-parameter tuning (GridSearchCV / RandomizedSearchCV)
- Handle class imbalance (SMOTE or class weights)
- Remove duplicate rows and use cross-validation
- Try more models (SVM, LightGBM, Neural Network)
- Test on a larger dataset to see a clearer GPU benefit


## 🛠️ Built With

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Google Colab

## Support 
Found a mistake in an answer? Want to add anything more Feel free to:

- Fork this repository
- Make your changes
- Submit a Pull Request
- All contributions are welcome!
:)
