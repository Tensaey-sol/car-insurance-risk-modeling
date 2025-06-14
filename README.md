# 🚗 Car Insurance Risk Modeling

This repository contains the setup and development for **Car Insurance Risk Modeling**, a real-world insurance analytics challenge focused on understanding risk segmentation and optimizing premiums for car insurance policies in South Africa.

The project simulates the role of a **Marketing Analytics Engineer**, leveraging exploratory data analysis (EDA), statistical hypothesis testing, and machine learning models to deliver insights that inform pricing and targeting strategies.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Tensaey-sol/car-insurance-risk-modeling.git
cd car-insurance-risk-modeling
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
car-insurance-risk-modeling/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow
├── .gitignore                 # Ignore rules for Git
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── notebooks/                 # Jupyter Notebooks
│   └── README.md
├── tests/                     # Unit tests for scripts and functions
│   └── __init__.py
├── scripts/                   # Scripts for EDA, testing, modeling
│   ├── __init__.py
│   └── README.md
├── data/                      # Raw and processed datasets (DVC tracked)
└── .venv/                     # Local virtual environment (ignored via .gitignore)
```

---

## ⚙️ GitHub Actions CI

This repository uses **GitHub Actions** for continuous integration. On every push:

- Python 3.13.1 is installed
- Dependencies are installed
- Tests are executed from `/tests`

See `.github/workflows/ci.yml` for workflow configuration.

---

## 🛠 Tools & Technologies

- Python **3.13.1**
- Pandas, NumPy, Scikit-learn, XGBoost, SHAP
- Matplotlib, Seaborn, Plotly (for visualization)
- DVC (Data Version Control)
- Git & GitHub for version control
- GitHub Actions for CI/CD
- Jupyter Notebooks for analysis

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Questions or Contributions?

Feel free to open an issue or submit a pull request to contribute or raise a question.
