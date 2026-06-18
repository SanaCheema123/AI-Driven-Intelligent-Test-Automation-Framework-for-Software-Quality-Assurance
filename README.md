# AI-Driven Intelligent Test Automation Framework for Software Quality Assurance
[screen-capture (8).webm](https://github.com/user-attachments/assets/a7ccdee0-e7ce-4f86-a868-a7787ea6742e)

This project predicts defective software modules using static code metrics and converts the prediction into a QA testing priority.

## Dataset
NASA JM1 Software Defect Dataset

## Target
defects: True / False

## Main Features
- Code complexity metrics
- Halstead software metrics
- Lines of code metrics
- Operator and operand metrics
- Branch count

## Models
- Logistic Regression
- Random Forest
- Gradient Boosting

## Run Commands

```powershell
pip install -r requirements.txt
python src\train_model.py
streamlit run app\dashboard.py
```

## Dashboard Pages
- QA Overview
- Defect Predictor
- Quality Analytics
- Test Prioritization
- Monitoring
- Project Guide
