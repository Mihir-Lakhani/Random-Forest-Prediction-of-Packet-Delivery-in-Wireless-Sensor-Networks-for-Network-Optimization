# WSN Packet Delivery Prediction

This project predicts latency categories in Wireless Sensor Networks (WSN) using a Random Forest classifier. It includes data preprocessing, model training, evaluation, and visualization steps.

## Project Structure
- `data/`: Dataset CSV file
- `notebooks/`: Exploratory Data Analysis (EDA)
- `src/`: Source code (preprocessing, modeling, evaluation, utils, visualization)
- `reports/`: Final report (to be created)
- `requirements.txt`: Python dependencies
- `main.py`: Pipeline orchestration

## Setup
1. Install dependencies:
   ```
pip install -r requirements.txt
   ```
2. Place your dataset in `data/wsn_dataset.csv`.
3. Run the pipeline:
   ```
python main.py
   ```

## Usage
- Modify and run `notebooks/exploratory_data_analysis.ipynb` for EDA.
- Source code is in `src/`.
- Outputs and plots are saved in `reports/`.

## Requirements
- Python 3.8+
- See `requirements.txt` for packages.

## Notes
- Replace placeholder files with your actual data and report.
- See code comments and docstrings for details.