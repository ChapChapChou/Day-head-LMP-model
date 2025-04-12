# Day-ahead LMP Prediction Model for US Energy Market

This project aims to predict day-ahead Locational Marginal Prices (LMP) for the US energy market, specifically focusing on data from PJM (https://www.pjm.com/home.aspx).  
It leverages the PJM public API, facilitated by the `pjm_dataminer` library (https://github.com/rzwink/pjm_dataminer).

## Project Structure
```
Day-head-LMP-model/
├── src/
│   ├── get_lmp.py                 # download data from PJM
│   ├── data_process.ipynb         # use model to predict
|   ├── hybrid_predication.py      # predict low-freq and high-freq and combine
|   └── use_hybrid_predication.py  # entry programm 
└── dataset/
├── hourly_metered_load.csv    # load series
└── pjm_data_COMED.csv         # pjm lmp date (where 'zone'== 'COMED')
```

## `src/get_lmp.py`

This script retrieves LMP data from the PJM API.

### Usage Examples

1.  **Default Parameters:**

    ```bash
    python get_lmp.py
    ```

    This will use default parameters to fetch data.

2.  **Specify Parameters:**

    ```bash
    python get_lmp.py --row_count=10000 --zone=COMED --start_date="01-09-2024 00:00" --end_date="01-09-2024 23:59"
    ```

    This example specifies the number of rows, the zone, and the start and end dates.

3.  **JSON Output:**

    ```bash
    python get_lmp.py --format=json
    ```

    This will output the data in JSON format.

**Note:** The date format is "DD-MM-YYYY HH:MM".

## `src/data_process.ipynb`

This Jupyter Notebook processes the retrieved data. The workflow includes:

1.  **Aggregation:** Aggregating regional load and nodal LMP data, using date as the index.
2.  **Time Series Extraction:** Obtaining time series of load and LMP for different nodes.
3.  **Wavelet Transformation:** Separating high-frequency and low-frequency components using wavelet transformation.
4.  **Model Analysis:** Analyzing the separated components using different models.
5.  **Price Prediction Synthesis:** Synthesizing the predictions to obtain the final predicted LMP.

## `dataset/`

This directory contains the datasets used in the project:

-   `hourly_metered_load.csv`: Hourly metered load data.
-   `pjm_data_COMED.csv`: LMP data for the COMED zone.

## Model Pick  

Generally we get multiple choices for low-freq series and high-freq series:

### Low-freq 

1. PSO-LVSVM (picked)
2. LSTM
3. Prophet
4. GBDT
   
### High-freq

1. ARIMA (picked)
2. GARCH

## Dependencies

* python 3.13+
* pandas
* numpy
* requests
* wavelets libraries
* scikit-learn
* matplotlib
* jupyter notebook

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd Day-head-LMP-model
    ```

2.  Install the required packages:

    ```bash
    pip install pandas numpy requests scikit-learn matplotlib jupyterlab pywavelets
    ```

3.  Run the python scripts, or open and run the Jupyter notebook.

## Further Development

* Adding more models to improve prediction accuracy.
* Expanding the dataset to include more zones and time periods.
* Implement real time data processing and display.
* Adding visualization to the prediction results.
