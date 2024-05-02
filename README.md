# EUKLID AI Systematic Trading Challenge

**Team Members:** Andrea Marcoccia, Leonardo Pulicati, Francesco Del Treste

## [Section 1] Introduction

The project aims to design and develop an AI-driven systematic trading model that can navigate the intricacies of financial markets with precision and agility. By harnessing the power of this dataset, the model will identify patterns, predict market movements, and execute trades to capitalize on these insights, all while managing risk and maximizing returns.

The datasets include 3 indices and 3 stocks: S&P500, Nasdaq, CAC, Microsoft, IBM, and Amazon, with the following data:

- **Date/Time Stamp:** Each record is associated with a specific week, allowing for temporal analysis of financial metrics over time.
- **OHLC Prices:** Provides the opening price, highest price, lowest price, and closing price of a stock or index for the week, crucial for understanding market trends and volatility.
- **Volume:** Indicates the total number of shares or contracts traded for the stock or index during the week, reflecting the level of activity and liquidity.

## [Section 2] Methods

### Pre-Processing

#### Missing Values

Indices had missing records for 2 weeks, which we handled through imputation:

- **Open price** is typically equal to or similar to the close price of the previous week:
  - We imputed the **close price of 2023-05-21** with the open price of 2023-05-28, and the **open price of 2023-05-14** with the close price of 2023-05-07 with high confidence.
  - The **2023-05-14 close price and 2023-05-21 open price** were reasonably imputed as the mean between the 2023-05-14 open and 2023-05-21 close.

- **High and Low prices** were imputed as follows:
  - The **greater between the open and close prices** was taken as the high price.
  - The **lower between the open and close prices** was taken as the low price.

- **Volume** was imputed as the **mean volume of the year**, based on the distribution of volume over the last 5 years.

The second issue addressed in pre-processing is **stock splits**. While IBM and Microsoft data were already adjusted for stock splits, Amazon's data did not account for a recent split, which was corrected.

[Two pictures can be placed here side by side.]

#### Additional Columns for Model Training and Evaluation

We computed two columns useful for the training and evaluation of our models:

- **Market direction:** Indicates if the price went up or down compared to the previous week. This helps evaluate the success of our trading choices.

- **Trading strategy:** Either -1 (short), 0 (exit), or 1 (long). This represents the trading choice we try to predict with our models. A moderate approach is used for trade signals, with a percentage change threshold for going long or short to exit the market approximately 30% of the time. This way, when the percentage change is minimal compared to the previous week, no trade signal is generated. This signal serves as the output value to predict with our models.
  
#### Indicators

Several indicators were computed for use in the models as predictors:

- **SMA and EMA (14 Weeks):** Targets medium-term trends, reflecting quarterly performance, crucial for understanding market dynamics over significant financial periods.
- **Stochastic Oscillator (Default: 14):** Spots overbought or oversold conditions, important for predicting potential price reversals on a medium-term basis.
- **RSI (14 Weeks):** Evaluates medium-term market momentum, useful in identifying overbought or oversold conditions over a quarter.
- **MACD (14 weeks):** Detects changes in medium-term trend strength and direction, offering signals for potential trading opportunities.
- **Hurst Coefficient:** Indicates the behavior of time series:
  - H < 0.5: Tends to revert to a mean, suggesting that increases will likely be followed by decreases and vice versa.
  - H = 0.5: Future price movements are completely independent of past movements.

Finally, we standardized the data and used a 90-10 split for training and testing.
