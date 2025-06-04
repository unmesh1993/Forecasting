**Forecasting-with-memory**
 prediction of stock prices using different methods:

  1. ML : Ridge regression (regularization), Random Forest, Neural Network
  2. TIME SERIES : SARIMA
  3. STOCHASTIC MODELS : Binomial Tree, Geometric brownain motion,  Merton jump diffusion process

**Data**
 The data used is from yfinance : downlaaded and read in the notebook. 
 Modifying the following section directly fetcheds the data from the yfinance server :

   !! Comment out !!      
         # Read CSV and parse the 'Date' column as datetime
        self.df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/IBN_stock_data.csv', parse_dates=['Date'])
        # Set 'Date' as the index
        self.df.set_index('Date', inplace=True)
        self.df = self.df[self.df.index >= '2016-01-01']
  !! -- !!
  And USE:
        ''' yfinance
        self.df = yf.download(self.ticker, start=self.start, end=self.end)
        '''
