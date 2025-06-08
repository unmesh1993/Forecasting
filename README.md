**Forecasting-stocks : main_complete_notebook.ipynb**
 prediction of stock prices (from yfinance) using different methods:

  1. ML : Ridge regression (regularization), Random Forest, Neural Network
  2. TIME SERIES : SARIMA
  3. STOCHASTIC MODELS : Binomial Tree, Geometric brownain motion,  Merton jump diffusion process

**Parameters**
 1. For stochastic modelling, initial parameters (mu, sigma etc) are abstracted from fitted log return of historical stock prices depending on their probability distribution. 
    For eg. GBM/Btree model prices have lognormal distribution and Merton jump diffusion model have a poisson weighted lognormal distribution. 
 2. Using Markov Chain Monte Carlo, the bayesian posterior of these prior parameter distribution are samples (For merton : still under process).  

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
