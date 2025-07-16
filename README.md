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
 The data used is from yfinance : downladed and read in the notebook. 
 Else USE:
        ''' yfinance
        self.df = yf.download(self.ticker, start=self.start, end=self.end)
        '''
