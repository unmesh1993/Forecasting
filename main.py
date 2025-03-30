

#####################################################################
class StockPredictor:
    def __init__(self, ticker, start, end, column ,noise_generator, npast = 10, nfuture = 252, model_type='Ridge'):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.model_type = model_type
        self.df=None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.npast = npast
        self.nfuture = nfuture
        self.noise_generator = noise_generator
        self.column=column

        if model_type == 'Ridge':
            self.model = RidgeModel()
        elif model_type == 'RandomForest':
            self.model = RandomForestModel()
        elif model_type == 'LSTM':
            self.model = LSTMModel(sequence_length=50)
        elif model_type == 'SARIMA':
            self.model = SARIMAModel()
        else:
            raise ValueError("Unsupported model type")

    def run(self):

        self.df = StockData(self.ticker, self.start, self.end, self.column).fetch_data()

        if self.model_type == 'SARIMA':
            self.model.train(self.df.flatten())
            predictions = self.model.predict(y_test, self.noise_generator)
        else:
            X, y = features_ml(self.df, self.noise_generator, self.npast, self.column, param = self.nfuture, split_type='index').features()

            split_index = int(X.shape[0] - self.nfuture)

            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.train(X_train_scaled, y_train)

            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = FutureDataPoint(X_train.iloc[[-1]], y_train_pred[-self.npast:] , self.model, self.scaler, self.npast, self.noise_generator, self.column, self.model_type).predict_future_steps(self.nfuture)

        y_pred=np.concatenate((y_train_pred, y_test_pred.flatten()))

        prediction_analysis(self.df, y_pred, self.npast, self.nfuture).plot_predictions()

noise_generator = NoiseGenerator(noise_type='gaussian', mu=0, sigma=1, s0=0, dt=1)
predictor = StockPredictor('^NSEI', '2000-01-01', date.today(), 'Close', noise_generator, npast = 10, nfuture = 252, model_type='Ridge')
predictor.run()