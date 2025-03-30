

######################################################################

class FutureDataPoint:
    def __init__(self, X_train_last, y_train_pred_npast , model, scaler, npast, noise_generator, column, model_type='LSTM'):
        self.X_new = X_train_last.copy()
        self.y_train_pred_npast = y_train_pred_npast.copy()
        self.model_type=model_type
        self.model = model
        self.scaler = scaler
        self.npast = npast
        self.noise_generator = noise_generator
        self.column = column

    def predict_future_steps(self, nfuture):
        y_test_pred = []
        for i in range(nfuture):

            next_index = self.X_new.index[0] + pd.tseries.offsets.BDay(1) # Set the next business day as the index
            self.X_new.index = [next_index]

            y_next = self.generate_datapoint_predict()
            y_test_pred.append(y_next)

        return np.array(y_test_pred)

    def generate_datapoint_predict(self):
        self.X_new["date"] = self.X_new.index
        self.X_new["year"] = self.X_new["date"].dt.year
        self.X_new["month"] = self.X_new["date"].dt.month
        #self.X_new["day"] = self.X_new["date"].dt.day
        #self.X_new["day_of_year"] = self.X_new["date"].dt.dayofyear
        self.X_new.drop(columns=["date"], inplace=True)

        # Add noise to the dataset
        self.X_new['Noise'] = self.noise_generator.generate(1)

        #add lags from predictions
        for i in range(1, self.npast+1):
            self.X_new[f'lag{i}'] = self.y_train_pred_npast[-i]
        self.X_new.dropna(inplace=True)

        # Scale the new data point
        X_new_scaled = self.scaler.transform(self.X_new)

        if self.model_type == 'LSTM':
            X_new_scaled = np.reshape(X_new_scaled, (X_new_scaled.shape[0], X_new_scaled.shape[1], 1))

        # Predict next value
        y_next = self.model.predict(X_new_scaled)

        for i in range(0,self.npast-1):
          self.y_train_pred_npast[i+1] = self.y_train_pred_npast[i]
        self.y_train_pred_npast[0] = y_next.flatten()[0]

        return y_next