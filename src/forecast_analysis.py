

###############################################

class prediction_analysis:

    def __init__(self, df, y_pred, npast, nfuture):
        self.df = df
        self.y_pred = y_pred
        self.npast = npast
        self.nfuture = nfuture


    def plot_predictions(self):
        df_plot = self.df.iloc[self.npast:].copy()
        df_plot['Predicted_Close'] = self.y_pred

        # Calculate percentage error
        df_plot['Prediction_Change_%'] = 100 * (df_plot['Close'] - df_plot['Predicted_Close']) / df_plot['Close']
        df_plot.dropna(inplace=True)

        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.5)

        # --- Subplot 1: Time Series ---
        axs[0].plot(df_plot.index, df_plot['Close'], label='Actual Close', color='blue')
        axs[0].plot(df_plot.index, df_plot['Predicted_Close'], label='Predicted Close', linestyle='--', color='orange')

        split_time = self.df.iloc[-self.nfuture].name
        axs[0].axvline(x=split_time, color='black', linestyle=':', label='Train/Test Split')

        axs[0].set_title('Actual vs Predicted Close Prices')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_xlim(self.df.index[-2 * self.nfuture], self.df.index[-1] )

        # --- Subplot 2: % Error vs Actual Price ---
        axs[1].scatter(df_plot.index, df_plot['Prediction_Change_%'], alpha=0.6, color='green')
        axs[1].axhline(y=0, color='red', linestyle='--', label='No Error')
        split_time = self.df.iloc[-self.nfuture].name
        axs[1].axvline(x=split_time, color='black', linestyle=':', label='Train/Test Split')

        axs[1].set_title('Percentage Prediction Error vs Actual Price')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Prediction_Change_%')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylim(-20, 20)

    #subplot 3
        test_preds = df_plot.iloc[-self.nfuture:]['Prediction_Change_%'].values

        # Fit a Student's t-distribution to the data
        df_t, loc_t, scale_t = t.fit(test_preds)

        # Compute statistics from t-distribution
        mean_pred = loc_t
        std_pred = scale_t * np.sqrt(df_t / (df_t - 2)) if df_t > 2 else np.inf
        ci_lower, ci_upper = t.interval(0.95, df_t, loc=loc_t, scale=scale_t)

        # Plot histogram with KDE
        sns.histplot(test_preds, kde=True, bins=50, color='skyblue', edgecolor='black', ax=axs[2])

        # Plot mean
        axs[2].axvline(mean_pred, color='red', linestyle='-', label=f'Mean (t-fit): {mean_pred:.2f}')

        # Plot ±1σ range
        axs[2].axvline(mean_pred - std_pred, color='purple', linestyle='--', label=f'-1σ: {mean_pred - std_pred:.2f}')
        axs[2].axvline(mean_pred + std_pred, color='purple', linestyle='--', label=f'+1σ: {mean_pred + std_pred:.2f}')

        # Plot 95% confidence interval (CI)
        axs[2].axvspan(ci_lower, ci_upper, color='gray', alpha=0.3, label=f'95% CI (t-fit): {ci_lower:.2f} - {ci_upper:.2f}')

        # Plot formatting
        axs[2].set_title('Histogram of Prediction_Change_% (Test Set) with T-Distribution Fit')
        axs[2].set_xlabel('Prediction_Change_%')
        axs[2].set_ylabel('Frequency')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_xlim(mean_pred - 4 * std_pred, mean_pred + 4 * std_pred)

        # --- Subplot 4: Bimodal Distribution with Gaussian Mixture Model ---
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(test_preds.reshape(-1, 1))
        labels = gmm.predict(test_preds.reshape(-1, 1))

        # Extract means and variances of both components
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()

        sns.histplot(test_preds, bins=30, kde=True, color="gray", alpha=0.5, ax=axs[3])
        axs[3].scatter(test_preds, [-0.01]*len(test_preds), c=labels, cmap="coolwarm", s=5)
        axs[3].set_title(f"GMM Fit: Mean1={means[0]:.2f}, Var1={variances[0]:.2f} | Mean2={means[1]:.2f}, Var2={variances[1]:.2f}")


        prediction = self.y_pred[-1]
        mean=df_plot['Prediction_Change_%'].mean()
        std=df_plot['Prediction_Change_%'].std()
        ci_lower, ci_upper = t.interval(0.95, df_t, loc=loc_t, scale=scale_t)

        # Corrected prediction
        corrected = prediction * (1 + mean / 100)

        # 95% CI bounds
        ci_low = prediction * (1 + ci_lower / 100)
        ci_high = prediction * (1 + ci_upper / 100)

        # Print results
        print("Actual + 1 year:" , self.df.iloc[-1, 0])
        print(f"Prediction + 1 year: {prediction:.2f}")
        print(f"Corrected Prediction: {corrected:.2f}")
        print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")
        print(f"±1 Standard Deviation Range: [{prediction * (1 + (mean - std)/100):.2f}, {prediction * (1 + (mean + std)/100):.2f}]")

