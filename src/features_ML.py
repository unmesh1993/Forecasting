
#######################################
class features_ml:

    def __init__(self, df, noise_generator, npast, column, param = 252, split_type='index'):
        self.df = df.copy()
        self.noise_generator = noise_generator
        self.npast = npast
        self.split_type = split_type
        self.param = param
        self.column=column
        self.X=None
        self.Y=None

    def features(self):
        self.df["date"] = self.df.index
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        #self.df["day"] = self.df["date"].dt.day
        #self.df['day_of_year'] = self.df["date"].dt.dayofyear
        self.df.drop(columns=["date"], inplace=True)

        self.df['Noise'] = self.noise_generator.generate(self.df.shape[0])

        for lag in range(1, self.npast + 1):
            self.df[f'lag{lag}'] = self.df[self.column].shift(lag)

        self.df.dropna(inplace=True)

        self.y = self.df[self.column].copy()
        self.X = self.df.drop(columns=self.column).copy()
        self.X.info()
        return self.X, self.y