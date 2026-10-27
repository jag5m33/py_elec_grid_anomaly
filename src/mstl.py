from statsmodels.tsa.seasonal import MSTL
import matplotlib.pyplot as plt


class SeasonalTrendDecomp:
    # series  : ND values as a pandas Series, with the GMT timestamp as the index
    # periods : cycle lengths in rows: 48 = daily, 336 = weekly
    def __init__(self, series, periods=(48, 336)):
        self.series = series
        self.periods = periods

    def mstl_model_train(self):
        
        # robust=True stops big anomalies (Covid, Christmas) bending the "normal" patterns
        self.mstl = MSTL(self.series, periods=self.periods, stl_kwargs={"robust": True})
        result = self.mstl.fit()
        seasonal, trend, resid = result.seasonal, result.trend, result.resid
        return seasonal, trend, resid

    def plot(self, seasonal, trend, resid):
        plt.figure(figsize=(16, 12))

        plt.subplot(5, 1, 1)
        plt.plot(self.series)
        plt.title('original series', fontsize=16)

        plt.subplot(5, 1, 2)
        plt.plot(trend)
        plt.title('trend', fontsize=16)

        # seasonal has two columns: 'seasonal_48' (daily) and 'seasonal_336' (weekly)
        sea_col1, sea_col2 = seasonal.columns

        plt.subplot(5, 1, 3)
        plt.plot(seasonal[sea_col1])
        plt.title('daily seasonal (48)', fontsize=16)

        plt.subplot(5, 1, 4)
        plt.plot(seasonal[sea_col2])
        plt.title('weekly seasonal (336)', fontsize=16)

        plt.subplot(5, 1, 5)
        plt.plot(resid)
        plt.title('residual', fontsize=16)

        plt.tight_layout()
        plt.show()