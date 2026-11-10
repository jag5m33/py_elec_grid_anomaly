from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


class PCAAnomalyDetector:
    '''
    Unsupervised anomaly detection by reconstruction error.

    Compress each row to a few components, rebuild it, measure what was lost:

        x (31 cols) --transform--> z (k cols) --inverse--> x_hat (31 cols)

    A normal row follows the shape PCA learned, so it rebuilds well.
    An odd row sits off that shape, so the rebuild misses. The gap is the score.

    HIGHER error = more anomalous (the opposite of Isolation Forest).

    Scaling is not optional: PCA maximises variance, so without it the
    components are just whichever columns have the biggest numbers.
    '''

    def __init__(self, x_train, x_val, x_test, n_components=2):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test

        self.scaler = StandardScaler()
        self.model = PCA(n_components=n_components, random_state=42)

    def train(self):
        print("Training PCA model...")
        x_scaled = self.scaler.fit_transform(self.x_train)   # fit on train only
        self.model.fit(x_scaled)

        ratios = self.model.explained_variance_ratio_
        print(f"Explained variance per component: {np.round(ratios, 4)}")
        print(f"Cumulative: {ratios.sum():.2%}")

    def scores(self, x):
        # scale -> compress -> rebuild -> mean squared gap per row
        x_scaled = self.scaler.transform(x)                  # transform only, not fit
        x_rebuilt = self.model.inverse_transform(self.model.transform(x_scaled))
        return np.mean((x_scaled - x_rebuilt) ** 2, axis=1)

    def evaluate(self, percentile=99.0):
        # Cutoff comes from TRAIN so all three splits are judged the same way.
        # A percentile, not a raw error value - the raw number depends on
        # scaling and n_components, so it is impossible to choose sensibly.
        train = self.scores(self.x_train)
        val = self.scores(self.x_val)
        test = self.scores(self.x_test)

        cutoff = np.percentile(train, percentile)

        for name, errors in [('train', train), ('val', val), ('test', test)]:
            flagged = errors > cutoff
            print(f"{name:<6} {len(errors):>8,} rows | {flagged.sum():>6,} flagged "
                  f"({flagged.mean():.2%}) | median {np.median(errors):.4f}")

        return {'train': train, 'val': val, 'test': test, 'cutoff': cutoff}