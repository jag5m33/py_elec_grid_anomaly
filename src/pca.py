from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class PCAAnomalyDetector:
    '''
    UNSUPERVISED anomaly detection by reconstruction error.

    THE IDEA
        PCA finds the few directions that explain most of the variance in the
        data - the "shape" that normal rows follow. Compressing a row down to
        those directions and rebuilding it is a round trip:

            x  --transform-->  z  --inverse_transform-->  x_hat
          (31 cols)         (k cols)                    (31 cols)

        A normal row sits close to that shape, so it rebuilds almost
        perfectly and x_hat is near x. An unusual row sits OFF the shape, so
        the compression throws away the very thing that made it unusual and
        the rebuild misses. The gap between x and x_hat is the anomaly score.

    DIRECTION - OPPOSITE TO ISOLATION FOREST
        Isolation Forest:  LOWER score  = more anomalous
        PCA reconstruction: HIGHER error = more anomalous
        So when passing these scores to UnsupervisedAnomalyEval, set
        higher_is_anomalous=True.

    WHY SCALING IS NOT OPTIONAL
        PCA maximises VARIANCE. A column measured in thousands has far more
        raw variance than one measured 0-1, so without scaling the components
        are simply whichever columns have the biggest numbers - nothing to do
        with which carry information. StandardScaler puts every column on
        mean 0, std 1 first, so they compete fairly.
    '''

    def __init__(self, x_train, x_val, x_test, n_components=2):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.n_components = n_components

        # Fitted on train only, in train(). Same leakage rule as supervised
        # work: the mean and std are numbers LEARNED from the data, so they
        # must come from train and be applied unchanged to val and test.
        self.scaler = StandardScaler()
        self.model = PCA(n_components=self.n_components, random_state=42)

    def train(self):
        print("Training PCA model...")

        x_scaled = self.scaler.fit_transform(self.x_train)
        self.model.fit(x_scaled)

        # explained_variance_ratio_: share of total variance each component captures. 
        # The cumulative sum tells you whether n_components is suitable 
            # too few and you discard real structure, too many and you reconstruct the anomalies perfectly 
            # too, leaving no error to detect and a model that overfits.

        ratios = self.model.explained_variance_ratio_
        print(f"Components: {self.n_components}")
        print(f"Explained variance per component: {np.round(ratios, 4)}")
        print(f"Cumulative variance explained: {ratios.sum():.2%}")

    def scores(self, x):
        '''
        The whole round trip in one call: scale, compress, rebuild, measure
        the gap. Returns one error per row.

        This is the method you actually use. transform() and
        inverse_transform() below are kept separate so you can inspect the
        intermediate steps while learning.
        '''
        x_scaled = self.scaler.transform(x)   # transform only -  fitted on train
        x_compressed = self.model.transform(x_scaled)
        x_rebuilt = self.model.inverse_transform(x_compressed)

# WHY DO MEAN SQUARED ERROR AND NOT ABSOLUTE ERROR?
#Mean absolute error would also work, but squaring makes the single large miss stand out more, which is the whole point here.

        # Mean squared error per ROW (axis=1), comparing the SCALED original
        # against its rebuild. Both are in scaled space with the same number
        # of columns, which is what makes the subtraction valid.
        return np.mean((x_scaled - x_rebuilt) ** 2, axis=1)

    def transform(self, x):
        # Compress to n_components dimensions. Shape: (rows, n_components)
        return self.model.transform(self.scaler.transform(x))

    def inverse_transform(self, x_compressed):
        # Rebuild to the original number of columns. Shape: (rows, n_features)
        # NOTE: this returns SCALED values, not the original units.
        return self.model.inverse_transform(x_compressed)

    def evaluate(self, percentile=99.0):
        '''
        Score every split and flag the worst rows.

        WHY A PERCENTILE AND NOT A RAW THRESHOLD
            A raw error value is impossible to choose sensibly - it depends
            on the scaling, the number of components and the data. A
            percentile is a decision you can state and defend: "I flagged the
            top 1% by reconstruction error."

            The cutoff is taken from TRAIN and applied to val and test, for
            the same reason the scaler is - so the three splits are judged by
            the same standard rather than each grading itself.

        Returns a dict of the raw error arrays, ready to pass into
        UnsupervisedAnomalyEval with higher_is_anomalous=True.
        '''
        train_errors = self.scores(self.x_train)
        val_errors = self.scores(self.x_val)
        test_errors = self.scores(self.x_test)

        cutoff = np.percentile(train_errors, percentile)

        for name, errors in [('train', train_errors),
                             ('validation', val_errors),
                             ('test', test_errors)]:
            flagged = errors > cutoff
            print(f"{name:<12} {len(errors):>8,} rows | "
                  f"{flagged.sum():>6,} flagged ({flagged.mean():.2%}) | "
                  f"error median {np.median(errors):.4f} max {errors.max():.4f}")

        print(f"\nCutoff: {cutoff:.4f} ({percentile}th percentile of train errors)")

        return {'train': train_errors, 'val': val_errors,
                'test': test_errors, 'cutoff': cutoff}

    def variance_curve(self, max_components=None):
        '''
        Fit PCA with every possible number of components and report the
        cumulative variance explained. Use this to CHOOSE n_components
        instead of guessing.

        Read it as: how many components do I need to capture ~90-95% of the
        normal structure? For anomaly detection you want enough to describe
        normal and no more - the directions you DISCARD are where the anomaly
        signal lives, so keeping all of them would reconstruct the anomalies
        perfectly too and every error would be zero.
        '''
        n = max_components or self.x_train.shape[1]

        x_scaled = self.scaler.fit_transform(self.x_train)
        full = PCA(n_components=n, random_state=42).fit(x_scaled)

        cumulative = np.cumsum(full.explained_variance_ratio_)

        return pd.DataFrame({
            'n_components': range(1, len(cumulative) + 1),
            'variance_explained': np.round(full.explained_variance_ratio_, 4),
            'cumulative': np.round(cumulative, 4),
        })
