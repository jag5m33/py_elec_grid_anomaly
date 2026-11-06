from sklearn.ensemble import IsolationForest as SkIsolationForest   # aliased: a class named IsolationForest below would shadow it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IsolationForestAnomalyDetector:
    '''
    UNSUPERVISED: there is no y labels. The model never sees a label, so it cannot be "right" or "wrong" during training 
        - it only learns what the bulk of the data looks like and flags whatever sits far from it.
        - How Isolation Forest works: it builds random trees by picking a random feature and a random split point, over and over. 
        - A point that is unusual gets separated from the rest after only a few splits; a point buried in the crowd takes many. 
        - "Path length to isolation" IS the anomaly score.
        - NOTE: this is the opposite logic to a supervised tree, which splits to maximise information gain about a target.
    '''
    
    def __init__(self, x_train, x_val, x_test, contamination='auto', n_estimators = 100):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test

        # contamination = what share of the data you ASSERT is anomalous.
            # No change to anomaly scores - it only sets where the -1 / +1 cutoff falls (model.offset_). 'auto' uses the original
        # paper's default rather than you guessing. Pick a number only when
        
        self.model = SkIsolationForest(
            contamination=contamination,
            n_estimators=n_estimators, # number of random trees; more = more stable scores
            max_samples='auto', # 256 rows per tree. Small on purpose: anomalies easier to isolate in a small sample than a large one
            random_state=42,
            n_jobs=-1
        )

    def train(self):
        # Note: fit(X) only. No y argument exists for an unsupervised model.
        print("Training unsupervised anomaly detection isolation forest model...")
        self.model.fit(self.x_train)

        # No .classes_ here - that is a supervised attribute. These exist instead:
        print(f"Trees fitted: {len(self.model.estimators_)}")
        print(f"Rows per tree: {self.model.max_samples_}")
        print(f"Decision offset: {self.model.offset_:.4f}")   # the cutoff contamination produced

    
    def _score(self, x, name):
        """
        Score one split and report how many rows were flagged.

        THE THREE OUTPUTS - same number, three forms
            score_samples(x)      the raw mark. How weird is this row?
                                  More negative = weirder. Always between -1 and 0.
                                  - It calculates the average number of random feature splits required to isolate a point across all trees. 
                                  - Shorter path lengths mean the point isolates very quickly, yielding a higher anomaly score

            offset_               where the line is drawn. ONE number, set by
                                  contamination. Below it = anomaly, above = normal.

            decision_function(x)  = score_samples(x) - offset_
                                  The same marks, shifted so the line sits at zero.
                                  Now you just check: is it negative?

            predict(x)            the yes/no answer. -1 if decision_function < 0, else +1.

        Worked example, with offset_ = -0.50
            score_samples:       -0.72    -0.45    -0.38
            minus offset_:       +0.50    +0.50    +0.50
            decision_function:   -0.22    +0.05    +0.12
            predict:               -1       +1       +1
                                anomaly   normal   normal

        RAW SCORES: Other scores don't tell you the extent to which a row is past threshold  

        No labels = No accuracy, precision, recall no indicator of genuine anomalies. 
        Model Role: Returns how many were flagged and how the scores are spread.

        Returns
            preds            -1 / +1 per row
            raw_scores       score_samples output
            decision_scores  decision_function output
            scores           dict of summary numbers
        """
        preds = self.model.predict(x)
        raw_scores = self.model.score_samples(x)
        decision_scores = self.model.decision_function(x)

        # WITHOUT LABELS THERE IS NO ACCURACY, PRECISION OR RECALL TO COMPUTE.
        # Nothing here says whether a flagged point is genuinely anomalous -
        # only how many were flagged and how the scores are distributed.
        # Evaluation is: do the flagged rows look wrong when you inspect them?

        n_anomalies = (preds == -1).sum()

        scores = {
            "rows": len(x),
            "n_anomalies": n_anomalies,
            "anomaly_rate": n_anomalies / len(x),
            "score_min": raw_scores.min(),
            "score_median": np.median(raw_scores),
            "score_max": raw_scores.max(),
        }

        print(f"{name:<12} {scores['rows']:>8,} rows | "
              f"{scores['n_anomalies']:>6,} flagged ({scores['anomaly_rate']:.2%}) | "
              f"score min {scores['score_min']:.4f} "
              f"median {scores['score_median']:.4f} "
              f"max {scores['score_max']:.4f}")

        return preds, raw_scores, decision_scores, scores

    def evaluate_train(self):
        return self._score(self.x_train, "train")

    def validate(self):
        # Validation here is NOT "check the answers". There are none.
        # What you are checking is STABILITY: the anomaly rate and score
        # distribution should look like train's. If validation flags 2% where
        # train flagged 8%, the two sets differ in some way worth understanding.
        return self._score(self.x_val, "validation")

    def eval_test(self):
        return self._score(self.x_test, "test")

    def plot_score_distribution(self, x, name="validation", bins=100):
        # THE MOST USEFUL PLOT IN UNSUPERVISED ANOMALY DETECTION.
        # A clear left tail separated from the main mass means the model found
        # a genuinely distinct group. A smooth unimodal curve means there is no
        # natural cut point, and whatever you flag is just "the lowest N%" indicative of decision not a data structure.
        scores = self.model.score_samples(x)

        plt.figure(figsize=(9, 5))
        plt.hist(scores, bins=bins, color='steelblue', edgecolor='none')
        plt.axvline(self.model.offset_, color='red', linestyle='--',
                    label=f'cutoff {self.model.offset_:.4f}')
        plt.xlabel('anomaly score (lower = more anomalous)')
        plt.ylabel('rows')
        plt.title(f'Anomaly score distribution - {name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        return scores

    def top_anomalies(self, x, n=20):
        # The real evaluation step: read the most anomalous rows and judge them
        # yourself. Sorted ascending because lower score = more anomalous.
        scores = self.model.score_samples(x)
        out = x.copy()
        out['anomaly_score'] = scores
        return out.sort_values('anomaly_score').head(n)

    def flagged_vs_normal(self, x, column):
        # Compare one feature between flagged and unflagged rows.
        # This is how you work out WHAT the model is reacting to - the same
        # move as splitting a feature by a second column: hold the grouping
        # fixed, look at the difference.
        preds = self.model.predict(x)
        return pd.DataFrame({
            'flagged': x.loc[preds == -1, column].describe(),
            'normal': x.loc[preds == 1, column].describe(),
        }).round(3)