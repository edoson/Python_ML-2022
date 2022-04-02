from sklearn.datasets import load_iris, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_2d_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_2d = X[:, :2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1
    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)
    return X_2d, y_2d


def make_circles_dataframe(n_samples, noise_level):
    points, label = make_circles(n_samples=n_samples, noise=noise_level)
    circles_df = pd.DataFrame(points, columns=['x','y'])
    circles_df['label'] = label
    circles_df.label = circles_df.label.map({0:'A', 1:'B'})
    return circles_df


def make_moons_dataframe(n_samples, noise_level):
    points, label = make_moons(n_samples=n_samples, noise=noise_level)
    moons_df = pd.DataFrame(points, columns=['x','y'])
    moons_df['label'] = label
    moons_df.label = moons_df.label.map({0:'A', 1:'B'})
    return moons_df