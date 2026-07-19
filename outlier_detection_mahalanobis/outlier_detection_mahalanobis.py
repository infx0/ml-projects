import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import math
from scipy.stats.distributions import chi2
from scipy.stats import norm
from scipy import stats
import seaborn as sns
from sklearn import datasets


class sigma_ellipse_plot:
    def __init__(
        self,
        std_devs: list[float],
        df=None,
        target="setosa",
        target_header="species",
        feature1="sepal_length",
        feature2="petal_width",
    ):

        self.data = df
        self.target = target
        self.feature1 = feature1
        self.feature2 = feature2
        self.target_header = target_header
        self.std_devs = std_devs
        self.largest_eigenvalue = None
        self.largest_eigenvector = None
        self.smallest_eigenvalue = None
        self.smallest_eigenvector = None
        self.angle = None
        self.mean = None
        self.r_ellipses = None
        self.mu_X = None
        self.mu_Y = None
        self.chisquare_val = None

    def get_data(self):

        self.data = self.data[self.data[self.target_header] == self.target].drop(
            self.target_header, axis=1
        )[[self.feature1, self.feature2]]

        return

    def get_eigens(self):

        covariance_matrix = self.data.cov()
        eigenvalues, eigenvectors = eigh(covariance_matrix)

        self.largest_eigenvector = eigenvectors[np.argmax(eigenvalues)]
        self.largest_eigenvalue = np.max(eigenvalues)
        self.smallest_eigenvector = eigenvectors[np.argmin(eigenvalues)]
        self.smallest_eigenvalue = np.min(eigenvalues)

        return

    def get_angle(self):

        self.angle = math.atan2(
            self.largest_eigenvector[1], self.largest_eigenvector[0]
        )

        return

    def shift_angle(self):

        if self.angle < 0:
            self.angle = self.angle + 2 * math.pi

        return

    def get_mean(self):

        self.mean = self.data.mean()

        return

    def get_chisquare_vals(self):

        self.chisquare_val = []
        for i in range(0, len(self.std_devs)):
            percent_covered = stats.norm.cdf(self.std_devs[i]) - stats.norm.cdf(
                self.std_devs[i] * -1
            )
            self.chisquare_val.append((chi2.ppf(percent_covered, df=2)) ** 0.5)

        return self.chisquare_val

    def get_ellipses(self):

        chisquare_val = self.get_chisquare_vals()

        self.r_ellipses = []
        for i in range(0, len(self.std_devs)):
            theta_grid = np.linspace(0, 2 * math.pi, 100)
            phi = self.angle
            self.mu_X = self.mean[0]
            self.mu_Y = self.mean[1]
            a = chisquare_val[i] * math.sqrt(self.largest_eigenvalue)
            b = chisquare_val[i] * math.sqrt(self.smallest_eigenvalue)

            ellipse_x_r = a * np.cos(theta_grid)
            ellipse_y_r = b * np.sin(theta_grid)

            R = [[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]]

            ellipses = np.array([ellipse_x_r, ellipse_y_r])

            r_ellipse = ellipses.T.dot(R).T

            self.r_ellipses.append(r_ellipse)

        return

    def get_labels(self, special_phrase=None):

        labels = []
        for i in range(0, len(self.std_devs)):
            if special_phrase is None:
                label = str(self.std_devs[i]) + " std. dev. from mean"
                labels.append(label)
            else:
                label = special_phrase + str(self.std_devs[i]) + " std. dev. from mean"
                labels.append(label)

        return labels

    def pipeline(self):

        self.get_data()
        self.get_eigens()
        self.get_angle()
        self.shift_angle()
        self.get_mean()
        self.get_ellipses()

        return self.data, self.r_ellipses, self.mu_X, self.mu_Y


## Import Dataset ##
if __name__ == "__main__":
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = (
        df.columns.str.replace(" ", "_")
        .str.replace(r"\(cm\)", "", regex=True)
        .str.lower()
        .str.rstrip("_")
    )
    df["species"] = iris.target
    df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    print(df.head())

    ## Feature Definition ##

    feature1 = "sepal_length"
    feature2 = "petal_width"

    ## Species Specific Ellipse Generation ##

    # add custom standard deviations
    custom_std_devs = [1.0, 2.0, 3.0]

    setosa_ellipses_obj = sigma_ellipse_plot(
        df=df,
        target="setosa",
        feature1=feature1,
        feature2=feature2,
        std_devs=custom_std_devs,
    )
    setosa_df, setosa_ellipses, setosa_mu_X, setosa_mu_Y = (
        setosa_ellipses_obj.pipeline()
    )
    setosa_plot_labels = setosa_ellipses_obj.get_labels()

    versicolor_ellipses_obj = sigma_ellipse_plot(
        df=df,
        target="versicolor",
        feature1=feature1,
        feature2=feature2,
        std_devs=custom_std_devs,
    )
    versicolor_df, versicolor_ellipses, versicolor_mu_X, versicolor_mu_Y = (
        versicolor_ellipses_obj.pipeline()
    )
    versicolor_plot_labels = versicolor_ellipses_obj.get_labels()

    virginica_ellipses_obj = sigma_ellipse_plot(
        df=df,
        target="virginica",
        feature1=feature1,
        feature2=feature2,
        std_devs=custom_std_devs,
    )
    virginica_df, virginica_ellipses, virginica_mu_X, virginica_mu_Y = (
        virginica_ellipses_obj.pipeline()
    )
    virginica_plot_labels = virginica_ellipses_obj.get_labels()

    ## Visualization of Ellipse Plots ##

    sns.set_style("white")

    df_subset = df[[feature1, feature2, "species"]]

    # palette order - setosa, virginica, versicolor
    petal_width_plot = sns.jointplot(
        data=df_subset,
        x=feature1,
        y=feature2,
        hue="species",
        palette=["#0747a1", "#d24e01", "#1e5631"],
        height=10,
    )
    colors_for_plot_setosa = ["#1065c0", "#1a8ae5", "#41a7f5"]
    colors_for_plot_virginica = ["#4c9a2a", "#68bb59", "#acdf87"]
    colors_for_plot_versicolor = ["#dc6601", "#e27602", "#e88504"]

    # Dynamic title for chart
    plt.suptitle(f"Error Ellipses for {feature1} and {feature2}")

    plt.scatter(
        setosa_mu_X,
        setosa_mu_Y,
        c="#1F51FF",
        s=150,
        label="Setosa mean",
        marker="$\mu$",
    )
    plt.scatter(
        versicolor_mu_X,
        versicolor_mu_Y,
        c="#FF6700",
        s=150,
        label="Versicolor mean",
        marker="$\mu$",
    )
    plt.scatter(
        virginica_mu_X,
        virginica_mu_Y,
        c="#2Bc20e",
        s=150,
        label="Virginica mean",
        marker="$\mu$",
    )
    plt.grid()
    plt.legend()

    # Plots the ellipses for each species, with the appropriate colors
    for i in range(0, len(setosa_ellipses)):
        plt.plot(
            setosa_ellipses[i][0] + setosa_mu_X,
            setosa_ellipses[i][1] + setosa_mu_Y,
            colors_for_plot_setosa[i],
            label=setosa_plot_labels[i],
        )
        plt.plot(
            versicolor_ellipses[i][0] + versicolor_mu_X,
            versicolor_ellipses[i][1] + versicolor_mu_Y,
            colors_for_plot_versicolor[i],
            label=versicolor_plot_labels[i],
        )
        plt.plot(
            virginica_ellipses[i][0] + virginica_mu_X,
            virginica_ellipses[i][1] + virginica_mu_Y,
            colors_for_plot_virginica[i],
            label=virginica_plot_labels[i],
        )

    # Save the plot as an image
    plt.savefig("MahalanobisOultlierExample.svg")

    # Show the plot (optional)
    plt.show()
