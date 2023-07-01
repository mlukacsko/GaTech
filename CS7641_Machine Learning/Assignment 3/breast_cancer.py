from scipy.stats import kurtosis, skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import scipy.stats
import time
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
cancer_dataset = pd.read_csv("Datasets/breast-cancer.csv")
print(cancer_dataset.info())

X = cancer_dataset.drop(['id','diagnosis'], axis=1)
y = cancer_dataset["diagnosis"]

# diagnosis set to binary values for B, M
y[y == 'B'] = 0
y[y == 'M'] = 1
y = y.astype(int)

# normalize X values
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier_accuracy = np.zeros(6)
time_train = np.zeros(6)
time_infer = np.zeros(6)

"""
PART 1 - Clustering
K Means
"""

clusters = np.arange(1,25,1)
inertia = []
ami_score = []
for c in clusters:
    k_means_clustering = KMeans(n_clusters=c, random_state=3)
    k_means_clustering.fit(X)
    inertia.append(k_means_clustering.inertia_)
    ami_score.append(adjusted_mutual_info_score(y, k_means_clustering.labels_))
inertia = np.array(inertia)
plt.plot(clusters,inertia)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('Clusters vs. Inertia')
plt.grid()
plt.savefig('cancer_kmeans_inertia.png')
plt.show()


range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()


plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('cancer_kmeans_silscore.png')
plt.show()

k = 2
k_means_clustering = KMeans(n_clusters=k, random_state=3)
k_means_clustering.fit(X)
kmeans_labels = k_means_clustering.predict(X)
print('K=2 Inertia: ', k_means_clustering.inertia_)
silhouette_score_value = silhouette_score(X, k_means_clustering.labels_)
print('K=2 Silhouette score: ', silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
print('K=2 Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

# perform K-means clustering
ari_kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = ari_kmeans.fit_predict(X)

# plot the original data points and the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.2, marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means clustering results vs true labels')
plt.show()

ari = adjusted_rand_score(y, y_pred)
print(f"Adjusted Rand Index: {ari:.3f}")

plt.figure()
plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('Distribution Of Data Per Cluster')
plt.grid()
plt.savefig('cancer_kmeans_data_distribution.png')
plt.show()


"""
PART 1 - Clustering
Expectation Maximization (EM)
"""
components = range(1,31)
covariances = ['spherical', 'tied', 'diag', 'full']
spherical_bic_score, spherical_aic_score = [], []
tied_bic_score, tied_aic_score = [], []
diag_bic_score, diag_aic_score = [], []
full_bic_score, full_aic_score = [], []

for components in components:
    gmm = GaussianMixture(n_components=components, covariance_type='spherical', random_state=3)
    gmm.fit(X)
    spherical_bic_score.append(gmm.bic(X))
    spherical_aic_score.append(gmm.aic(X))
    gmm = GaussianMixture(n_components=components, covariance_type='tied', random_state=3)
    gmm.fit(X)
    tied_bic_score.append(gmm.bic(X))
    tied_aic_score.append(gmm.aic(X))
    gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=3)
    gmm.fit(X)
    diag_bic_score.append(gmm.bic(X))
    diag_aic_score.append(gmm.aic(X))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=3)
    gmm.fit(X)
    full_bic_score.append(gmm.bic(X))
    full_aic_score.append(gmm.aic(X))


plt.figure()
plt.plot(range(1,31), np.array(spherical_bic_score), label = 'Spherical')
plt.plot(range(1,31), np.array(tied_bic_score), label = 'Tied')
plt.plot(range(1,31), np.array(diag_bic_score), label = 'Diag')
plt.plot(range(1,31), np.array(full_bic_score), label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("Number of Components vs. BIC")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('cancer_gmm_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), spherical_aic_score, label = 'Spherical')
plt.plot(range(1,31), tied_aic_score, label = 'Tied')
plt.plot(range(1,31), diag_aic_score, label = 'Diag')
plt.plot(range(1,31), full_aic_score, label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("Number of Components vs. AIC")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('cancer_gmm_aic.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=3)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('cancer_gmm_silscore.png')
plt.show()

best_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=3)
best_gmm.fit(X)
gmm_labels = best_gmm.predict(X)
silhouette_score_value = silhouette_score(X, gmm_labels)
print('Silhouette score: ', silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, gmm_labels)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(gmm_labels, bins=np.arange(0, 3) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, 2))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('Distribution of Data Per Cluster')
plt.grid()
plt.savefig('cancer_gmm_data_distribution.png')
plt.show()


# fit EM model
y_pred = best_gmm.fit_predict(X)

# compute ARI
ari = adjusted_rand_score(y, y_pred)
print("ARI:", ari)


"""
Part 2 - Dimensionality reduction
PCA
"""
pca = PCA(random_state=3)
pca.fit(X)
print('Variance explained by all components: ', sum(pca.explained_variance_ratio_ * 100))
print()
print('Cumulative Variance for all components: ', np.cumsum(pca.explained_variance_ratio_ * 100))

plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.title('Cumulative Explained Variance vs. PCA Component')
plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1))
plt.grid()
plt.savefig('cancer_pca_cumsum_explained_variance.png')
plt.show()

plt.plot(range(1,31), pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Per Component')
plt.grid()
plt.savefig('cancer_pca_explained_variance.png')
plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
pca_8 = PCA(n_components=7)
pca_8.fit(X)
X_pca = pca_8.fit_transform(X)
print('Variance explained by 7 components: ', sum(pca_8.explained_variance_ratio_ * 100))
print()
print('Cumulative Variance for 7 components: ', np.cumsum(pca_8.explained_variance_ratio_ * 100))



for i in range(2):
    plt.scatter(X_pca[kmeans.labels_ == i, 0], X_pca[kmeans.labels_ == i, 1], label='Class {}'.format(i+1))
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Breast Cancer Dataset after PCA (7 Features)')
plt.legend()
plt.savefig('cancer_pca_8feature_scatter.png')
plt.show()

df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7'])
df['target'] = y
scatter_matrix(df, alpha=0.8, c=df['target'], figsize=(10, 10), diagonal='hist')
plt.savefig('cancer_pca_7feature_matrix.png')
plt.show()


"""
Part 2 - Dimensionality reduction
ICA
"""
kurtosis_values = []
for components in range(1,31):
    X_ICA = FastICA(n_components = components).fit_transform(X)
    kur = scipy.stats.kurtosis(X_ICA)
    print('kurtosis with ', components, ' components: ', (np.mean(kur)/components))
    kurtosis_values.append(np.mean(kur)/components)
kurtosis_values = np.array(kurtosis_values)
plt.plot(np.arange(1,31),kurtosis_values)
plt.xlabel('Components')
plt.ylabel('Normalized Mean Kurtosis Value')
plt.grid()
plt.title('Normalized Mean Kurtosis Value vs. Components')
plt.savefig('cancer_ica_kurtosis.png')
plt.show()

skew_ica = FastICA(n_components=30, random_state=3)
S = skew_ica.fit_transform(X)
skewness_values = []
for i in range(S.shape[1]):
    skewness = skew(S[:, i])
    skewness_values.append(skewness)
ranked_indices = np.argsort(skewness_values)[::-1]
num_components_to_select = 12
selected_indices = ranked_indices[:num_components_to_select]
selected_components = S[:, selected_indices]
plt.figure()
plt.bar(range(len(skewness_values)), skewness_values)
plt.xlabel('Independent component')
plt.ylabel('Skewness')
plt.title('Skewness Values For Independent Components')
plt.grid()
plt.savefig('cancer_ica_skewness.png')
plt.show()

ica = FastICA(n_components=12)
ica.fit(X)
components = ica.transform(X)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the colors for each component
colors = ['red', 'green']

# Plot each component with a different color
for i in range(2):
    x = components[:, i]
    y = components[:, (i+1)%8]
    z = components[:, (i+2)%8]
    ax.scatter(x, y, z, c=colors[i])

# Set the axis labels and legend
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.legend(['Component 1', 'Component 2'])
plt.title('Cancer Dataset after ICA (12 Components)')
plt.savefig('cancer_ica_3d_scatter.png')
plt.show()

"""
Part 2 - Dimensionality reduction
RP
"""

max_components = 30
reconstruction_errors = []
for n_components in range(1, max_components+1):
    grp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_proj = grp.fit_transform(X)
    X_reconstructed = grp.inverse_transform(X_proj)
    error = ((X - X_reconstructed) ** 2).mean()
    reconstruction_errors.append(error)

# Plot the reconstruction errors as a function of the number of components
plt.plot(range(1, max_components+1), reconstruction_errors)
plt.xlabel('Number of components')
plt.ylabel('Reconstruction error')
plt.title('Reconstruction Error vs Number of components')
plt.grid()
plt.xticks(np.arange(1,30 + 1))
plt.savefig('cancer_rp_error.png')
plt.show()

max_components = 30
num_repeats = 50
reconstruction_variances = []
for n_components in range(1, max_components+1):
    errors = []
    for i in range(num_repeats):
        grp = GaussianRandomProjection(n_components=n_components, random_state=i)
        X_proj = grp.fit_transform(X)
        X_reconstructed = grp.inverse_transform(X_proj)
        error = ((X - X_reconstructed) ** 2).mean()
        errors.append(error)
    variance = np.var(errors)
    print('component: ', n_components, ' variance: ', variance)
    reconstruction_variances.append(variance)

# Plot the reconstruction variances as a function of the number of components
plt.plot(range(1, max_components+1), reconstruction_variances)
plt.xlabel('Number of components')
plt.ylabel('Reconstruction variance')
plt.title('Reconstruction variance vs Number of components')
plt.grid()
plt.savefig('cancer_rp_variance.png')
plt.show()

grp = GaussianRandomProjection(n_components=17, random_state=42)
components = grp.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(components.shape[1]):
    color = np.random.rand(3,) # Generate a random RGB color for each component
    x = components[:, i]
    y = components[:, (i+1) % components.shape[1]]
    z = components[:, (i+2) % components.shape[1]]
    ax.scatter(x, y, z, c=color, label=f'Component {i+1}')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.title('Cancer Dataset after Random Projections (17 Components)')
ax.legend()
plt.savefig('cancer_rp_3d_scatter.png')
plt.show()

"""
Part 2 - Dimensionality reduction
You Pick One!
ExtraTreesClassifier
"""
df = pd.read_csv("Datasets/breast-cancer.csv")
df_X = cancer_dataset.drop(['id','diagnosis'], axis=1)
df_y = cancer_dataset["diagnosis"]

# diagnosis set to binary values for B, M
df_y[df_y == 'B'] = 0
df_y[df_y == 'M'] = 1
df_y = df_y.astype(int)

extra_tree_forest = ExtraTreesClassifier(n_estimators=5,
                                         criterion='entropy', max_features=2)

# Training the model
extra_tree_forest.fit(df_X, df_y)

# Computing the importance of each feature
feature_importance = extra_tree_forest.feature_importances_

# Normalizing the individual importances
feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                        extra_tree_forest.estimators_],
                                       axis=0)

# Plotting a Bar Graph to compare the models
plt.bar(df_X.columns, feature_importance_normalized)
plt.xlabel('Feature Labels')
plt.ylabel('Feature Importances')
plt.xticks(rotation=90)
plt.title('Breast Cancer Dataset Feature Importance - Extra Tree Classifier')
plt.tight_layout()
plt.savefig('cancer_extraTree_feature_importance.png')
plt.show()

importances = extra_tree_forest.feature_importances_
indices = np.argsort(importances)[::-1]
n_features = 5  # Select the top 5 features
selected_indices = indices[:n_features]
X_reduced = X[:, selected_indices]
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Breast Cancer Dataset Using Top 10 Features')
plt.savefig('cancer_extraTree_3d_feature_importance.png')
plt.show()

X_PCA = PCA(n_components = 7, random_state = 3).fit_transform(X)
X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(X_PCA, y, test_size=0.2, random_state=3)
X_ICA = FastICA(n_components = 7).fit_transform(X)
X_ICA_train, X_ICA_test, y_ICA_train, y_ICA_test = train_test_split(X_ICA, y, test_size=0.2, random_state=3)
X_RP = GaussianRandomProjection(n_components = 7).fit_transform(X)
X_RP_train, X_RP_test, y_RP_train, y_RP_test = train_test_split(X_RP, y, test_size=0.2, random_state=3)
tree = ExtraTreesClassifier(n_estimators=100)
tree= tree.fit(X,y)
model = SelectFromModel(tree, prefit=True)
X_ETC = model.transform(X)
X_ETC_train, X_ETC_test, y_Trees_train, y_Trees_test = train_test_split(X_ETC, y, test_size=0.2, random_state=3)

"""
Part 3
Run Clustering on dimensional reduced output
K-Means - PCA
"""
print('starting PCA')
clusters = np.arange(1,25,1)
pca_inertia = []
ami_score = []

for n_clusters in clusters:
    k_means_clustering = KMeans(n_clusters=n_clusters, random_state=3)
    k_means_clustering.fit(X_PCA)
    pca_inertia.append(k_means_clustering.inertia_)
    ami_score.append(adjusted_mutual_info_score(y, k_means_clustering.labels_))
pca_inertia = np.array(pca_inertia)
plt.plot(clusters,pca_inertia)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('(PCA) Clusters vs. Inertia')
plt.grid()
plt.savefig('Cancer_Kmeans_and_PCA/cancer_kmeans_inertia_pca.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()


plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(PCA) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_Kmeans_and_PCA/cancer_kmeans_silscore.png')
plt.show()


k = 2
k_means_clustering = KMeans(n_clusters=k, random_state=3)
k_means_clustering.fit(X_PCA)
pca_inertia_score = k_means_clustering.inertia_
print('Inertia: ', k_means_clustering.inertia_)
pca_silhouette_score_value = silhouette_score(X_PCA, k_means_clustering.labels_)
print('Silhouette score: ', pca_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(PCA) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_Kmeans_and_PCA/cancer_kmeans_data_distribution.png')
plt.show()

ari_kmeans = KMeans(n_clusters=3, random_state=3)
y_pred = ari_kmeans.fit_predict(X_PCA)
pca_ari = adjusted_rand_score(y, y_pred)
print(f"Adjusted Rand Index: {pca_ari:.3f}")

# plot the original data points and the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Predicted')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.2, marker='x', label='True')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('K-means clustering results vs true labels')
plt.savefig('Cancer_Kmeans_and_PCA/cancer_kmeans_actual_vs_true_plot.png')
plt.show()
print('PCA Done')

"""
Part 3
Run Clustering on dimensional reduced output
K-Means - ICA
"""
print('starting ICA')
clusters = np.arange(1,25,1)
ica_inertia = []
ami_score = []

for n_clusters in clusters:
    k_means_clustering = KMeans(n_clusters=n_clusters, random_state=3)
    k_means_clustering.fit(X_ICA)
    ica_inertia.append(k_means_clustering.inertia_)
ica_inertia = np.array(ica_inertia)
plt.plot(clusters,ica_inertia)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('(ICA) Clusters vs. Inertia')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ICA/cancer_kmeans_inertia.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()


plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(ICA) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ICA/cancer_kmeans_silscore.png')
plt.show()


k = 2
k_means_clustering = KMeans(n_clusters=k, random_state=3)
k_means_clustering.fit(X_ICA)
ica_inertia_score = k_means_clustering.inertia_
print('Inertia: ', k_means_clustering.inertia_)
ica_silhouette_score_value = silhouette_score(X_ICA, k_means_clustering.labels_)
print('Silhouette score: ', ica_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(ICA) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ICA/cancer_data_distribution.png')
plt.show()

ari_kmeans = KMeans(n_clusters=3, random_state=3)
y_pred = ari_kmeans.fit_predict(X_ICA)
ica_ari = adjusted_rand_score(y, y_pred)
print(f"Adjusted Rand Index: {ica_ari:.3f}")
# plot the original data points and the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Predicted')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.2, marker='x', label='True')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('(ICA) K-means clustering results vs true labels')
plt.savefig('Cancer_Kmeans_and_ICA/cancer_kmeans_actual_vs_true_plot.png')
plt.show()
print('ICA Done')

"""
Part 3
Run Clustering on dimensional reduced output
K-Means - RP
"""
print('starting RP')
clusters = np.arange(1,25,1)
rp_inertia = []
ami_score = []

for n_clusters in clusters:
    k_means_clustering = KMeans(n_clusters=n_clusters, random_state=3)
    k_means_clustering.fit(X_RP)
    rp_inertia.append(k_means_clustering.inertia_)
rp_inertia = np.array(rp_inertia)
plt.plot(clusters,rp_inertia)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('(RP) Clusters vs. Inertia')
plt.grid()
plt.savefig('Cancer_Kmeans_and_RP/cancer_kmeans_inertia.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()


plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(RP) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_Kmeans_and_RP/cancer_kmeans_silscore.png')
plt.show()


k = 2
k_means_clustering = KMeans(n_clusters=k, random_state=3)
k_means_clustering.fit(X_RP)
rp_inertia_score = k_means_clustering.inertia_
print('Inertia: ', k_means_clustering.inertia_)
rp_silhouette_score_value = silhouette_score(X_RP, k_means_clustering.labels_)
print('Silhouette score: ', rp_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(RP) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_Kmeans_and_RP/cancer_kmeans_data_distribution.png')
plt.show()

ari_kmeans = KMeans(n_clusters=3, random_state=3)
y_pred = ari_kmeans.fit_predict(X_RP)
rp_ari = adjusted_rand_score(y, y_pred)
print(f"Adjusted Rand Index: {rp_ari:.3f}")

# plot the original data points and the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Predicted')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.2, marker='x', label='True')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('(RP) K-means clustering results vs true labels')
plt.savefig('Cancer_Kmeans_and_RP/cancer_kmeans_actual_vs_true_plot.png')
plt.show()
print('RP Done')

"""
Part 3
Run Clustering on dimensional reduced output
K-Means - ETC
"""
print('starting ETC')
clusters = np.arange(1,25,1)
etc_inertia = []
ami_score = []

for n_clusters in clusters:
    k_means_clustering = KMeans(n_clusters=n_clusters, random_state=3)
    k_means_clustering.fit(X_ETC)
    etc_inertia.append(k_means_clustering.inertia_)
etc_inertia = np.array(etc_inertia)
plt.plot(clusters,etc_inertia)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('(ETC) Clusters vs. Inertia')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ETC/cancer_kmeans_inertia.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()


plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(ETC) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ETC/cancer_kmeans_silscore.png')
plt.show()


k = 2
k_means_clustering = KMeans(n_clusters=k, random_state=3)
k_means_clustering.fit(X_ETC)
etc_inertia_score = k_means_clustering.inertia_
print('Inertia: ', k_means_clustering.inertia_)
etc_silhouette_score_value = silhouette_score(X_ETC, k_means_clustering.labels_)
print('Silhouette score: ', etc_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(ETC) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_Kmeans_and_ETC/cancer_kmeans_data_distribution.png')
plt.show()

ari_kmeans = KMeans(n_clusters=3, random_state=3)
y_pred = ari_kmeans.fit_predict(X_ETC)
etc_ari = adjusted_rand_score(y, y_pred)
print(f"Adjusted Rand Index: {etc_ari:.3f}")

# plot the original data points and the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Predicted')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.2, marker='x', label='True')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('(ETC) K-means clustering results vs true labels')
plt.savefig('Cancer_Kmeans_and_ETC/cancer_kmeans_actual_vs_true_plot.png')
plt.show()
print('ETC Done')

fig = plt.figure(figsize = (10, 5))
x = ['PCA','ICA','RP','ETC']
plt.bar(x,[pca_inertia_score,ica_inertia_score,rp_inertia_score,etc_inertia_score])
plt.title("K-means Inertia Score (cancer)")
plt.xlabel("Algorithms")
plt.ylabel("Inertia")
plt.savefig('Cancer_Kmeans_all/all_intertia.png')
plt.show()

fig = plt.figure(figsize = (10, 5))
x = ['PCA','ICA','RP','ETC']
plt.bar(x,[pca_silhouette_score_value,ica_silhouette_score_value,rp_silhouette_score_value,etc_silhouette_score_value])
plt.title("K-means Silhouette Score (cancer)")
plt.xlabel("Algorithms")
plt.ylabel("Silhouette")
plt.savefig('Cancer_Kmeans_all/cancer_all_silscore.png')
plt.show()

fig = plt.figure(figsize = (10, 5))
x = ['PCA','ICA','RP','ETC']
plt.bar(x, [pca_ari,ica_ari,rp_ari,etc_ari])
plt.title("K-means ARI Scores (cancer)")
plt.xlabel("Algorithms")
plt.ylabel("ARI")
plt.savefig('Cancer_Kmeans_all/cancer_all_ariscore.png')
plt.show()

"""
Part 3
Run Clustering on dimensional reduced output
EM - PCA
"""
print('starting PCA')
components = range(1,31)
covariances = ['spherical', 'tied', 'diag', 'full']
spherical_bic_score, spherical_aic_score = [], []
tied_bic_score, tied_aic_score = [], []
diag_bic_score, diag_aic_score = [], []
full_bic_score, full_aic_score = [], []
pca_bic, pca_aic = [],[]

for components in components:
    gmm = GaussianMixture(n_components=components, covariance_type='spherical', random_state=3)
    gmm.fit(X_PCA)
    spherical_bic_score.append(gmm.bic(X_PCA))
    spherical_aic_score.append(gmm.aic(X_PCA))
    gmm = GaussianMixture(n_components=components, covariance_type='tied', random_state=3)
    gmm.fit(X_PCA)
    tied_bic_score.append(gmm.bic(X_PCA))
    tied_aic_score.append(gmm.aic(X_PCA))
    gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=3)
    gmm.fit(X_PCA)
    diag_bic_score.append(gmm.bic(X_PCA))
    diag_aic_score.append(gmm.aic(X_PCA))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=3)
    gmm.fit(X_PCA)
    full_bic_score.append(gmm.bic(X_PCA))
    full_aic_score.append(gmm.aic(X_PCA))
    pca_bic.append(gmm.bic(X_PCA))
    pca_aic.append(gmm.aic(X_PCA))


plt.figure()
plt.plot(range(1,31), np.array(spherical_bic_score), label = 'Spherical')
plt.plot(range(1,31), np.array(tied_bic_score), label = 'Tied')
plt.plot(range(1,31), np.array(diag_bic_score), label = 'Diag')
plt.plot(range(1,31), np.array(full_bic_score), label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(PCA) Number of Components vs. BIC")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_PCA/cancer_gmm_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), spherical_aic_score, label = 'Spherical')
plt.plot(range(1,31), tied_aic_score, label = 'Tied')
plt.plot(range(1,31), diag_aic_score, label = 'Diag')
plt.plot(range(1,31), full_aic_score, label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(PCA) Number of Components vs. AIC")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_PCA/cancer_gmm_aic.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=3)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(PCA) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_EM_and_PCA/cancer_gmm_silscore.png')
plt.show()

best_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=3)
best_gmm.fit(X_PCA)
gmm_labels = best_gmm.predict(X_PCA)
pca_silhouette_score_value = silhouette_score(X_PCA, gmm_labels)
print('Silhouette score: ', pca_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, gmm_labels)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(gmm_labels, bins=np.arange(0, 3) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, 2))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(PCA) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_EM_and_PCA/cancer_gmm_data_distribution.png')
plt.show()


# fit EM model
y_pred = best_gmm.fit_predict(X_PCA)

# compute ARI
pca_ari = adjusted_rand_score(y, y_pred)
print("ARI:", pca_ari)
print('PCA Done')

"""
Part 3
Run Clustering on dimensional reduced output
EM - ICA
"""
print("ICA Starting")
components = range(1,31)
covariances = ['spherical', 'tied', 'diag', 'full']
spherical_bic_score, spherical_aic_score = [], []
tied_bic_score, tied_aic_score = [], []
diag_bic_score, diag_aic_score = [], []
full_bic_score, full_aic_score = [], []
ica_bic, ica_aic = [],[]

for components in components:
    gmm = GaussianMixture(n_components=components, covariance_type='spherical', random_state=3)
    gmm.fit(X_ICA)
    spherical_bic_score.append(gmm.bic(X_ICA))
    spherical_aic_score.append(gmm.aic(X_ICA))
    gmm = GaussianMixture(n_components=components, covariance_type='tied', random_state=3)
    gmm.fit(X_ICA)
    tied_bic_score.append(gmm.bic(X_ICA))
    tied_aic_score.append(gmm.aic(X_ICA))
    gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=3)
    gmm.fit(X_ICA)
    diag_bic_score.append(gmm.bic(X_ICA))
    diag_aic_score.append(gmm.aic(X_ICA))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=3)
    gmm.fit(X_ICA)
    full_bic_score.append(gmm.bic(X_ICA))
    full_aic_score.append(gmm.aic(X_ICA))
    ica_bic.append(gmm.bic(X_ICA)*-1)
    ica_aic.append(gmm.aic(X_ICA)*-1)


plt.figure()
plt.plot(range(1,31), np.array(spherical_bic_score), label = 'Spherical')
plt.plot(range(1,31), np.array(tied_bic_score), label = 'Tied')
plt.plot(range(1,31), np.array(diag_bic_score), label = 'Diag')
plt.plot(range(1,31), np.array(full_bic_score), label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(ICA) Number of Components vs. BIC")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_ICA/cancer_gmm_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), spherical_aic_score, label = 'Spherical')
plt.plot(range(1,31), tied_aic_score, label = 'Tied')
plt.plot(range(1,31), diag_aic_score, label = 'Diag')
plt.plot(range(1,31), full_aic_score, label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(ICA) Number of Components vs. AIC")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_ICA/cancer_gmm_aic.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=3)
    cluster_labels = clusterer.fit_predict(X_ICA)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_ICA, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(ICA) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_EM_and_ICA/cancer_gmm_silscore.png')
plt.show()

best_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=3)
best_gmm.fit(X_ICA)
gmm_labels = best_gmm.predict(X_ICA)
ica_silhouette_score_value = silhouette_score(X_ICA, gmm_labels)
print('Silhouette score: ', ica_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, gmm_labels)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(gmm_labels, bins=np.arange(0, 3) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, 2))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(ICA) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_EM_and_ICA/cancer_gmm_data_distribution.png')
plt.show()


# fit EM model
y_pred = best_gmm.fit_predict(X_ICA)

# compute ARI
ica_ari = adjusted_rand_score(y, y_pred)
print("ARI:", ica_ari)
print('ICA Done')
"""
Part 3
Run Clustering on dimensional reduced output
EM - RP
"""
print("RP Starting")
components = range(1,31)
covariances = ['spherical', 'tied', 'diag', 'full']
spherical_bic_score, spherical_aic_score = [], []
tied_bic_score, tied_aic_score = [], []
diag_bic_score, diag_aic_score = [], []
full_bic_score, full_aic_score = [], []
rp_bic, rp_aic = [], []

for components in components:
    gmm = GaussianMixture(n_components=components, covariance_type='spherical', random_state=3)
    gmm.fit(X_RP)
    spherical_bic_score.append(gmm.bic(X_RP))
    spherical_aic_score.append(gmm.aic(X_RP))
    gmm = GaussianMixture(n_components=components, covariance_type='tied', random_state=3)
    gmm.fit(X_RP)
    tied_bic_score.append(gmm.bic(X_RP))
    tied_aic_score.append(gmm.aic(X_RP))
    gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=3)
    gmm.fit(X_RP)
    diag_bic_score.append(gmm.bic(X_RP))
    diag_aic_score.append(gmm.aic(X_RP))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=3)
    gmm.fit(X_RP)
    full_bic_score.append(gmm.bic(X_RP))
    full_aic_score.append(gmm.aic(X_RP))
    rp_bic.append(gmm.bic(X_RP))
    rp_aic.append(gmm.aic(X_RP))



plt.figure()
plt.plot(range(1,31), np.array(spherical_bic_score), label = 'Spherical')
plt.plot(range(1,31), np.array(tied_bic_score), label = 'Tied')
plt.plot(range(1,31), np.array(diag_bic_score), label = 'Diag')
plt.plot(range(1,31), np.array(full_bic_score), label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(RP) Number of Components vs. BIC")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_RP/Cancer_gmm_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), spherical_aic_score, label = 'Spherical')
plt.plot(range(1,31), tied_aic_score, label = 'Tied')
plt.plot(range(1,31), diag_aic_score, label = 'Diag')
plt.plot(range(1,31), full_aic_score, label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(RP) Number of Components vs. AIC")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_RP/cancer_gmm_aic.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=3)
    cluster_labels = clusterer.fit_predict(X_RP)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_RP, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(RP) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_EM_and_RP/cancer_gmm_silscore.png')
plt.show()

best_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=3)
best_gmm.fit(X_RP)
gmm_labels = best_gmm.predict(X_RP)
rp_silhouette_score_value = silhouette_score(X_RP, gmm_labels)
print('Silhouette score: ', rp_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, gmm_labels)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(gmm_labels, bins=np.arange(0, 3) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, 2))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(RP) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_EM_and_RP/cancer_gmm_data_distribution.png')
plt.show()


# fit EM model
y_pred = best_gmm.fit_predict(X_RP)

# compute ARI
rp_ari = adjusted_rand_score(y, y_pred)
print("ARI:", rp_ari)
print('RP Done')

"""
Part 3
Run Clustering on dimensional reduced output
EM - RP
"""
print("ETC Starting")
components = range(1,31)
covariances = ['spherical', 'tied', 'diag', 'full']
spherical_bic_score, spherical_aic_score = [], []
tied_bic_score, tied_aic_score = [], []
diag_bic_score, diag_aic_score = [], []
full_bic_score, full_aic_score = [], []
etc_bic, etc_aic = [], []

for components in components:
    gmm = GaussianMixture(n_components=components, covariance_type='spherical', random_state=3)
    gmm.fit(X_ETC)
    spherical_bic_score.append(gmm.bic(X_ETC))
    spherical_aic_score.append(gmm.aic(X_ETC))
    gmm = GaussianMixture(n_components=components, covariance_type='tied', random_state=3)
    gmm.fit(X_ETC)
    tied_bic_score.append(gmm.bic(X_ETC))
    tied_aic_score.append(gmm.aic(X_ETC))
    gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=3)
    gmm.fit(X_ETC)
    diag_bic_score.append(gmm.bic(X_ETC))
    diag_aic_score.append(gmm.aic(X_ETC))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=3)
    gmm.fit(X_ETC)
    full_bic_score.append(gmm.bic(X_ETC))
    full_aic_score.append(gmm.aic(X_ETC))
    etc_bic.append(gmm.bic(X_ETC)*-1)
    etc_aic.append(gmm.aic(X_ETC)*-1)


plt.figure()
plt.plot(range(1,31), np.array(spherical_bic_score), label = 'Spherical')
plt.plot(range(1,31), np.array(tied_bic_score), label = 'Tied')
plt.plot(range(1,31), np.array(diag_bic_score), label = 'Diag')
plt.plot(range(1,31), np.array(full_bic_score), label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(ETC) Number of Components vs. BIC")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_ETC/cancer_gmm_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), spherical_aic_score, label = 'Spherical')
plt.plot(range(1,31), tied_aic_score, label = 'Tied')
plt.plot(range(1,31), diag_aic_score, label = 'Diag')
plt.plot(range(1,31), full_aic_score, label = 'Full')
plt.legend()
plt.xticks(range(1,31))
plt.title("(ETC) Number of Components vs. AIC")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('Cancer_EM_and_ETC/cancer_gmm_aic.png')
plt.show()

range_n_clusters = np.arange(2, 7, 1)
sil_score = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=3)
    cluster_labels = clusterer.fit_predict(X_ETC)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sil_score.append(silhouette_avg)
    print( "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_ETC, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

plt.plot(range_n_clusters,sil_score)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('(ETC) Clusters vs. Silhouette Score')
plt.grid()
plt.savefig('Cancer_EM_and_ETC/cancer_gmm_silscore.png')
plt.show()

best_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=3)
best_gmm.fit(X_ETC)
gmm_labels = best_gmm.predict(X_ETC)
etc_silhouette_score_value = silhouette_score(X_ETC, gmm_labels)
print('Silhouette score: ', etc_silhouette_score_value)
adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, gmm_labels)
print('Adjusted Mutual Information (AMI) score: ', adjusted_mutual_info_score_value)

plt.figure()
plt.hist(gmm_labels, bins=np.arange(0, 3) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, 2))
plt.xlabel('Cluster')
plt.ylabel('Samples per Cluster')
plt.title('(ETC) Distribution of Data Per Cluster')
plt.grid()
plt.savefig('Cancer_EM_and_ETC/cancer_gmm_data_distribution.png')
plt.show()


# fit EM model
y_pred = best_gmm.fit_predict(X_ETC)

# compute ARI
etc_ari = adjusted_rand_score(y, y_pred)
print("ARI:", etc_ari)
print('ETC Done')

plt.figure()
plt.plot(range(1,31), pca_bic, label = 'PCA')
plt.plot(range(1,31), ica_bic, label = 'ICA')
plt.plot(range(1,31), rp_bic, label = 'RP')
plt.plot(range(1,31), etc_bic, label = 'ETC')
plt.legend()
plt.xticks(range(1,31))
plt.title("EM Number of Components vs. BIC (cancer)")
plt.xlabel("Number of Components")
plt.ylabel("BIC Values")
plt.grid()
plt.savefig('Cancer_EM_all/cancer_all_bic.png')
plt.show()

plt.figure()
plt.plot(range(1,31), pca_aic, label = 'PCA')
plt.plot(range(1,31), ica_aic, label = 'ICA')
plt.plot(range(1,31), rp_aic, label = 'RP')
plt.plot(range(1,31), etc_aic, label = 'ETC')
plt.legend()
plt.xticks(range(1,31))
plt.title("EM Number of Components vs. AIC (cancer)")
plt.xlabel("Number of Components")
plt.ylabel("AIC Values")
plt.grid()
plt.savefig('Cancer_EM_all/aic.png')
plt.show()

fig = plt.figure(figsize = (10, 5))
x = ['PCA','ICA','RP','ETC']
plt.bar(x,[pca_ari,ica_ari, rp_ari, etc_ari])
plt.title("EM ARI Score (cancer)")
plt.xlabel("Algorithms")
plt.ylabel("ARI")
plt.savefig('Cancer_EM_all/all_ari.png')
plt.show()

fig = plt.figure(figsize = (10, 5))
x = ['PCA','ICA','RP','ETC']
plt.bar(x,[pca_silhouette_score_value,ica_silhouette_score_value,rp_silhouette_score_value,etc_silhouette_score_value])
plt.title("EM Silhouette Score (cancer)")
plt.xlabel("Algorithms")
plt.ylabel("Silhouette")
plt.savefig('Cancer_EM_all/silscore.png')
plt.show()