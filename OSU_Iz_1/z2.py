import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("titanic.csv")

data.columns = data.columns.str.strip()

def plot_decision_regions_pca(X, y, classifier, title="Decision Regions (PCA)"):
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Train classifier on PCA-reduced data
    classifier.fit(X_pca, y)
    
    # Setup meshgrid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Predict on meshgrid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()

#Priprema dataseta        
data.dropna()
data.drop_duplicates()
data=data[['Pclass','Embarked','Sex', 'Fare','Survived']]
X=data.drop(columns=['Survived'])
y=data['Survived']

X_categorical=X[['Sex', 'Embarked']]
X=X.drop(columns=['Sex', 'Embarked'])

encoder = OneHotEncoder(sparse_output=False)
X_categorical_n = encoder.fit_transform(X_categorical)
X_categorical_n=pd.DataFrame({'Sex':X_categorical_n[:,0], 'Embarked':X_categorical_n[:,1]})

X_n=pd.concat([X,X_categorical_n],axis=1)

from sklearn.neighbors import KNeighborsClassifier
# podijeli podatke u omjeru 80-20%
X_train_n, X_test_n, y_train, y_test = train_test_split(X_n, y, test_size = 0.2, stratify=y, random_state = 10)

sc=StandardScaler()

print(X_train_n)
X_train_s=sc.fit_transform(X_train_n)
X_test_s=sc.transform(X_test_n)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)

knn_y_train_p = knn.predict(X_train_s)
knn_y_test_p = knn.predict(X_test_s)

plot_decision_regions_pca(X_train_s, y_train, classifier=knn)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, knn_y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, knn_y_test_p))))


#Unakrsna validacija
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(
    [
        ('classifier', KNeighborsClassifier(metric='euclidean'))
    ]
)
param_grid = {'classifier__n_neighbors': range(1,100)}
gscv = GridSearchCV(pipe,param_grid=param_grid,scoring='accuracy')
gscv.fit(X_train_s,y_train)
print(f"Optimal K-value: {gscv.best_params_}")
print(gscv.best_score_)


#Treniranje s K=64
knn = KNeighborsClassifier(n_neighbors=58)
knn.fit(X_train_s, y_train)

knn_y_train_p = knn.predict(X_train_s)
knn_y_test_p = knn.predict(X_test_s)
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, knn_y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, knn_y_test_p))))