import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# Zadatak 6.5.1

#1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_n, y_train)

knn_y_train_p = knn.predict(X_train_n)
knn_y_test_p = knn.predict(X_test_n)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, knn_y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, knn_y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Tocnost logisticke regresije: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# granica odluke pomocu KNN
plot_decision_regions(X_train_n, y_train, classifier=knn)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Tocnost pomocu KNN: " + "{:0.3f}".format((accuracy_score(y_train, knn_y_train_p))))
plt.tight_layout()
plt.show()

#Vidimo bolje rezultate s KNN algoritmom
#2)
KNN_Model1 = KNeighborsClassifier(n_neighbors=1)
KNN_Model1.fit(X_train_n,y_train)
knn_y_test_p = KNN_Model1.predict(X_test_n)
knn_y_train_p = KNN_Model1.predict(X_train_n)

print("KNN za K=1: ")
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test,knn_y_test_p))))
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train,knn_y_train_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_Model1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("K=1")
plt.tight_layout()
plt.show()

KNN_Model100 = KNeighborsClassifier(n_neighbors=100)
KNN_Model100.fit(X_train_n,y_train)
knn_y_test_p = KNN_Model100.predict(X_test_n)
knn_y_train_p = KNN_Model1.predict(X_train_n)

print("KNN za K=100: ")
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test,knn_y_test_p))))
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train,knn_y_train_p))))



plot_decision_regions(X_train_n, y_train, classifier=KNN_Model100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("K=100")
plt.tight_layout()
plt.show()

#Veca je tocnost kod K=1, ali manja od K=5

# Zadatak 6.5.2
pipe = Pipeline(
    [
        ('classifier', KNeighborsClassifier(metric='euclidean'))
    ]
)
param_grid = {'classifier__n_neighbors': range(1,100)}
gscv = GridSearchCV(pipe,param_grid=param_grid,scoring='accuracy')
gscv.fit(X_train_n,y_train)
print(f"Optimal K-value: {gscv.best_params_}")

print("----------------------------------------------------------------------------------")
# Zadatak 6.5.3.
# Gamma =1, c=0.1, kernel=rbf
SVM_model = svm.SVC(kernel='rbf', gamma = 1 , C=0.1 )
SVM_model.fit( X_train_n , y_train )
y_test_p=SVM_model.predict(X_test_n)
print("Tocnost za test skup Gamma =1, c=0.1, kernel=rbf: " + "{:0.3f}".format((accuracy_score(y_test,y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Gamma=1, C=0.1")
plt.tight_layout()


# Gamma =5, c=0.1, kernel=rbf
SVM_model = svm.SVC(kernel='rbf', gamma = 5 , C=0.1 )
SVM_model.fit( X_train_n , y_train )
y_test_p=SVM_model.predict(X_test_n)
print("Tocnost za Gamma =5, c=0.1, kernel=rbf: " + "{:0.3f}".format((accuracy_score(y_test,y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Gamma=5, C=0.1")
plt.tight_layout()


# Gamma =1, c=1, kernel=rbf
SVM_model = svm.SVC(kernel='rbf', gamma = 1 , C=1 )
SVM_model.fit( X_train_n , y_train )
y_test_p=SVM_model.predict(X_test_n)
print("Tocnost za Gamma =1, c=1, kernel=rbf: " + "{:0.3f}".format((accuracy_score(y_test,y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Gamma=1, C=1")
plt.tight_layout()


# Gamma =1, c=5, kernel=rbf
SVM_model = svm.SVC(kernel='rbf', gamma = 1 , C=5 )
SVM_model.fit( X_train_n , y_train )
y_test_p=SVM_model.predict(X_test_n)
print("Tocnost za Gamma =1, c=5, kernel=rbf: " + "{:0.3f}".format((accuracy_score(y_test,y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Gamma=1, C=5")
plt.tight_layout()


# Gamma =1, c=5, kernel=linear
SVM_model = svm.SVC(kernel='linear' , C=5 )
SVM_model.fit( X_train_n , y_train )
y_test_p=SVM_model.predict(X_test_n)
print(" c=1, kernel=linear: " + "{:0.3f}".format((accuracy_score(y_test,y_test_p))))
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("C=1")
plt.tight_layout()
plt.show()

#Vidimo da povećanjem gamme, tj. smanjenjem raspršenja smo dobili smanjenu točnost, te sa linearnom jezgrom također imamo manju točnost

#Zadatak 6.5.4

from sklearn.svm import SVC

param_grid_svc = {
    'svc__C': [0.1,1,10,100],
    'svc__gamma':[10,1,0.1,0.01],
    'svc__kernel':['rbf']}


svm_pipe= make_pipeline(svm.SVC(random_state=10))



svm_gscv = GridSearchCV( svm_pipe , param_grid_svc , cv = 5 , scoring = 'accuracy')
svm_gscv.fit( X_train_n , y_train )
print ( svm_gscv.best_params_ )
print ( svm_gscv.best_score_ )
print ( svm_gscv.cv_results_ )