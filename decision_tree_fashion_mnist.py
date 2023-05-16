from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import utils_fastion_mnist

module_name = 'Decision Tree'

((x_train, y_train), (x_test, y_test)) = utils_fastion_mnist.load_fashion_mnist_data_reshape()

pca = PCA(n_components=340)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

print("---------  run " + module_name + "  -------------")
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(x_train_pca, y_train)


# Decision Tree report and analysis
y_pred_clf = clf.predict(x_test_pca)

utils_fastion_mnist.generate_report(module_name, y_pred_clf, y_test)

