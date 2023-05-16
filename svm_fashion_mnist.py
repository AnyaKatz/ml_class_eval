from sklearn.svm import SVC
import utils_fastion_mnist

module_name = 'SVM'


((x_train, y_train), (x_test, y_test)) = utils_fastion_mnist.load_fashion_mnist_data_reshape()

print("---------  run " + module_name + "  -------------")
clf = SVC(C=1, kernel='linear', gamma="auto")
clf.fit(x_train, y_train)

y_pred_clf = clf.predict(x_test)

utils_fastion_mnist.generate_report(module_name, y_pred_clf, y_test)

