from lib import utils
from sklearn import svm
"""
mydata = utils.TrainingData("./../testing/class_test")


mydata.collect_files_from_dir(file_ending="txt")
mydata.add_categories()
mydata.add_text(lang="de")
mydata.to_sentences()
mydata.generate_sequences(scope="tokens", num_words=1000)
mydata.padding_sequences(maxlen=1000)
mydata.to_categorical_trainingdata(scope="sequences")
mydata.split_training_data(ratio=0.2)


clf = svm.SVC()
clf.fit(mydata.x_train, mydata.y_train)
print(clf.score(mydata.x_test, mydata.y_test))
"""

mydata = utils.TrainingData("./../testing/sequence_test")

mydata.collect_files_from_dir("tsv")
mydata.load_sequential_data()
mydata.add_sequential_context(3)
print(mydata.corpus_df["sequence_training"][0])