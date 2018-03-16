from lib import utils
from sklearn import svm

pipeline = [utils.TrainingData.collect_files_from_dir,
            utils.TrainingData.load_sequential_data,
            utils.TrainingData.add_sequential_context,
            utils.TrainingData.generate_sequences,
            utils.TrainingData.padding_sequences,
            utils.TrainingData.to_sequential_trainingdata,
            utils.TrainingData.split_training_data]






parameters = {"scope": ["sequence_training"],
            "file_ending": ["tsv"],
            "num_words": [100],
            "windowsize": [3,4,5],
            "ratio": [0.1],
            "folder": ["/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/software/TfDPy/testing/sequence_test"],
            "maxlen": [3,4,5]}

classifier = [svm.SVC]

values = utils.td_paramsearch(pipeline, parameters, classifier)

print(values)



scope = ["sequence_training"]
file_ending = ["tsv"]
num_words = [10, 10000]
windowsize = [3, 5, 8]
ratio = [0.1]
folder = ["/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/software/TfDPy/testing/sequence_test"]
for scope_g in scope:
    for num_words_g in num_words:
        for file_ending_g in file_ending:
            for folder_g in folder:
                for windowsize_g in windowsize:
                    for ratio_g in ratio:
                        mydata = utils.TrainingData(folder=folder_g)
                        process = [
                        mydata.collect_files_from_dir(file_ending=file_ending_g),
                        mydata.load_sequential_data(),
                        mydata.add_sequential_context(windowsize=windowsize_g),
                        mydata.generate_sequences(scope=scope_g, num_words=num_words_g),
                        mydata.padding_sequences(maxlen=num_words_g),
                        mydata.to_sequential_trainingdata(),
                        mydata.split_training_data(ratio=ratio_g)]

                        print(process)



                        #clf = svm.SVC()
                        #clf.fit(mydata.x_train, mydata.y_train)
                        #print(clf.score(mydata.x_test, mydata.y_test))



