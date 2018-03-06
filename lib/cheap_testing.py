from lib import utils
import os
test1 = utils.collect_files_from_dir("/home/leo/Documents/schemaliteratur_DS/schemakorpus" ,".xml")




test1["test"] = test1["filepath"].apply(os.path.split)
test1["test"] = test1[0]
print(test1["test"])