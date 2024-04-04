import pandas as pd 
def generate_category_list(category):
    df = pd.read_csv('/weka/scratch/tshu2/hshi33/habitat/habitat-lab/hssd-hab/objects/hssd-receptacles.csv')
    df.columns = ["object_id", "name"]
    file_path = '/weka/scratch/tshu2/hshi33/habitat/example_scenario/object_list/{}.txt'.format(category)
    with open(file_path, "w+") as file:
        for i in range(0, df.shape[0]):
            if df.loc[i, "name"] == category:
                file.write("- " + "\"" + df.loc[i, "object_id"] + "\"" + "\n")
category_list = []
df = pd.read_csv('/weka/scratch/tshu2/hshi33/habitat/habitat-lab/hssd-hab/objects/hssd-receptacles.csv')
df.columns = ["object_id", "name"]
for i in range(0, df.shape[0]):
    if df.loc[i, "name"] not in category_list:
        category_list.append(df.loc[i, "name"])
for category in category_list:
    generate_category_list(category)