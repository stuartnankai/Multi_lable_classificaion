import pandas as pd
from collections import defaultdict


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def generate_csv(dataset, target):
    gender_temp = []
    temp_list = dataset[target].tolist()
    all_temp = []
    list_count_target = []
    temp_dict = {}
    dict_male = defaultdict(int)
    dict_female = defaultdict(int)
    dict_other = defaultdict(int)
    for n in range(len(dataset.index)):
        temp_length = len(dataset[target].loc[n].split(","))
        for m in range(temp_length):
            gender_temp.append(dataset['gender'].loc[n])

    for i in range(len(temp_list)):
        temp = temp_list[i].split(',')
        for j in range(len(temp)):
            all_temp.append(temp[j])
    print("This is : ", all_temp)
    list_target = list(set(all_temp))

    for i in range(len(list_target)):
        list_count_target.append(all_temp.count(list_target[i]))
        temp_target = list_target[i]
        list_position = duplicates(all_temp, temp_target)
        for j in list_position:
            if gender_temp[j] == "Male":
                dict_male[temp_target] += 1
            if gender_temp[j] == "Female":
                dict_female[temp_target] += 1
            if gender_temp[j] == "Other":
                dict_other[temp_target] += 1
        if temp_target not in dict_male:
            dict_male[temp_target] = 0
        if temp_target not in dict_female:
            dict_female[temp_target] = 0
        if temp_target not in dict_other:
            dict_other[temp_target] = 0

    temp_dict[target] = list_target
    temp_dict['count'] = list_count_target
    temp_dict['male'] = list(dict_male.values())
    temp_dict['female'] = list(dict_female.values())
    temp_dict['other'] = list(dict_other.values())

    print("This is dict: ", temp_dict)
    temp_df = pd.DataFrame(temp_dict, columns=[target, 'count', 'male', 'female', 'other'])
    filename = 'output_' + target
    temp_df.to_csv(filename, encoding='utf-8', index=False)


# Read csv and clean the null as No record
df = pd.read_csv('results-20180117-140237.csv') # load the csv file from bigquery
df = df.where((pd.notnull(df)), "No record")
target_list = ['Nations', 'Grade']

# for i in target_list:
    # generate_csv(df, i)
