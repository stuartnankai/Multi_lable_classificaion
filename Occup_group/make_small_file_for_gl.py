import pandas as pd
import ast
import numpy as np

df = pd.read_csv('Occup_group/cleaned_18000.csv', sep=',')
tempdf = pd.DataFrame(columns=['title', 'id'], data=None)
tempdf1 = pd.DataFrame(columns=['title', 'id'], data=None)
df1 = df.iloc[:150, :]
df3 = df.iloc[200:250, :]


def get_id(group_id):
    temp = ast.literal_eval(group_id)
    temp = [n for n in temp]
    # myarray = np.asarray(temp)
    return str(temp[0])

def get_title(title):
    print("This is : ", title)
    temp  = title.split()
    temptitle = ''
    for i in range (len(temp)):
        temptitle +=str(temp[i])
        if i !=len(temp)-1:
            temptitle += '-'
    return temptitle

# #
df2 = df1[['title', 'group_id']]
df4 = df3[['title', 'group_id']]
df4 = df4.reset_index()
#
for i in range(len(df2.index)):
    get_title(df2.at[i, 'title'])
    tempdf.at[i, 'title'] = get_title(df2.at[i, 'title'])
    tempdf.at[i, 'id'] = get_id(df2.at[i, 'group_id'])

for j in range(len(df4.index)):
    tempdf1.at[j, 'title'] = get_title(df4.at[j, 'title'])
    tempdf1.at[j, 'id'] = get_id(df4.at[j, 'group_id'])

tempdf.to_csv('occupation.data.csv', index=False,header=False)
tempdf1.to_csv('occupation.test.csv', index=False,header=False)

# df = pd.read_csv('Untitled_20180215.csv',sep=',')
#
# df = df['name'].tolist()
#
# templist = []
#
# for i in df:
#     templist.append(str(i))
#
# print("This is : ", templist)