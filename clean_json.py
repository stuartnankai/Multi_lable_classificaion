import json
import plotly.plotly as py
from plotly.graph_objs import *

# py.sign_in('sunjiannankai', 'r8kdW8nbxiw5HJeCehBj')

# py.sign_in('JianSun', 'AmAEUGYZCUR2D1dxFCZk')
py.sign_in('eddyrain','nzUXim14zjLU5cwWKtC0')
#
data = json.load(open('results-20180115-164050.json'))

# print("This is : ", data[1])
for i in range(len(data)):
    print("This is : ", data[i])
    with open('test.json', 'a') as outfile:
        json.dump(data[i], outfile)
        outfile.write('\n')
        # outfile.write(data[i])

