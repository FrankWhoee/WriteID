import os
import re
import random

files = os.listdir("data")
total = 0
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
authors = {}
for f in files:
    with open("data/" + f, 'r') as file:
        data = file.read()
        total += len(re.findall(r'\w+', data))

    a = f.split("-")[0]

    print(a)
    if a not in authors:
        authors[a] = {
            'index': random.choice(indices)
        }
        indices.remove(authors[a]['index'])
    print(authors[a])
for f in files:
    a = f.split("-")[0]
    index = authors[a]['index']
    os.rename("data/" + f, "data/" + str(index) + "-" + str(random.randint(0,99999999999999)) + ".txt")
print(total)
