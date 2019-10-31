import csv
import random as r

csv_file = open("sela.csv", "w")
c = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
faixa = 1000
for i in range(1000):
    x = r.randint(-faixa,faixa) + r.random()
    y = r.randint(-faixa,faixa) + r.random()
    z = x**2 - y**2
    c.writerow([x,y,z])

   