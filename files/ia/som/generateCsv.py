import csv
import random

import sys 
arg = sys.argv[1:]


header = []



titles = [
        'Nitrogênio Foliar (g/kg)',
        'Fósforo(P) (mg/dm³)',
        'Potássio(K) (cmol c/dm³)',
        'Boro(B) (mg/dm³)',
        'Cálcio(Ca) (cmol c/dm³)',
        'Magnésio(Mg) (cmol c/dm³)',
        'Alumínio(Al) (cmol c/dm³)',
        'Sódio(Na) (cmol c/dm³)',
        'H+Al (cmol c/dm³)',
        'Zinco(Zn) (mg/dm³)',
        'Cobre(Cu) (mg/dm³)',
        'Manganês(Mn) (mg/dm³)',
        'PRNT do calcário', 
        ]
        

hasIndex = False
rowNumber = 100
groups = len(titles)

try:
        if(arg[0]):
                rowNumber = int(arg[0])
        if(arg[1]):
                columns = int(arg[1])
        if(arg[2]):
                groups = int(arg[2])
        if(arg[3]):
                hasIndex = True
                
        


except:
        print("python generateCsv.py rowLength colunms groups hasIndex")





def setHeader(v):
        global titles
        h = []
        if(hasIndex):
                h.append('index')
        for i in range(v):
                h.append(titles[i])
        
        return h


header = setHeader(columns)

colunNumber = len(header)

__dir = './datasets/'

def generateRow(j):
    global colunNumber,hasIndex,groups
    row = []
    for i in range(colunNumber):
        a = random.randint(1,100)*0.01
        if(hasIndex and i == 0):
                row.append(1)
                continue

        #val = round(1000 + a + b,2)         
        #if(j%2==0):
        #       val = round(j*5 + a + b,2)         
        val = round(j*20 + a,2)         
  
        if(groups == 0):
                val = round(random.randint(1,100)*0.01 + random.randint(0,5),2)
                
        row.append(val)
    return row


def createCsv():
    global rowNumber,header,groups
    csv_file = open(__dir+'solo.csv','w')
    csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    h =    header
    csv_file.writerow(h)
    j = 0
    for i in range(rowNumber):
        j = j + 1
        row = generateRow(j)
        csv_file.writerow(row)
        if(j>= groups):j = 0


print('Createing data set...')
print('Rows: ',rowNumber)
print('Coluns: ',len(header))

createCsv()
print('Done')