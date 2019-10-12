import csv
import random





header = [
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
        
import sys 
arg = sys.argv[1]
rowNumber = 100,

if(arg):
    rowNumber = int(arg)


colunNumber = len(header)

__dir = './datasets/'

def generateRow():
    global colunNumber
    row = []
    '''
    row.append("Lucas")
    row.append("Bahia")
    row.append("Cruz das Almas")
    row.append("-15678162563869")
    row.append("16871454563869")
    '''
    for i in range(colunNumber):
        val = round(random.random()*1,(2))
        row.append(val)
    return row


def createCsv():
    global rowNumber,header
    csv_file = open(__dir+'solo.csv','w')
    csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
   
    csv_file.writerow(header)

    for i in range(rowNumber):
        row = generateRow()
        csv_file.writerow(row)



print('Createing data set...')
print('Rows: ',rowNumber)
print('Coluns: ',len(header))

createCsv()
print('Done')