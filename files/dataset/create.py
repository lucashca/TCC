




arq = open('amostras.txt','r')
arqout = open('said.txt','w')

line = []
texto = arq.readlines()
cont = 0
for linha in texto :
    linha = linha.replace(' ',',')
    line.append(linha)
  
        
arqout.writelines(line)
arqout.close()
arq.close()

