import matplotlib.pyplot as plt
import pandas as pd

def somme(tab):
    somme=0
    for i in range(len(tab)):
        somme=somme+tab[i]
    print(somme)
    #return somme

def temporelle(fen,t,y):
    res=[]
    #for i in range(t-fen,t+fen):
    indice=(1/(2*fen)+1)*somme(y)
        #res.append(indice)
    print(indice)
    return indice
data=pd.read_csv('airline_passengers.csv')

y=list(data['Thousands of Passengers'])
somme(y)
y_chap=temporelle(1,10,y)
#yss=somme(1,y)
print(y_chap)

plt.plot(y, label='Reelle')
plt.plot(y_chap,label='Pred')
plt.legend()
plt.show()