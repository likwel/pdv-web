#1-Algorithme recursif
def U(n):
	if n==0:
		suite=4.0
	else:
		suite=3*U(n-1)-1
	return suite
Ucent=U(11)
print(Ucent)

#2- Les 10 premiers termes de U(n)
res=[U(0),U(1),U(2),U(3),U(4),U(5),U(6),U(7),U(8),U(9)]
print("Resultat : ",res)

#3- Tri par ordre decroissante

def permuter(tab,i,j):
	tmp=tab[i]
	tab[i]=tab[j]
	tab[j]=tmp

def tri_insert(tab):
	for i in range(2,len(tab)): 				#c1,n
		x=tab[i]								#c2,n-1
		j=i-1									#c3,n-1
		while j>=0 and tab[j]<x: #ou tab[j]<x	#c4,somme(j=2, n) de tj
			tab[j+1]=tab[j]						#c5,somme(j=2, n) de tj-1
			j=j-1								#c6,somme(j=2, n) de tj-1
		tab[j+1]=x								#c7,n-1
	print (tab)

tri_insert(res)

#4- Calcul moyenne
def moyenne(res):
	somme=0
	for i in range(len(res)):						#c8,n
		somme=somme+res[i]							#c9,n-1
	moyenne=somme/len(res)							
	print("Moyenne = ", moyenne)

tab=[1,2,3,4,5,6,7,8,9,10]

moyenne(res)

#5- Temps d'execution de l'algorithme (cout et nombre d'execution)
#T(n)=somme(cj*nbj)=c1*n+c2*(n-1)+...+c9*(n-1)

#6- Complexites
# meilleur cas (tj=1)
# pire cas (tj=j)
# en moyenne (tj=j/2)
#plus ordre de grandeur + classe (quadratique, lineaire, exp,log,....)
# 0(???)

def factoriel(n):
	if n==0:
		res=1
	else:
		res=n*factoriel(n-1)
	return res

print("factoriel de n est : ", factoriel(0))