# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:52:28 2023

@author: cschi
"""
###############################################
#         Exercice 1                          #
###############################################


# Importation package nécessaire
import pandas as pds


import numpy as np

from scipy import stats
#création du jeu de données observé


obs= np.array([[592,119,849,504,36],[544,97,677,451,14]])

print(obs)

#création du jeu de données théorique

chi2, p, dof, thq = stats.chi2_contingency(obs, correction=False)
print(thq)
#thq = le tableau de contingence théorique


# a test du chi 2

# Vérification des condition d'application

#test de Cochran

def coch(thq):
    Coch=0
    for i in range(thq.shape[0]): 
        for j in range(thq.shape[1]):
            if thq[i,j]<5:
                Coch+=1
            percent = Coch/(thq.shape[0]*thq.shape[1])*100
    return percent 
percent = coch(thq)
print(coch(thq))
print('il y a ', percent,'% des effectifs théoriques strictement inférieur à 5')





#  Statistique de décision du chi^2

print('La statistique de décision vaut',chi2,'. Sous H0 elle suit une loi du Chi2 à', dof, 'degrés de liberté. Ainsi la p-value est égale à',p)

#p_value < alpha , H_1

# c G-test

# Statistique de décision du G-test
g, p, dof, thq = stats.chi2_contingency(obs,lambda_ ="log-likelihood", correction=False)
print(thq)


print("La statistique de décision vaut {:.2f}. Sous H0 elle suit une loi du Chi2 à {} degrés de liberté. Ainsi la p-value est égale à {:.4f}.".format(g, dof, p))

###############################################
#         Exercice 2                          #
###############################################


# Importation des données

import pandas as pds

man = pds.read_csv("C:/Users/cschi/Documents/Python Scripts/Man.csv",delimiter=";",decimal=",")


man

print(man)

print(np.mean(man),man.isna().sum())

import numpy as np

from scipy import stats
################################################
#          Statistiques descriptive            #
################################################


#Pour Manele 

median = np.median(man["Manele"].dropna())
quartile1 = np.percentile(man["Manele"].dropna(),25)
quartile3 = np.percentile(man["Manele"].dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )


print(stats.describe(man["Manele"].dropna()))


#Pour Manala

median = np.median(man["Manala"].dropna())
quartile1 = np.percentile(man["Manala"].dropna(),25)
quartile3 = np.percentile(man["Manala"].dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )

print(stats.describe(man["Manala"].dropna()))


# Graphique

import matplotlib.pyplot as plt

# Données à utiliser dans l'histogramme
donneesMe = man["Manele"].dropna()
donneesMa = man["Manala"].dropna()


# Création de l'histogramme
plt.hist(donneesMa, bins=10, alpha=0.5, label='Manala')
plt.hist(donneesMe, bins=10, alpha=0.5, label='Manele')

# Ajout des titres et des légendes
plt.xlabel('Poids(g)')
plt.ylabel('Fréquence')
plt.title('Histogramme du poids des Manala et Manele')
plt.legend()

# Affichage de l'histogramme
plt.show()

# Creation boite à moustache

poids = [donneesMa, donneesMe]

plt.boxplot(poids, labels=['Manala', 'Manele'])
plt.title('Poids des Manalas et Maneles')
plt.xlabel('Catégories')
plt.ylabel('Poids')
plt.show()







################################################
#          Traitement statistique              #
################################################

#5 Importation des données

import numpy as np
from scipy.stats import ttest_ind


donneesMe = man["Manele"].dropna()
donneesMa = man["Manala"].dropna()

#6 Conditions d'application : Normalité


#test de shapiro wilk
from scipy import stats

stats.shapiro(donneesMe.dropna())
stats.shapiro(donneesMa.dropna())

#QQ-plot

import statsmodels.api as sm
import numpy as np
sm.qqplot(donneesMa,loc=np.mean(donneesMa),scale=np.std(donneesMa), line='45')
sm.qqplot(donneesMe,loc=np.mean(donneesMe),scale=np.std(donneesMe), line='45')


#7 Condition d'application : variance égale


def f_test(x,y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x,ddof=1)/np.var(y,ddof=1)
    dfn= x.size - 1
    dfd = y.size -1
    p = 1-stats.f.cdf(f,dfn,dfd)  
    return f,p,dfn,dfd
f_test(donneesMe, donneesMa)


f,p,dfn,dfd = f_test(donneesMe, donneesMa)
print("La statistique de décision vaut",f,"Sous H_0 elle suit une lois de Fisher à",dfn,"et",dfd,"ddl.la p_value est égale à",p)



# 8 Statistique de décison du test de Student

stats.ttest_ind(donneesMe,donneesMa)
stats.ttest_ind(donneesMe,donneesMa,nan_policy='omit')
t,p=stats.ttest_ind(donneesMe,donneesMa,nan_policy='omit')
print("La statistique de décision vaut", t,". Sous H0, elle suit une loi de Student à ", len(donneesMe)+len(donneesMa)-2,". La p-value est égale à",p)




###############################################
#         Exercice 3                          #
###############################################




#Importation des données

import pandas as pds

timbres = pds.read_csv("C:/Users/cschi/Documents/Python Scripts/timbres.csv",delimiter=";")

timbres

print(timbres)
# Afficher les premières lignes des données
print(timbres.head())

tbrAl=timbres["epaisseur"][timbres["pays"]=="Allemagne"].dropna()
tbrAu=timbres["epaisseur"][timbres["pays"]=="Autriche"].dropna()
tbrBe=timbres["epaisseur"][timbres["pays"]=="Belgique"].dropna()
tbrFr=timbres["epaisseur"][timbres["pays"]=="France"].dropna()

################################################
#          Statistiques descriptive            #
################################################



# Statistique de l'Allemagne

import numpy as np

from scipy import stats
median = np.median(tbrAl)
quartile1 = np.percentile(tbrAl,25)
quartile3 = np.percentile(tbrAl.dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )

print(stats.describe(tbrAl))



# Statistique de l'Autriche

median = np.median(tbrAu)
quartile1 = np.percentile(tbrAu,25)
quartile3 = np.percentile(tbrAu.dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )

print(stats.describe(tbrAu))

# Statistique de la Belgique

median = np.median(tbrBe)
quartile1 = np.percentile(tbrBe,25)
quartile3 = np.percentile(tbrBe.dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )

print(stats.describe(tbrBe))


# Statistique de la France

median = np.median(tbrFr)
quartile1 = np.percentile(tbrFr,25)
quartile3 = np.percentile(tbrFr.dropna(),75)
print('Mediane=', median ,',quartile 1=', quartile1 ,',quartile 3 =', quartile3 )

print(stats.describe(tbrFr))

print(stats.describe(timbres["epaisseur"]))
# Calculer les statistiques descriptives par pays
stats_descriptives = timbres.groupby('pays')['epaisseur'].describe()
print(stats_descriptives)
#renvoie tout les stat descriptive
timbres.epaisseur.describe()


# Graphique


import matplotlib.pyplot as plt

# Données à utiliser dans la boîte à moustaches
donnees = timbres["epaisseur"]
donnes = timbres["pays"]
# Créer la figure et l'axe

tbrAl.describe()
tbrAu.describe()
tbrBe.describe()
tbrFr.describe()
# Dessiner la boîte à moustaches
Pays = [tbrAl, tbrAu, tbrBe, tbrFr]

plt.boxplot(Pays, labels=['Allemagne', 'Autriche', 'Belgique','France'])
plt.title('Epaisseur du timbres en fonction du pays')
plt.xlabel('Pays')
plt.ylabel('Epaisseur (micromètre)')
plt.show()

# Titres et étiquettes des axes
ax.set_title('Boîte à moustaches')
ax.set_ylabel('epaisseur')

# Afficher la figure
plt.show()



# Création de l'histogramme
plt.hist(tbrAl, bins=10, alpha=0.5, label='Allemagne')
plt.hist(tbrAu, bins=10, alpha=0.5, label='Autriche')
plt.hist(tbrBe, bins=10, alpha=0.5, label='Belgique')
plt.hist(tbrFr, bins=10, alpha=0.5, label='France')

# Ajout des titres et des légendes
plt.xlabel('Epaisseur')
plt.ylabel('Fréquence')
plt.title('Histogramme de lépaisseur des timbres selon le pays')
plt.legend()

# Affichage de l'histogramme
plt.show()



################################################
#          Traitement statistique              #
################################################



pds.crosstab(timbres.epaisseur, "freq")
#Il est préférable de regarder la répartition de la variable
#épaisseur :
    
# 1 récolte des données

#Afin d’´eviter d’alourdir le code, on va créer des objets qui
#seront récurrents dans le code :
    
tbrAl=timbres["epaisseur"][timbres["pays"]=="Allemagne"].dropna()
tbrAu=timbres["epaisseur"][timbres["pays"]=="Autriche"].dropna()
tbrBe=timbres["epaisseur"][timbres["pays"]=="Belgique"].dropna()
tbrFr=timbres["epaisseur"][timbres["pays"]=="France"].dropna()



#6 Vérification des conditions du test

from scipy import stats

# a Test de shapiro-wilk

stats.shapiro(tbrAl)
stats.shapiro(tbrAu)
stats.shapiro(tbrBe)
stats.shapiro(tbrFr)

# on compare pvalue à alpha 5%
#si la p-value est supérieure au niveau alpha choisi (par exemple 0.05), 
#alors on ne doit pas rejeter l'hypothèse nulle. La valeur de la p-value
# alors obtenue ne présuppose en rien de la nature de la distribution des données.

# b QQ-plot

sm.qqplot(tbrAl,loc=np.mean(tbrAl),scale=np.std(tbrAl), line='45')
sm.qqplot(tbrAu,loc=np.mean(tbrAu),scale=np.std(tbrAu), line='45')
sm.qqplot(tbrBe,loc=np.mean(tbrBe),scale=np.std(tbrBe), line='45')
sm.qqplot(tbrFr,loc=np.mean(tbrFr),scale=np.std(tbrFr), line='45')

# c test avec les résidus

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('epaisseur ~ pays', data=timbres).fit()
stats.shapiro(model.resid)

sm.qqplot(model.resid,loc=np.mean(model.resid),scale=np.std(model.resid ), line='45')
                                                       
# test de Bartlett

B,p=stats.bartlett(tbrAl,tbrAu,tbrBe,tbrFr)
print("La statistique de décision est égale à", B,". La p-value est",p )

#7) Statistique de décision

aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
aov_table = sm.stats.anova_lm(model, typ=2)
p_value = aov_table["PR(>F)"][0]
print('p_value=',p_value)

test_anova(aov_table)



# Partie 3 Après ANOVA


#Bonferroni

import statsmodels.stats.multicomp as mc



test = timbres.filter(items = ['epaisseur', 'pays']).dropna()

comp= mc.MultiComparison(test['epaisseur'], test['pays'])

tbl,a,b = comp.allpairtest(stats.ttest_ind,method ='bonf')

print(tbl)

#Tukey

post_hoc = comp.tukeyhsd()
print(post_hoc)

# graphiquement

post_hoc.plot_simultaneous(ylabel = 'pays',xlabel='epaisseur')


# Vérification



# Filtrer les données pour exclure le pays 'France'
timbreswfrance = timbres[timbres['pays'] != 'France']

# Ajuster le modèle en utilisant les données filtrées
model2 = ols('epaisseur ~ pays', data=timbreswfrance).fit()

stats.shapiro(model2.resid)

sm.qqplot(model2.resid,loc=np.mean(model2.resid),scale=np.std(model2.resid ), line='45')
 

B,p=stats.bartlett(tbrAl,tbrAu,tbrBe)
print("La statistique de décision est égale à", B,". La p-value est",p )


aov_table2 = sm.stats.anova_lm(model2, typ=2)
aov_table2
p_value2 = aov_table2["PR(>F)"][0]
print('p_value=',p_value2)

test_anova(aov_table2)

















