import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

print("Zadanie 1")
dates = pd.date_range("20200301", periods=5)
df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=list('ABC'))
print(df)

print("\nZadanie 2")
ids = np.arange(0, 21)
pandas_load = pd.DataFrame(np.random.randint(0, 20, size=(21, 3)), index=ids, columns=list('ABC'))
print(pandas_load)
print("\nTrzy pierwsze wiersze:")
print(pandas_load[1:4])  # albo pandas_load.head(3)
print("\nTrzy ostatnie wiersze:")
print(pandas_load[18:21])  # albo pandas_load.tail(3)
print("\nIndeksy:")
print(pandas_load.index)
print("\nKolumny:")
print(pandas_load.columns)
print("\nSame wartości:")
print(pandas_load.values)
print("\nWyświetlanie losowych wartości:")
print(pandas_load.loc[np.random.randint(1, 21, size=(5,))])
print("\nWyświetlanie tylko kolumny A:")
print(pandas_load['A'].values)
print("\nWyświetlanie tylko kolumny A i B:")
print(pandas_load.loc['1':'20', ['A', 'B']].values)
print("\nTrzy pierwsze wiersze i kolumny A i B:")
print(pandas_load.iloc[0:3, [0, 1]])
print("\nWiersz piąty:")
print(pandas_load.iloc[5])
print("\nWiersz 0, 5, 6, 7 i kolumny 1 i 2:")
print(pandas_load.iloc[[0, 5, 6, 7], [0, 2]])

print("\n\nZadanie 3")
print("\nWyświetl podstawowe statystyki")
print(pandas_load.describe())

print("\nKtóre dane są większe od zera")
print(pandas_load > 0)
# lub
print(pandas_load.gt(0))

print("\nWyświetl tylko wartości które są większe od 0")
# Aby sprawdzić jeden rząd, należało by jeszcze dodać najpierw pandas_load = pandas_load.loc[[<rząd>]]
print(pandas_load.loc[:, pandas_load.gt(0).all()])

print("\nWybierz z kolumny 'A' tylko dane większe od 0")
pandas_load_mask = pandas_load['A'].values > 0
print(pandas_load['A'][pandas_load_mask])

print("\nŚrednia w kolumnach")
print(pandas_load.mean())  # pandas_load.mean(<nazwa_kolumny>) by obliczyć średnią jednej kolumny

print("\nŚrednia w wierszach")
print(pandas_load.mean(axis=1))

print("\n\nZadanie 4")
print("Stwórz dwie tablice i je łacze:")
df3 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df4 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
df5 = pd.concat([df3, df4])
print(df5)

print("\nTransponuje nową tablice:")
print(df5.transpose())

# Sortowanie
print("\n\nZadanie 5")
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ['a', 'b', 'a', 'b', 'b']}, index=np.arange(5))
df.index.name = 'id'
print(df)
print("\nSortuje po 'id':")
print(df.sort_values(by='id', ascending=False))

print("\nSortuje po 'y':")
print(df.sort_values(by='y', ascending=True))

# Grupowanie danych
slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'],
           'Pound':
               [10, 15, 50, 40, 5], 'Profit': [20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print("\n\n")
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day', 'Fruit']).sum())


# Wypełnianie danych
df = pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A', 'B', 'C'])
df.index.name = 'id'
print("\n\n")
print(df)
print("\nUstawianie wszystkich wartości w kolumnie 'B' na 1")
df['B'] = 1
print(df)
print("\nIterowanie po: id = 1 oraz po indeksach kolumny zaczynając od 0 i ustawianie wartości na 10")
df.iloc[1, 2] = 10
print(df)
print("\nWszystkie wartości, które są mniejsze od 0, mają swój znak zmieniony na przeciwny")
df[df < 0] = -df
print(df)


# Uzupełnianie danych
print("\n\nWstawianie w zerowe i trzecie miejsce drugiej kolumny wartość NaN (Not-a-Number")
df.iloc[[0, 3], 1] = np.nan
print(df)
print("\nW miejsce NaN są wstawiane zera. Operacja jest wykonywana w miejscu (nie jest tworzona kopia tabeli")
df.fillna(0, inplace=True)
print(df)
print("\nPonownie w zerowe i trzecie miejsce drugiej kolumny są wstawiane wartości NaN. Później w ich miejsce jest "
      "wstawiana wartość -9999")
df.iloc[[0, 3], 1] = np.nan
df = df.replace(to_replace=np.nan, value=-9999)
print(df)
print("\nPonownie w zerowe i trzecie miejsce drugiej kolumny są wstawiane wartości NaN. Następnie jest sprawdzane"
      "które wartości są zerowe. Zdaje się, że wartość NaN jest traktowana jako zero")
df.iloc[[0, 3], 1] = np.nan
print(pd.isnull(df))


# Zadania
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
print("\n\n", df)

print("\n\nZadanie 1")
print(df.groupby('y').mean())

print("\nZadanie 2")
print(df.value_counts())

print("\nZadanie 3")
numpy_load = np.loadtxt("autos.csv", delimiter=",", dtype="str")
print(numpy_load)
pandas_load = pd.read_csv('autos.csv', index_col=0)
print("\n", pandas_load)

print("\nZadanie 4")
print(pandas_load.groupby('make').mean())

print("\nZadanie 5")
print(pandas_load.groupby(['make', 'fuel-type'])['fuel-type'].count())

print("\nZadanie 6")
wynik1 = np.polyfit(pandas_load["city-mpg"].values, pandas_load["length"].values, 1) # polyfit(x, y, <stopień-wielomianu>)
wynik2 = np.polyfit(pandas_load["city-mpg"].values, pandas_load["length"].values, 1)
pv1 = np.polyval(wynik1, pandas_load['city-mpg'])
pv2 = np.polyval(wynik2, pandas_load['city-mpg'])
print("Wielomian 1 stopnia:", wynik1)
print("Wielomian 2 stopnia:", wynik2)

print("\nZadanie 7")
print(stats.pearsonr(pandas_load['city-mpg'], pandas_load['length']))

print("\nZadanie 8")
fig, az = plt.subplots()
az.scatter(pandas_load['city-mpg'], pandas_load['length'])
az.plot(pandas_load['city-mpg'], pv1)
plt.show()

print("\nzadanie 9")

#estymator1 = stats.gaussian_kde(pandas_load['length'])  # Estymator funkcji gestości dla długości
# estymator2 = stats.gaussian_kde(pandas_load['width'])

est = stats.gaussian_kde(pandas_load['length'])
f = np.linspace(pandas_load["length"].min(), pandas_load["length"].max(), num=205)
fig, ax = plt.subplots()
# ax.bar(f, est(f), width=5, edgecolor="purple", linewidth=1)
# ax.plot(f, est(f), label='Gaussian_kde for \'length\'')
plt.show()
# fw = np.linspace(pandas_load['width'].min(), pandas_load['width'].max(), num=205)
# fig, ax = plt.subplots(1, 2)

plt.show()


est2 = stats.gaussian_kde(pandas_load['width'])
# Zad 11

'''
x = pandas_load['length']
y = pandas_load['width']
xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
cfset = ax.contourf(xx, yy, f, cmap='RdYlBu_r')
cset = ax.contour(xx, yy, f, colors='k')
ax.set_xlabel('length')
ax.set_ylabel('width')
plt.show()
'''