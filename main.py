import numpy as np
import pandas as pd

print("Zadanie 1")
dates = pd.date_range("20200301", periods=5)
df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=list('ABC'))
print(df)

print("\nZadanie 2")
ids = np.arange(0, 21)
df2 = pd.DataFrame(np.random.randint(0, 20, size=(21, 3)), index=ids, columns=list('ABC'))
print(df2)
print("\nTrzy pierwsze wiersze:")
print(df2[1:4])  # albo df2.head(3)
print("\nTrzy ostatnie wiersze:")
print(df2[18:21])  # albo df2.tail(3)
print("\nIndeksy:")
print(df2.index)
print("\nKolumny:")
print(df2.columns)
print("\nSame wartości:")
print(df2.values)
print("\nWyświetlanie losowych wartości:")
print(df2.loc[np.random.randint(1, 21, size=(5,))])
print("\nWyświetlanie tylko kolumny A:")
print(df2['A'].values)
print("\nWyświetlanie tylko kolumny A i B:")
print(df2.loc['1':'20', ['A', 'B']].values)
print("\nTrzy pierwsze wiersze i kolumny A i B:")
print(df2.iloc[0:3, [0, 1]])
print("\nWiersz piąty:")
print(df2.iloc[5])
print("\nWiersz 0, 5, 6, 7 i kolumny 1 i 2:")
print(df2.iloc[[0, 5, 6, 7], [0, 2]])

print("\n\nZadanie 3")
print("\nWyświetl podstawowe statystyki")
print(df2.describe())

print("\nKtóre dane są większe od zera")
print(df2 > 0)
# lub
print(df2.gt(0))

print("\nWyświetl tylko wartości które są większe od 0")
# Aby sprawdzić jeden rząd, należało by jeszcze dodać najpierw df2 = df2.loc[[<rząd>]]
print(df2.loc[:, df2.gt(0).all()])

print("\nWybierz z kolumny 'A' tylko dane większe od 0")
df2_mask = df2['A'].values > 0
print(df2['A'][df2_mask])

print("\nŚrednia w kolumnach")
print(df2.mean())  # df2.mean(<nazwa_kolumny>) by obliczyć średnią jednej kolumny

print("\nŚrednia w wierszach")
print(df2.mean(axis=1))

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
pandas_load = pd.read_csv('autos.csv')
print("\n", pandas_load)
