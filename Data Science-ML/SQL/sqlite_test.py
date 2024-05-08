import sqlite3
import pandas as pd

conn = sqlite3.connect("C:\\Users\\erenc\\Desktop\\sql_exercise\\vbo.db")
c = conn.cursor()

c.execute("SELECT * FROM Customers LIMIT 3;")
results = c.fetchall()
for result in results:
    print(result)

pd.read_sql("SELECT COUNT(CustomerID) AS MUSTERI_SAYISI, Country FROM Customers GROUP BY Country;",conn)

ab = pd.read_sql("SELECT COUNT(CustomerID) AS MUSTERI_SAYISI, Country FROM Customers GROUP BY Country;",conn)
type(ab)


c.execute("SELECT CustomerName FROM Customers LIMIT 5;")
c.fetchall()
for row in c.execute("SELECT CustomerName FROM Customers LIMIT 5;"):
    print(row)


# bağlantıyı kapatmak için
conn.close()