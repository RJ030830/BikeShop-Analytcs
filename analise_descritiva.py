import pandas as pd
pd.set_option('display.width', None)

# CONFIGURAÇÃO DE PATH
PATH_INPUT  = r"C:\Users\renat\ProjetoBikeShop\dados\\"

# Importação de bases .csv
brands      = pd.read_csv(PATH_INPUT + "brands.csv")
categories  = pd.read_csv(PATH_INPUT + "categories.csv")
customers   = pd.read_csv(PATH_INPUT + "customers.csv")
orders      = pd.read_csv(PATH_INPUT + "orders.csv")
order_itens = pd.read_csv(PATH_INPUT + "order_itens1.csv")
products    = pd.read_csv(PATH_INPUT + "products.csv")
staffs      = pd.read_csv(PATH_INPUT + "staffs.csv")
stocks      = pd.read_csv(PATH_INPUT + "stocks.csv")
stores      = pd.read_csv(PATH_INPUT + "stores.csv")

# Entendimento inicial dos Dados
print("Descrição Dados Brands \n:", brands.describe())
print("\n Descrição Dados Categories \n:", categories.describe())
print("\n Descrição Dados Customers \n:", customers.describe())
print("\n Descrição Dados Order_itens \n:", order_itens.describe())
print("\n Descrição Dados Orders \n:", orders.describe())
print("\n Descrição Dados Products \n:", products.describe())
print("\n Descrição Dados Staffs \n:", staffs.describe())
print("\n Descrição Dados Stocks \n:", stocks.describe())
print("\n Descrição Dados Stores \n:", stores.describe())


"""
 Aqui já podemos anotar alguns números interessantes, como por exemplo:
 As bases de dados apontam:
 - 3 Lojas.
 - 1615 pedidos.
 - 1445 Clientes.
 - 321 Produtos.
 - A empresa trabalha com 9 marcas diferentes.
 - Vende 7 categorias de produtos.
 - Possui registros de 10 Funcionários.
 - Tem média de estoque de 14 produtos.

"""

print("\n analise_descritiva finalizada. Próximo passo: tratamento_dados")
