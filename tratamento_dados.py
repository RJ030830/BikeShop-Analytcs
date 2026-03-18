# =============================================================================
# BIKESHOP — FASE 00: TRATAMENTO E QUALIDADE DOS DADOS
# =============================================================================
# Autor      : Seu Nome
# Dataset    : BikeStores Sample Database (Kaggle)
# Descrição  : Pipeline de limpeza, padronização e construção da master table.
#              Cada decisão de tratamento está documentada com sua justificativa.
# Output     : 6 arquivos CSV limpos + master_bikeshop.csv (tabela unificada)
# =============================================================================

import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE PATHS
# -----------------------------------------------------------------------------
PATH_INPUT  = r"C:\Users\renat\ProjetoBikeShop\dados\\"
PATH_OUTPUT = r"C:\Users\renat\ProjetoBikeShop\dados\tratados\\"

os.makedirs(PATH_OUTPUT, exist_ok=True)


# =============================================================================
# SEÇÃO 1 — CARREGAMENTO DAS TABELAS
# =============================================================================

brands      = pd.read_csv(PATH_INPUT + "brands.csv")
categories  = pd.read_csv(PATH_INPUT + "categories.csv")
customers   = pd.read_csv(PATH_INPUT + "customers.csv")
orders      = pd.read_csv(PATH_INPUT + "orders.csv")
order_itens = pd.read_csv(PATH_INPUT + "order_itens1.csv")
products    = pd.read_csv(PATH_INPUT + "products.csv")
staffs      = pd.read_csv(PATH_INPUT + "staffs.csv")
stocks      = pd.read_csv(PATH_INPUT + "stocks.csv")
stores      = pd.read_csv(PATH_INPUT + "stores.csv")

print("=" * 60)
print("FASE 00 — TRATAMENTO DE DADOS")
print("=" * 60)
print("\n[1/8] Tabelas carregadas com sucesso")
for nome, df in [("brands", brands), ("categories", categories),
                 ("customers", customers), ("orders", orders),
                 ("order_itens", order_itens), ("products", products),
                 ("staffs", staffs), ("stocks", stocks), ("stores", stores)]:
    print(f"      {nome:<15} → {df.shape[0]:>5} linhas × {df.shape[1]} colunas")


# =============================================================================
# SEÇÃO 2 — TABELA: PRODUCTS
# =============================================================================
# PROBLEMA 1: coluna 'list_price' está em formato brasileiro (string)
#   Exemplos:  '379,99'      → separador decimal com vírgula
#              '2.899,99'    → ponto como milhar, vírgula como decimal
#              '1,549,00'    → dois separadores — ambiguidade no formato
#              '-233,875,00' → sinal negativo + formato corrompido (erro de digitação)
#
# PROBLEMA 2: coluna 'média' contém mistura de valores numéricos e labels
#   de estatística ('moda', 'mediana', 'desvio padrão') junto com NaN.
#   Essa coluna foi gerada durante análise anterior e salva por engano no CSV.
#   DECISÃO: descartar a coluna integralmente.
# -----------------------------------------------------------------------------

def parse_price_br(valor):
    """
    Converte preço em formato brasileiro (string) para float.

    Casos tratados:
      '379,99'       → 379.99       (padrão: vírgula decimal)
      '2.899,99'     → 2899.99      (ponto milhar + vírgula decimal)
      '1,549,00'     → 1549.00      (dois separadores — ambiguidade)
      '-233,875,00'  → 2338.75      (erro de digitação — produto 258)

    O caso '-233,875,00' é tratado separadamente porque:
      - O produto 258 (Electra Amsterdam Royal 8i) tem preço corrompido no CSV
      - Investigação da família de produtos confirma faixa de R$ 1.199 a R$ 3.999
      - O pedido 1459 (único com esse produto) tem status 'Completed' e foi enviado
      - Portanto não é devolução, mas provavelmente é um erro de digitação no cadastro do produto
      - Valor corrigido estimado: R$ 2.338,75
    """
    if pd.isna(valor):
        return np.nan

    v = str(valor).strip()

    # caso especial: produto 258 com erro de sinal e formato corrompido
    if v == '-233,875,00':
        return 2338.75

    # dois separadores: '1,549,00' → remover primeira vírgula, trocar segunda por ponto
    if v.count(',') > 1:
        v = v.replace(',', '', 1).replace(',', '.')
    else:
        # formato padrão: remover pontos de milhar, trocar vírgula por ponto decimal
        v = v.replace('.', '').replace(',', '.')

    return float(v)


products['list_price'] = products['list_price'].apply(parse_price_br)

# DECISÃO: remover coluna 'média' — lixo analítico (ver justificativa acima)
products = products.drop(columns=['média'])

print("\n[2/8] products — tratamento concluído")
print(f"      list_price convertida para float")
print(f"      coluna 'média' removida")
print(f"      Faixa de preços: ${products['list_price'].min():,.2f} → ${products['list_price'].max():,.2f}")


# =============================================================================
# SEÇÃO 3 — TABELA: ORDERS
# =============================================================================
# PROBLEMA 1: coluna 'shipped_date' contém o valor textual "Não Enviado"
#   em vez de NaN para pedidos não despachados.
#   DECISÃO: substituir por NaT (equivalente a NaN para datas).
#   Justificativa: 170 pedidos com status Pending, Processing ou Rejected
#   não foram enviados — o valor textual impedia a conversão para datetime.
#
# PROBLEMA 2: colunas de data estão como string (object).
#   DECISÃO: converter para datetime para permitir cálculos temporais.
#
# ENRIQUECIMENTO: criar colunas derivadas úteis para análise
#   status_label  → mapeamento legível do código numérico de status
#   year/month    → extração para análise temporal
#   delivery_days → lead time de entrega (apenas pedidos enviados)
# -----------------------------------------------------------------------------

# substituir "Não Enviado" por NaN antes de converter para datetime
orders['shipped_date'] = orders['shipped_date'].replace('Não Enviado', np.nan)

orders['order_date']    = pd.to_datetime(orders['order_date'])
orders['required_date'] = pd.to_datetime(orders['required_date'])
orders['shipped_date']  = pd.to_datetime(orders['shipped_date'], errors='coerce')

# mapeamento de status (fonte: documentação do dataset BikeStores)
STATUS_MAP = {1: 'Pending', 2: 'Processing', 3: 'Rejected', 4: 'Completed'}
orders['status_label'] = orders['order_status'].map(STATUS_MAP)

# colunas derivadas para análise temporal
orders['year']          = orders['order_date'].dt.year
orders['month']         = orders['order_date'].dt.month
orders['month_name']    = orders['order_date'].dt.strftime('%b')
orders['quarter']       = orders['order_date'].dt.quarter

# lead time: dias entre pedido e envio (NaN para pedidos não enviados — esperado)
orders['delivery_days'] = (orders['shipped_date'] - orders['order_date']).dt.days

print("\n[3/8] orders — tratamento concluído")
print(f"      'Não Enviado' → NaT: {orders['shipped_date'].isna().sum()} registros")
print(f"      Datas convertidas para datetime")
print(f"      Colunas derivadas criadas: status_label, year, month, quarter, delivery_days")
print(f"      Distribuição de status:")
for label, count in orders['status_label'].value_counts().items():
    print(f"        {label:<12} → {count}")


# =============================================================================
# SEÇÃO 4 — TABELA: CUSTOMERS
# =============================================================================
# PROBLEMA: colunas 'zip_code_uniques' e 'zip_code_count' têm 1.250 e 1.444
#   nulos respectivamente em uma tabela de 1.445 linhas.
#   Essas colunas foram geradas durante análise exploratória anterior e salvas
#   por engano no CSV original. Não fazem parte do schema original do dataset.
#   DECISÃO: descartar ambas.
# -----------------------------------------------------------------------------

customers = customers.drop(columns=['zip_code_uniques', 'zip_code_count'])

print("\n[4/8] customers — tratamento concluído")
print(f"      Colunas 'zip_code_uniques' e 'zip_code_count' removidas")
print(f"      Clientes com múltiplos pedidos:")
pedidos_por_cliente = orders['customer_id'].value_counts()
dist = pedidos_por_cliente.value_counts().rename_axis('pedidos').reset_index(name='clientes')
for _, row in dist.iterrows():
    print(f"        {row['pedidos']} pedido(s) → {row['clientes']} clientes")


# =============================================================================
# SEÇÃO 5 — TABELA: ORDER_ITENS
# =============================================================================
# PROBLEMA: 1 registro com list_price = -233.875 (product_id = 258)
#   Mesma raiz do problema em products: erro de digitação no cadastro.
#   O pedido 1459 tem status Completed e data de envio — não é devolução.
#   DECISÃO: corrigir para 2338.75 (valor estimado pela família de produtos).
#
# ENRIQUECIMENTO: calcular receita líquida por item
#   revenue = list_price × quantity × (1 - discount)
# -----------------------------------------------------------------------------

# corrigir preço negativo (mesmo erro do produto 258 em products)
order_itens.loc[order_itens['list_price'] < 0, 'list_price'] = 2338.75

# receita líquida após desconto
order_itens['revenue'] = (
    order_itens['list_price'] * order_itens['quantity'] * (1 - order_itens['discount'])
).round(2)

print("\n[5/8] order_itens — tratamento concluído")
print(f"      Preço negativo corrigido: 1 registro (product_id=258, order_id=1459)")
print(f"      Coluna 'revenue' criada (receita líquida por item)")
print(f"      Receita bruta total: ${order_itens['revenue'].sum():,.2f}")


# =============================================================================
# SEÇÃO 6 — TABELA: STAFFS
# =============================================================================
# Sem problemas de qualidade identificados.
# ENRIQUECIMENTO: criar coluna staff_name para facilitar visualizações.
# -----------------------------------------------------------------------------

staffs['staff_name'] = staffs['first_name'] + ' ' + staffs['last_name']
staffs_merge = staffs[['staff_id', 'staff_name', 'store_id']].rename(
    columns={'store_id': 'staff_store_id'}
)

print("\n[6/8] staffs — sem problemas críticos")
print(f"      Coluna 'staff_name' criada para visualizações")


# =============================================================================
# SEÇÃO 7 — CONSTRUÇÃO DA MASTER TABLE
# =============================================================================
# Objetivo: unir todas as tabelas em uma única tabela analítica desnormalizada.
# Essa tabela será a fonte primária para análises e para o dashboard Power BI.
#
# Estrutura do join:
#   order_itens (fato central)
#     ← orders        (via order_id)
#     ← products      (via product_id)
#     ← categories    (via category_id)
#     ← brands        (via brand_id)
#     ← customers     (via customer_id)
#     ← stores        (via store_id)     → renomear city/state → store_city/state
#     ← staffs        (via staff_id)
#
# Todos os joins são LEFT para preservar todos os itens de pedido,
# mesmo que alguma chave estrangeira não encontre correspondência.
# -----------------------------------------------------------------------------

master = (
    order_itens
    .merge(orders,
           on='order_id', how='left')
    .merge(products[['product_id', 'product_name', 'brand_id',
                      'category_id', 'model_year', 'list_price']],
           on='product_id', how='left', suffixes=('_sold', '_catalog'))
    .merge(categories,
           on='category_id', how='left')
    .merge(brands,
           on='brand_id', how='left')
    .merge(customers,
           on='customer_id', how='left')
    .merge(stores[['store_id', 'store_name', 'city', 'state']],
           on='store_id', how='left', suffixes=('_customer', '_store'))
    .merge(staffs_merge,
           on='staff_id', how='left')
)

# renomear colunas ambíguas geradas pelos sufixos
master = master.rename(columns={
    'city_customer':  'customer_city',
    'state_customer': 'customer_state',
    'city_store':     'store_city',
    'state_store':    'store_state',
})

print("\n[7/8] Master table construída")
print(f"      Shape: {master.shape[0]} linhas × {master.shape[1]} colunas")
print(f"      Colunas: {list(master.columns)}")


# =============================================================================
# SEÇÃO 8 — VERIFICAÇÃO FINAL DE QUALIDADE
# =============================================================================

print("\n[8/8] Verificação final de qualidade")

# nulos esperados: shipped_date e delivery_days para pedidos não concluídos
nulos = master.isnull().sum()
nulos_relevantes = nulos[nulos > 0]

if len(nulos_relevantes) > 0:
    print("\n      Nulos encontrados (verificar se são esperados):")
    for col, qtd in nulos_relevantes.items():
        esperado = "(esperado — pedidos não enviados)" if col in ['shipped_date', 'delivery_days'] else "⚠ VERIFICAR"
        print(f"        {col:<20} → {qtd:>4} nulos  {esperado}")
else:
    print("      Nenhum nulo inesperado encontrado ✓")

# resumo executivo — apenas pedidos concluídos
completed = master[master['status_label'] == 'Completed']

print("\n" + "=" * 60)
print("RESUMO EXECUTIVO — PEDIDOS CONCLUÍDOS")
print("=" * 60)
print(f"  Período          : {master['order_date'].min().date()} → {master['order_date'].max().date()}")
print(f"  Pedidos           : {completed['order_id'].nunique()}")
print(f"  Itens vendidos    : {len(completed)}")
print(f"  Receita total     : ${completed['revenue'].sum():,.2f}")
print(f"  Ticket médio/pedido: ${completed.groupby('order_id')['revenue'].sum().mean():,.2f}")
print(f"  Ticket médio/item  : ${completed['revenue'].mean():,.2f}")
print(f"  Lojas             : {completed['store_name'].nunique()}")
print(f"  Produtos únicos   : {completed['product_id'].nunique()}")
print(f"  Clientes únicos   : {completed['customer_id'].nunique()}")
print(f"  Vendedores        : {completed['staff_name'].nunique()}")


# =============================================================================
# SEÇÃO 9 — EXPORTAÇÃO
# =============================================================================

master.to_csv(PATH_OUTPUT + "master_bikeshop.csv",        index=False, decimal=',')
products.to_csv(PATH_OUTPUT + "products_clean.csv",       index=False)
orders.to_csv(PATH_OUTPUT + "orders_clean.csv",           index=False)
customers.to_csv(PATH_OUTPUT + "customers_clean.csv",     index=False)
order_itens.to_csv(PATH_OUTPUT + "order_itens_clean.csv", index=False)
stocks.to_csv(PATH_OUTPUT + "stocks_clean.csv",           index=False)

print("\n" + "=" * 60)
print("EXPORTAÇÃO CONCLUÍDA")
print("=" * 60)
print(f"  Destino: {PATH_OUTPUT}")
print("  Arquivos gerados:")
print("    → master_bikeshop.csv       (tabela analítica unificada)")
print("    → products_clean.csv        (produtos com preços corrigidos)")
print("    → orders_clean.csv          (pedidos com datas e status)")
print("    → customers_clean.csv       (clientes sem colunas inválidas)")
print("    → order_itens_clean.csv     (itens com receita calculada)")
print("    → stocks_clean.csv          (estoque por loja/produto)")
print("\n Tratamento_dados realizado. Próximo passo: desempenho_comercial")
