# =============================================================================
# BIKESHOP — FASE 02: RENTABILIDADE DE PRODUTOS
# =============================================================================
# Autor      : Seu Nome
# Descrição  : Análise de receita por produto, categoria e marca.
#              Impacto dos descontos na receita. Diagnóstico de estoque parado.
#
# Perguntas respondidas:
#   P3 — Quais produtos e categorias são mais rentáveis?
#   P4 — Os descontos estão corroendo receita — em quanto e onde?
#   P5 — O estoque atual está alinhado com o que vende?
#
# Input  : master_bikeshop.csv, stocks_clean.csv, products_clean.csv
# Output : fase_02_rentabilidade_produtos.png
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
import os

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE PATHS — ajuste para seu ambiente
# -----------------------------------------------------------------------------
PATH_INPUT  = r"C:\Users\renat\ProjetoBikeShop\dados\tratados\\"
PATH_RAW    = r"C:\Users\renat\ProjetoBikeShop\dados\\"
PATH_OUTPUT = r"C:\Users\renat\ProjetoBikeShop\outputs\\"

os.makedirs(PATH_OUTPUT, exist_ok=True)


# =============================================================================
# SEÇÃO 1 — CARREGAMENTO
# =============================================================================

master   = pd.read_csv(PATH_INPUT + "master_bikeshop.csv", parse_dates=['order_date'])
stocks   = pd.read_csv(PATH_INPUT + "stocks_clean.csv")
products = pd.read_csv(PATH_INPUT + "products_clean.csv")
stores   = pd.read_csv(PATH_RAW   + "stores.csv")

completed = master[master['status_label'] == 'Completed'].copy()

# ENRIQUECIMENTO: métricas de desconto por item
# receita_bruta  = preço × quantidade (sem aplicar desconto)
# desconto_valor = diferença entre bruto e líquido
completed['receita_bruta']  = completed['list_price_sold'] * completed['quantity']
completed['desconto_valor'] = (completed['receita_bruta'] - completed['revenue']).round(2)

print("=" * 60)
print("FASE 02 — RENTABILIDADE DE PRODUTOS")
print("=" * 60)
print(f"Itens analisados: {len(completed)}")


# =============================================================================
# SEÇÃO 2 — P3: TOP PRODUTOS POR RECEITA
# =============================================================================

print("\n--- P3: Top 10 produtos por receita líquida ---")
top_prod = (completed.groupby(['product_id', 'product_name'])
            .agg(receita=('revenue', 'sum'),
                 qtd=('quantity', 'sum'),
                 pedidos=('order_id', 'nunique'))
            .sort_values('receita', ascending=False)
            .head(10).reset_index())
top_prod['ticket_medio'] = (top_prod['receita'] / top_prod['qtd']).round(2)
print(top_prod[['product_name', 'receita', 'qtd', 'ticket_medio']].to_string(index=False))

# OBSERVAÇÃO: Trek Slash 8 27.5 lidera em receita ($544K) com 151 unidades.
# Trek Silque SLR 8 Women's tem apenas 28 unidades mas $168K — ticket altíssimo.
# Esses dois perfis precisam de estratégias diferentes: volume vs. margem.


# =============================================================================
# SEÇÃO 3 — P3: RECEITA POR CATEGORIA E MARCA
# =============================================================================

print("\n--- P3: Receita por categoria ---")
cat = (completed.groupby('category_name')
       .agg(receita=('revenue', 'sum'),
            qtd=('quantity', 'sum'),
            pedidos=('order_id', 'nunique'))
       .sort_values('receita', ascending=False).reset_index())
cat['ticket_cat'] = (cat['receita'] / cat['qtd']).round(2)
cat['pct_receita'] = (cat['receita'] / cat['receita'].sum() * 100).round(1)
print(cat.to_string(index=False))

# ALERTA ANALÍTICO: Mountain Bikes responde por 37% da receita mas tem
# ticket médio de apenas $1.547. Road Bikes tem 19.9% com ticket de $2.955.
# Isso indica que Road Bikes é mais eficiente por unidade vendida.

print("\n--- P3: Receita por marca ---")
marca = (completed.groupby('brand_name')
         .agg(receita=('revenue', 'sum'), qtd=('quantity', 'sum'))
         .sort_values('receita', ascending=False).reset_index())
marca['pct'] = (marca['receita'] / marca['receita'].sum() * 100).round(1)
print(marca.to_string(index=False))

# OBSERVAÇÃO: Trek representa 58.8% da receita com apenas 1.564 unidades.
# Electra tem 2.329 unidades mas apenas 15.3% da receita — perfil de volume/baixo ticket.


# =============================================================================
# SEÇÃO 4 — P4: IMPACTO DOS DESCONTOS
# =============================================================================
# NOTA METODOLÓGICA: usamos list_price_sold (preço cobrado no pedido) como
# base do cálculo de desconto, não list_price_catalog (preço de tabela).
# Isso garante que estamos medindo o desconto real aplicado em cada venda,
# independente de atualizações de catálogo.

print("\n--- P4: Impacto dos descontos por categoria ---")
desc = (completed.groupby('category_name')
        .agg(receita_bruta=('receita_bruta', 'sum'),
             receita_liquida=('revenue', 'sum'),
             desconto_valor=('desconto_valor', 'sum'),
             desc_medio_pct=('discount', 'mean'))
        .reset_index())
desc['pct_desconto'] = (desc['desconto_valor'] / desc['receita_bruta'] * 100).round(1)
desc['desc_medio_pct'] = (desc['desc_medio_pct'] * 100).round(1)
desc = desc.sort_values('desconto_valor', ascending=False)
print(desc[['category_name', 'desc_medio_pct', 'desconto_valor',
            'pct_desconto', 'receita_liquida']].to_string(index=False))

total_bruto = desc['receita_bruta'].sum()
total_desc  = desc['desconto_valor'].sum()
print(f"\nReceita bruta total:   ${total_bruto:,.2f}")
print(f"Total em descontos:    ${total_desc:,.2f}")
print(f"Impacto % na receita:  {total_desc/total_bruto*100:.1f}%")

# ALERTA ANALÍTICO: o desconto médio é praticamente uniforme entre categorias
# (~10-11%). Isso sugere que a política de desconto é aplicada de forma
# indiscriminada — sem considerar elasticidade de preço por categoria.
# Mountain Bikes perdeu $293K em desconto. Vale investigar se esses descontos
# realmente geraram mais volume ou foram concedidos desnecessariamente.

print("\n--- P4: Distribuição dos níveis de desconto ---")
print(completed['discount'].value_counts().sort_index()
      .rename(index=lambda x: f'{x*100:.0f}%').to_string())
# OBSERVAÇÃO: apenas 4 valores de desconto (5%, 7%, 10%, 20%) em distribuição
# quase uniforme. Isso é um sinal de política de desconto padronizada —
# não há negociação por cliente ou produto.


# =============================================================================
# SEÇÃO 5 — P5: DIAGNÓSTICO DE ESTOQUE
# =============================================================================

vendas_prod = (completed.groupby('product_id')['quantity']
               .sum().reset_index().rename(columns={'quantity': 'vendido'}))

estoque = (stocks
           .merge(products[['product_id', 'product_name']], on='product_id')
           .merge(stores[['store_id', 'store_name']], on='store_id')
           .merge(vendas_prod, on='product_id', how='left'))
estoque['vendido'] = estoque['vendido'].fillna(0)

print("\n--- P5: Produtos sem estoque por loja ---")
zero = estoque[estoque['quantity'] == 0]
print(zero.groupby('store_name').size().reset_index(name='produtos_zerados'))

print("\n--- P5: Estoque parado (qtd > 20 e vendas < 5 unidades) ---")
parado = (estoque[(estoque['quantity'] > 20) & (estoque['vendido'] < 5)]
          .sort_values('quantity', ascending=False))
print(parado[['store_name', 'product_name', 'quantity', 'vendido']].to_string(index=False))

print("\n--- P5: Produtos mais vendidos vs estoque disponível ---")
top_vend = vendas_prod.sort_values('vendido', ascending=False).head(10)
top_vend_est = (top_vend
    .merge(estoque.groupby('product_id')['quantity'].sum().reset_index()
           .rename(columns={'quantity': 'estoque_total'}), on='product_id')
    .merge(products[['product_id', 'product_name']], on='product_id'))
print(top_vend_est[['product_name', 'vendido', 'estoque_total']].to_string(index=False))

# ALERTA: Trek Slash 8 27.5 — produto de maior receita — tem apenas 28 un. de estoque
# total contra 151 vendidos no período. Risco de ruptura em pico de demanda.


# =============================================================================
# SEÇÃO 6 — VISUALIZAÇÕES
# =============================================================================

AZUL       = '#2563EB'
AZUL_CLARO = '#BFDBFE'
AMARELO    = '#F59E0B'
VERMELHO   = '#EF4444'
CINZA_GRADE = '#E5E7EB'
CINZA_TEXTO = '#6B7280'

def estilo_limpo(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(CINZA_GRADE)
    ax.tick_params(colors=CINZA_TEXTO, labelsize=10)
    ax.yaxis.grid(True, color=CINZA_GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

def fmt_k(ax, eixo='y'):
    f = mticker.FuncFormatter(
        lambda x, _: f'${x/1e3:.0f}K' if x < 1e6 else f'${x/1e6:.1f}M'
    )
    if eixo == 'y': ax.yaxis.set_major_formatter(f)
    else:           ax.xaxis.set_major_formatter(f)


fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor('white')

# --- Gráfico 1: Top 10 produtos por receita ---
ax1 = fig.add_subplot(3, 2, (1, 2))
top_plot = (completed.groupby('product_name')['revenue']
            .sum().sort_values(ascending=False).head(10).reset_index())
top_plot['label'] = top_plot['product_name'].str.replace(r' - \d{4}.*', '', regex=True)
cores_prod = [AZUL if i < 3 else AZUL_CLARO for i in range(len(top_plot))]
bars1 = ax1.barh(top_plot['label'][::-1], top_plot['revenue'][::-1],
                 color=cores_prod[::-1], height=0.6, zorder=3)
for bar, val in zip(bars1, top_plot['revenue'][::-1]):
    ax1.text(val + 5000, bar.get_y() + bar.get_height()/2,
             f'${val/1e3:.0f}K', va='center', fontsize=9.5,
             color='#374151', fontweight='bold')
estilo_limpo(ax1)
fmt_k(ax1, 'x')
ax1.set_xlim(0, 680000)
ax1.set_title('Top 10 produtos por receita líquida', fontsize=13,
              fontweight='bold', pad=12, color='#111827')

# --- Gráfico 2: Receita por categoria (pizza) ---
ax2 = fig.add_subplot(3, 2, 3)
cat_plot = (completed.groupby('category_name')['revenue']
            .sum().sort_values(ascending=False).reset_index())
n = len(cat_plot)
cores_cat = [AZUL if i == 0 else '#60A5FA' if i == 1 else AZUL_CLARO for i in range(n)]
wedges, _, autotexts = ax2.pie(
    cat_plot['revenue'], labels=None, autopct='%1.1f%%',
    colors=cores_cat, startangle=140,
    wedgeprops=dict(linewidth=1.5, edgecolor='white'), pctdistance=0.78)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color('white')
    at.set_fontweight('bold')
ax2.legend(cat_plot['category_name'], loc='lower left', fontsize=8.5,
           frameon=False, bbox_to_anchor=(-0.1, -0.15))
ax2.set_title('Receita por categoria', fontsize=13, fontweight='bold',
              pad=12, color='#111827')

# --- Gráfico 3: Receita por marca ---
ax3 = fig.add_subplot(3, 2, 4)
marca_plot = (completed.groupby('brand_name')['revenue']
              .sum().sort_values().reset_index())
cores_m = [AZUL if m == 'Trek' else
           '#60A5FA' if m in ['Electra', 'Surly'] else AZUL_CLARO
           for m in marca_plot['brand_name']]
bars3 = ax3.barh(marca_plot['brand_name'], marca_plot['revenue'],
                 color=cores_m, height=0.6, zorder=3)
for bar, val in zip(bars3, marca_plot['revenue']):
    pct = val / marca_plot['revenue'].sum() * 100
    ax3.text(val + 5000, bar.get_y() + bar.get_height()/2,
             f'${val/1e3:.0f}K ({pct:.0f}%)',
             va='center', fontsize=9, color='#374151', fontweight='bold')
estilo_limpo(ax3)
fmt_k(ax3, 'x')
ax3.set_xlim(0, 5e6)
ax3.set_title('Receita por marca', fontsize=13, fontweight='bold',
              pad=12, color='#111827')

# --- Gráfico 4: Desconto por categoria ---
ax4 = fig.add_subplot(3, 2, 5)
desc_plot = desc.sort_values('desconto_valor', ascending=False)
bars4 = ax4.bar(desc_plot['category_name'], desc_plot['desconto_valor'],
                color=AMARELO, zorder=3, width=0.6)
for bar, val, pct in zip(bars4, desc_plot['desconto_valor'], desc_plot['pct_desconto']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f'${val/1e3:.0f}K\n({pct:.0f}%)',
             ha='center', fontsize=8.5, color='#374151', fontweight='bold')
estilo_limpo(ax4)
fmt_k(ax4)
ax4.set_xticklabels(desc_plot['category_name'], rotation=30, ha='right', fontsize=9)
ax4.set_title('Receita perdida em descontos por categoria',
              fontsize=13, fontweight='bold', pad=12, color='#111827')

# --- Gráfico 5: Estoque parado ---
ax5 = fig.add_subplot(3, 2, 6)
parado_plot = (estoque[(estoque['quantity'] > 20) & (estoque['vendido'] < 5)]
               .sort_values('quantity', ascending=False).head(8))
parado_plot['label'] = (
    parado_plot['product_name'].str.replace(r' - \d{4}.*', '', regex=True)
    + '\n(' + parado_plot['store_name'].str.split().str[0] + ')'
)
bars5 = ax5.barh(parado_plot['label'][::-1], parado_plot['quantity'][::-1],
                 color=VERMELHO, height=0.6, zorder=3, alpha=0.85)
for bar, est, vend in zip(bars5, parado_plot['quantity'][::-1],
                          parado_plot['vendido'][::-1]):
    ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{int(est)} un. / {int(vend)} vendidos',
             va='center', fontsize=9, color='#374151', fontweight='bold')
estilo_limpo(ax5)
ax5.set_xlim(0, 50)
ax5.set_title('Estoque parado\n(estoque > 20 e vendas < 5 unidades)',
              fontsize=13, fontweight='bold', pad=12, color='#111827')

plt.suptitle('Fase 2 — Rentabilidade de Produtos',
             fontsize=15, fontweight='bold', y=1.005, color='#111827')
plt.tight_layout(h_pad=4, w_pad=3)
plt.savefig(PATH_OUTPUT + "fase_02_rentabilidade_produtos.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nGráfico salvo: fase_02_rentabilidade_produtos.png")
print("\nFase 02 finalizada. Próximo passo: fase_03_clientes.py")
