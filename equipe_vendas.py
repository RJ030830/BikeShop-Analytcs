# =============================================================================
# BIKESHOP — FASE 04: DESEMPENHO DA EQUIPE DE VENDAS
# =============================================================================
# Descrição  : Análise de receita, ticket médio, evolução anual e risco
#              operacional por vendedor.
#
# Perguntas respondidas:
#   P8 — Qual vendedor gera mais receita? Qual tem maior ticket médio?
#   P9 — Há correlação entre staff e pedidos rejeitados ou pendentes?
#
# DECISÃO METODOLÓGICA:
#   Para receita (P8): apenas pedidos COMPLETED.
#   Para status (P9): TODOS os pedidos — pedidos rejeitados/pendentes são
#   eventos atribuídos ao vendedor e precisam ser contabilizados como risco.
#
# Input  : master_bikeshop.csv, orders_clean.csv
# Output : fase_04_equipe_vendas.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import os

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE PATHS
# -----------------------------------------------------------------------------
PATH_INPUT  = r"C:\Users\renat\ProjetoBikeShop\dados\tratados\\"
PATH_OUTPUT = r"C:\Users\renat\ProjetoBikeShop\outputs\\"

os.makedirs(PATH_OUTPUT, exist_ok=True)


# =============================================================================
# SEÇÃO 1 — CARREGAMENTO
# =============================================================================

master = pd.read_csv(PATH_INPUT + "master_bikeshop.csv", parse_dates=['order_date'])
orders = pd.read_csv(PATH_INPUT + "orders_clean.csv",    parse_dates=['order_date'])

completed = master[master['status_label'] == 'Completed'].copy()

print("=" * 60)
print("FASE 04 — DESEMPENHO DA EQUIPE DE VENDAS")
print("=" * 60)
print(f"Pedidos concluídos: {completed['order_id'].nunique()}")
print(f"Vendedores ativos:  {completed['staff_name'].nunique()}")


# =============================================================================
# SEÇÃO 2 — P8: RECEITA E TICKET MÉDIO POR VENDEDOR
# =============================================================================
# NOTA: a rede tem 10 staffs cadastrados, mas apenas 6 aparecem em pedidos
# concluídos. Os demais (Fabiola Jackson, Jannette David, Bernardine Houston,
# Virgie Wiggins) são gerentes ou staffs sem pedidos atribuídos.

print("\n--- P8: Receita por vendedor ---")
p8 = (completed.groupby(['staff_id', 'staff_name', 'store_name'])
      .agg(receita   = ('revenue',    'sum'),
           pedidos   = ('order_id',   'nunique'),
           itens     = ('item_id',    'count'),
           clientes  = ('customer_id','nunique'))
      .reset_index())
p8['ticket_medio']       = (p8['receita'] / p8['pedidos']).round(2)
p8['itens_por_pedido']   = (p8['itens']   / p8['pedidos']).round(2)
p8['receita_por_cliente']= (p8['receita'] / p8['clientes']).round(2)

receita_loja = p8.groupby('store_name')['receita'].sum().reset_index()
receita_loja.columns = ['store_name', 'receita_loja']
p8 = p8.merge(receita_loja, on='store_name')
p8['pct_loja'] = (p8['receita'] / p8['receita_loja'] * 100).round(1)
p8 = p8.sort_values('receita', ascending=False)

print(p8[['staff_name', 'store_name', 'receita', 'pedidos',
          'ticket_medio', 'itens_por_pedido', 'pct_loja']].to_string(index=False))

# OBSERVAÇÃO: dentro de cada loja, a divisão é quase 50/50.
# Isso indica que pedidos são distribuídos de forma equilibrada.
# A diferença real de desempenho está no TICKET MÉDIO:
#   Kali Vargas (Rowlett) = $5.288 — melhor ticket da rede
#   Genna Serrano (Santa Cruz) = $4.378 — menor ticket


# =============================================================================
# SEÇÃO 3 — P8: EVOLUÇÃO ANUAL 2016 vs 2017
# =============================================================================

print("\n--- P8: Evolução anual por vendedor ---")
yoy_staff = (completed[completed['year'].isin([2016, 2017])]
             .groupby(['staff_name', 'store_name', 'year'])['revenue']
             .sum().round(2).reset_index())

pivot_yoy = (yoy_staff.pivot_table(index=['staff_name', 'store_name'],
                                    columns='year', values='revenue',
                                    fill_value=0).reset_index())
pivot_yoy.columns.name = None
pivot_yoy['var_%'] = (
    (pivot_yoy[2017] - pivot_yoy[2016]) /
    pivot_yoy[2016].replace(0, np.nan) * 100
).round(1)
print(pivot_yoy.to_string(index=False))

# ALERTA ANALÍTICO:
#   Marcelene Boyer (Baldwin): +82.2% — crescimento excepcional
#   Layla Terrell (Rowlett):  +106.2% — dobrou a receita em 1 ano
#   Genna Serrano (Santa Cruz): -11.4% — única em queda
# Isso reforça o achado da Fase 1: Santa Cruz está estagnada.
# A queda não é da loja como um todo — é dos dois vendedores.


# =============================================================================
# SEÇÃO 4 — P9: RISCO OPERACIONAL — STATUS DOS PEDIDOS POR STAFF
# =============================================================================
# Usamos TODOS os pedidos (qualquer status) para calcular taxa de rejeição.
# Um pedido rejeitado pode indicar: falha no atendimento, produto indisponível,
# problema de crédito do cliente, ou erro operacional do vendedor.
# Taxas muito acima da média da rede merecem investigação.

print("\n--- P9: Status dos pedidos por vendedor ---")
staff_orders = orders.merge(
    master[['order_id', 'staff_name', 'store_name']].drop_duplicates(),
    on='order_id', how='left')

p9 = (staff_orders.groupby(['staff_name', 'store_name', 'status_label'])['order_id']
      .count().reset_index())
p9.columns = ['staff_name', 'store_name', 'status', 'pedidos']

pivot_p9 = p9.pivot_table(index=['staff_name', 'store_name'],
                           columns='status', values='pedidos',
                           fill_value=0).reset_index()
pivot_p9.columns.name = None

for col in ['Completed', 'Pending', 'Processing', 'Rejected']:
    if col not in pivot_p9.columns:
        pivot_p9[col] = 0

pivot_p9['total']              = pivot_p9[['Completed','Pending','Processing','Rejected']].sum(axis=1)
pivot_p9['taxa_rejected']      = (pivot_p9['Rejected'] / pivot_p9['total'] * 100).round(1)
pivot_p9['taxa_nao_concluido'] = (
    (pivot_p9['Pending'] + pivot_p9['Processing'] + pivot_p9['Rejected'])
    / pivot_p9['total'] * 100).round(1)
pivot_p9 = pivot_p9.sort_values('taxa_rejected', ascending=False)

print(pivot_p9[['staff_name', 'store_name', 'Completed', 'Pending',
                'Processing', 'Rejected', 'taxa_rejected',
                'taxa_nao_concluido']].to_string(index=False))

media_rejected = pivot_p9['taxa_rejected'].mean()
print(f"\nTaxa média de rejeição na rede: {media_rejected:.1f}%")
alto_risco = pivot_p9[pivot_p9['taxa_rejected'] > 5]
if len(alto_risco) > 0:
    print("Vendedores acima de 5% de rejeição:")
    print(alto_risco[['staff_name', 'store_name', 'taxa_rejected']].to_string(index=False))

# ALERTA: Kali Vargas tem 12.5% de rejeição — muito acima da média.
# Isso é paradoxal: melhor ticket médio da rede + maior taxa de rejeição.
# Hipótese: Kali tenta fechar pedidos maiores (daí o ticket alto),
# mas com maior risco de falha. Merece investigação qualitativa.


# =============================================================================
# SEÇÃO 5 — CONSISTÊNCIA DO TICKET MÉDIO MENSAL
# =============================================================================
# Coeficiente de variação (CV) = desvio padrão / média × 100
# CV alto → vendedor inconsistente (bons e maus meses alternados)
# CV baixo → vendedor consistente e previsível

print("\n--- Consistência do ticket médio mensal ---")
ticket_mensal = (completed.groupby(['staff_name', 'year', 'month'])
                 .apply(lambda x: x.groupby('order_id')['revenue'].sum().mean())
                 .reset_index().rename(columns={0: 'ticket_medio'}))

consistencia = (ticket_mensal.groupby('staff_name')['ticket_medio']
                .agg(['mean', 'std', 'min', 'max'])
                .round(2).reset_index())
consistencia.columns = ['staff_name', 'media', 'desvio_padrao', 'minimo', 'maximo']
consistencia['coef_variacao'] = (
    consistencia['desvio_padrao'] / consistencia['media'] * 100
).round(1)

print(consistencia.sort_values('media', ascending=False).to_string(index=False))

# OBSERVAÇÃO: Genna Serrano tem o menor CV (25.3%) — mais consistente da rede.
# Kali Vargas tem CV de 54.1% — alta variabilidade.
# Para gestão: consistência pode ser tão valiosa quanto ticket médio alto.


# =============================================================================
# SEÇÃO 6 — VISUALIZAÇÕES
# =============================================================================

COR_LOJA    = {'Baldwin Bikes':'#2563EB','Santa Cruz Bikes':'#10B981','Rowlett Bikes':'#F59E0B'}
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

# --- Gráfico 1: Receita total por vendedor ---
ax1 = fig.add_subplot(3, 2, 1)
p8s = p8.sort_values('receita', ascending=True)
bars1 = ax1.barh(p8s['staff_name'], p8s['receita'],
                  color=[COR_LOJA[s] for s in p8s['store_name']],
                  height=0.55, zorder=3)
for bar, val, pct in zip(bars1, p8s['receita'], p8s['pct_loja']):
    ax1.text(bar.get_width() + 20000, bar.get_y() + bar.get_height()/2,
             f'${val/1e3:.0f}K\n({pct:.0f}% da loja)',
             va='center', fontsize=8.5, color='#374151', fontweight='bold')
estilo_limpo(ax1)
fmt_k(ax1, 'x')
ax1.set_xlim(0, 3.3e6)
ax1.set_title('Receita total por vendedor', fontsize=13,
              fontweight='bold', pad=12, color='#111827')
ax1.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in COR_LOJA.items()],
           frameon=False, fontsize=8, loc='lower right')

# --- Gráfico 2: Ticket médio ---
ax2 = fig.add_subplot(3, 2, 2)
p8t = p8.sort_values('ticket_medio', ascending=True)
bars2 = ax2.barh(p8t['staff_name'], p8t['ticket_medio'],
                  color=[COR_LOJA[s] for s in p8t['store_name']],
                  height=0.55, zorder=3)
for bar, val in zip(bars2, p8t['ticket_medio']):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', fontsize=9.5,
             color='#374151', fontweight='bold')
estilo_limpo(ax2)
fmt_k(ax2, 'x')
ax2.set_xlim(0, 7500)
ax2.set_title('Ticket médio por pedido', fontsize=13,
              fontweight='bold', pad=12, color='#111827')

# --- Gráfico 3: Evolução 2016 vs 2017 ---
ax3 = fig.add_subplot(3, 2, (3, 4))
staffs_ord = pivot_yoy.sort_values(2017, ascending=False)
x = np.arange(len(staffs_ord))
w = 0.35
ax3.bar(x - w/2, staffs_ord[2016], width=w - 0.03,
        color='#CBD5E1', label='2016', zorder=3)
bars_17 = ax3.bar(x + w/2, staffs_ord[2017], width=w - 0.03,
                   color=[COR_LOJA[s] for s in staffs_ord['store_name']],
                   label='2017', zorder=3)
for bar, val in zip(bars_17, staffs_ord[2017]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15000,
             f'${val/1e3:.0f}K', ha='center', fontsize=8.5,
             color='#374151', fontweight='bold')
for i, (_, row) in enumerate(staffs_ord.iterrows()):
    v = row['var_%']
    ax3.text(i + w/2, row[2017] + 60000,
             f'{"+" if v > 0 else ""}{v:.0f}%',
             ha='center', fontsize=9,
             color='#10B981' if v > 0 else '#EF4444',
             fontweight='bold')
estilo_limpo(ax3)
fmt_k(ax3)
ax3.set_xticks(x)
ax3.set_xticklabels(staffs_ord['staff_name'], rotation=15, ha='right', fontsize=9.5)
ax3.set_title('Evolução de receita por vendedor — 2016 vs 2017',
              fontsize=13, fontweight='bold', pad=12, color='#111827')
ax3.legend(handles=[mpatches.Patch(color='#CBD5E1', label='2016'),
                    mpatches.Patch(color='#6B7280', label='2017 (cor = loja)')],
           frameon=False, fontsize=9)

# --- Gráfico 4: Taxa de rejeição ---
ax4 = fig.add_subplot(3, 2, 5)
p9s = pivot_p9.sort_values('taxa_rejected', ascending=True)
cores4 = ['#EF4444' if v > 5 else '#F59E0B' if v > 2 else '#10B981'
          for v in p9s['taxa_rejected']]
bars4 = ax4.barh(p9s['staff_name'], p9s['taxa_rejected'],
                  color=cores4, height=0.55, zorder=3)
for bar, val, rej, tot in zip(bars4, p9s['taxa_rejected'],
                               p9s['Rejected'], p9s['total']):
    ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%  ({int(rej)}/{int(tot)} pedidos)',
             va='center', fontsize=9, color='#374151', fontweight='bold')
estilo_limpo(ax4)
ax4.set_xlim(0, 20)
ax4.axvline(x=5, color='#EF4444', linewidth=1.2, linestyle='--', alpha=0.5)
ax4.text(5.2, ax4.get_ylim()[1] * 0.95, 'limiar 5%',
         fontsize=8, color='#EF4444')
ax4.set_title('Taxa de pedidos rejeitados por vendedor',
              fontsize=13, fontweight='bold', pad=12, color='#111827')
ax4.legend(handles=[
    mpatches.Patch(color='#EF4444', label='>5% — atenção'),
    mpatches.Patch(color='#F59E0B', label='2-5% — monitorar'),
    mpatches.Patch(color='#10B981', label='<2% — ok'),
], frameon=False, fontsize=9, loc='lower right')

# --- Gráfico 5: Consistência do ticket médio ---
ax5 = fig.add_subplot(3, 2, 6)
cons = consistencia.sort_values('media', ascending=True)
y_pos = np.arange(len(cons))
cores5 = [COR_LOJA.get(
    p8[p8['staff_name'] == s]['store_name'].values[0], '#94A3B8')
    for s in cons['staff_name']]
ax5.barh(y_pos, cons['media'], height=0.55, color=cores5, alpha=0.8, zorder=3)
ax5.errorbar(cons['media'], y_pos, xerr=cons['desvio_padrao'],
             fmt='none', color='#374151', capsize=4, linewidth=1.5, zorder=4)
for i, (_, row) in enumerate(cons.iterrows()):
    ax5.text(row['media'] + 80, i, f'CV: {row["coef_variacao"]:.0f}%',
             va='center', fontsize=9, color='#374151')
ax5.set_yticks(y_pos)
ax5.set_yticklabels(cons['staff_name'], fontsize=10)
estilo_limpo(ax5)
fmt_k(ax5, 'x')
ax5.set_xlim(0, 10000)
ax5.set_title('Consistência do ticket médio mensal\n'
              '(barra = média, traço = ±1 desvio padrão)',
              fontsize=13, fontweight='bold', pad=12, color='#111827')

plt.suptitle('Fase 4 — Desempenho da Equipe de Vendas',
             fontsize=15, fontweight='bold', y=1.005, color='#111827')
plt.tight_layout(h_pad=4, w_pad=3)
plt.savefig(PATH_OUTPUT + "fase_04_equipe_vendas.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nGráfico salvo: fase_04_equipe_vendas.png")
print("\nFase 04 finalizada. Próximo passo: dashboard Power BI.")
