# =============================================================================
# BIKESHOP — FASE 01: DESEMPENHO COMERCIAL DA REDE
# =============================================================================
# Autor      : Seu Nome
# Descrição  : Análise de receita, ticket médio e sazonalidade por loja.
#              Responde às perguntas de negócio P1 e P2 do projeto.
#
# Perguntas respondidas:
#   P1 — Qual loja gera mais receita? Qual tem melhor ticket médio?
#   P2 — Como variam as vendas ao longo do ano? Qual dia da semana vende mais?
#
# Input  : master_bikeshop.csv (gerado pela fase_00)
# Output : fase_01_desempenho_comercial.png
#          fase_01_dia_semana.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE PATHS
# -----------------------------------------------------------------------------
PATH_INPUT  = r"C:\Users\renat\ProjetoBikeShop\dados\tratados\\"
PATH_OUTPUT = r"C:\Users\renat\ProjetoBikeShop\outputs\\"

import os
os.makedirs(PATH_OUTPUT, exist_ok=True)


# =============================================================================
# SEÇÃO 1 — CARREGAMENTO E FILTRO
# =============================================================================
# Carregamos apenas pedidos com status 'Completed'.
# Justificativa: pedidos Pending, Processing e Rejected não geraram receita
# real para o negócio — incluí-los distorceria qualquer métrica financeira.

master = pd.read_csv(PATH_INPUT + "master_bikeshop.csv", parse_dates=['order_date'])
completed = master[master['status_label'] == 'Completed'].copy()

# colunas de tempo úteis para sazonalidade
completed['weekday']     = completed['order_date'].dt.day_name()
completed['weekday_num'] = completed['order_date'].dt.dayofweek

print("=" * 60)
print("FASE 01 — DESEMPENHO COMERCIAL")
print("=" * 60)
print(f"Pedidos concluídos carregados: {completed['order_id'].nunique()}")
print(f"Período: {completed['order_date'].min().date()} → {completed['order_date'].max().date()}")


# =============================================================================
# SEÇÃO 2 — P1: RECEITA E TICKET MÉDIO POR LOJA
# =============================================================================
# ALERTA ANALÍTICO: Baldwin Bikes tem 3x mais pedidos que Santa Cruz.
# Volume alto não equivale a maior eficiência — por isso calculamos também
# o ticket médio por pedido para uma comparação justa entre as lojas.

print("\n--- P1: Receita por loja ---")
receita_loja = (completed.groupby('store_name')['revenue']
                .sum().sort_values(ascending=False).reset_index())
receita_loja.columns = ['store', 'receita_total']
receita_loja['participacao_pct'] = (
    receita_loja['receita_total'] / receita_loja['receita_total'].sum() * 100
).round(1)
print(receita_loja.to_string(index=False))

print("\n--- P1: Ticket médio por pedido ---")
ticket_medio = (completed.groupby(['store_name', 'order_id'])['revenue']
                .sum().reset_index()
                .groupby('store_name')['revenue']
                .mean().round(2).reset_index())
ticket_medio.columns = ['store', 'ticket_medio']
print(ticket_medio.sort_values('ticket_medio', ascending=False).to_string(index=False))


# =============================================================================
# SEÇÃO 3 — P1: EVOLUÇÃO ANUAL POR LOJA
# =============================================================================
# DECISÃO METODOLÓGICA: excluímos 2018 da comparação anual.
# 2018 só tem dados até março (292 pedidos vs ~660 nos anos completos).
# Comparar 2018 com 2016/2017 seria comparar anos incompletos com completos —
# qualquer conclusão sobre "queda de receita em 2018" seria falsa.

print("\n--- P1: Evolução anual 2016 vs 2017 (anos completos) ---")
yoy = (completed[completed['year'].isin([2016, 2017])]
       .groupby(['year', 'store_name'])['revenue']
       .sum().round(2).reset_index())

pivot_yoy = yoy.pivot(index='store_name', columns='year', values='revenue').fillna(0)
pivot_yoy['variacao_%'] = (
    (pivot_yoy[2017] - pivot_yoy[2016]) / pivot_yoy[2016] * 100
).round(1)
print(pivot_yoy.to_string())
print("\nNOTA: Santa Cruz Bikes teve variação de -0.1% — crescimento praticamente nulo")
print("      Baldwin e Rowlett cresceram ~52% — investigar causas na Fase 3 (clientes)")


# =============================================================================
# SEÇÃO 4 — P2: SAZONALIDADE MENSAL
# =============================================================================

print("\n--- P2: Receita mensal 2016 vs 2017 ---")
mensal = (completed[completed['year'].isin([2016, 2017])]
          .groupby(['year', 'month', 'month_name'])['revenue']
          .sum().reset_index().sort_values(['year', 'month']))
print(mensal.to_string(index=False))

pico = mensal.loc[mensal['revenue'].idxmax()]
vale = mensal.loc[mensal['revenue'].idxmin()]
print(f"\nMês de maior receita: {pico['month_name']} {int(pico['year'])} — ${pico['revenue']:,.0f}")
print(f"Mês de menor receita: {vale['month_name']} {int(vale['year'])} — ${vale['revenue']:,.0f}")


# =============================================================================
# SEÇÃO 5 — P2: RECEITA POR DIA DA SEMANA
# =============================================================================

print("\n--- P2: Receita por dia da semana ---")
dow = (completed.groupby(['weekday_num', 'weekday'])
       .agg(receita=('revenue', 'sum'), pedidos=('order_id', 'nunique'))
       .reset_index().sort_values('weekday_num'))
dow['ticket_medio'] = (dow['receita'] / dow['pedidos']).round(2)
print(dow[['weekday', 'receita', 'pedidos', 'ticket_medio']].to_string(index=False))
print("\nNOTA: Segunda-feira lidera em receita total, mas Domingo tem mais pedidos.")
print("      Isso indica que segunda tem ticket médio mais alto — compras de maior valor.")


# =============================================================================
# SEÇÃO 6 — VISUALIZAÇÕES
# =============================================================================

COR_LOJAS = {
    'Baldwin Bikes':    '#2563EB',
    'Santa Cruz Bikes': '#10B981',
    'Rowlett Bikes':    '#F59E0B',
}
CINZA_GRADE = '#E5E7EB'
CINZA_TEXTO = '#6B7280'

def formatar_dolar(ax, eixo='y'):
    fmt = mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'
    )
    if eixo == 'y': ax.yaxis.set_major_formatter(fmt)
    else:           ax.xaxis.set_major_formatter(fmt)

def estilo_limpo(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(CINZA_GRADE)
    ax.tick_params(colors=CINZA_TEXTO, labelsize=10)
    ax.yaxis.grid(True, color=CINZA_GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ── Figura principal (4 gráficos) ────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('white')

# --- Gráfico 1: Receita total por loja ---
ax1 = fig.add_subplot(3, 2, 1)
p1 = receita_loja.sort_values('receita_total')
cores = [COR_LOJAS[s] for s in p1['store']]
bars = ax1.barh(p1['store'], p1['receita_total'], color=cores, height=0.5, zorder=3)
for bar, val, pct in zip(bars, p1['receita_total'], p1['participacao_pct']):
    ax1.text(val + 30000, bar.get_y() + bar.get_height()/2,
             f'${val/1e6:.2f}M  ({pct:.0f}%)',
             va='center', fontsize=10, color='#374151', fontweight='bold')
estilo_limpo(ax1)
formatar_dolar(ax1, 'x')
ax1.set_xlim(0, 6e6)
ax1.set_title('Receita total por loja', fontsize=13, fontweight='bold',
              pad=12, color='#111827')

# --- Gráfico 2: Ticket médio por pedido ---
ax2 = fig.add_subplot(3, 2, 2)
tk = ticket_medio.sort_values('ticket_medio')
cores2 = [COR_LOJAS[s] for s in tk['store']]
bars2 = ax2.barh(tk['store'], tk['ticket_medio'], color=cores2, height=0.5, zorder=3)
for bar, val in zip(bars2, tk['ticket_medio']):
    ax2.text(val + 50, bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', fontsize=10,
             color='#374151', fontweight='bold')
estilo_limpo(ax2)
formatar_dolar(ax2, 'x')
ax2.set_xlim(0, 7000)
ax2.set_title('Ticket médio por pedido', fontsize=13, fontweight='bold',
              pad=12, color='#111827')

# --- Gráfico 3: Evolução anual 2016 vs 2017 ---
ax3 = fig.add_subplot(3, 2, (3, 4))
lojas  = list(COR_LOJAS.keys())
anos   = [2016, 2017]
x      = np.arange(len(anos))
width  = 0.25

for i, loja in enumerate(lojas):
    vals = [yoy[(yoy['year'] == a) & (yoy['store_name'] == loja)]['revenue'].values[0]
            for a in anos]
    offset  = (i - 1) * width
    bars3 = ax3.bar(x + offset, vals, width=width - 0.03,
                    color=COR_LOJAS[loja], label=loja, zorder=3)
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15000,
                 f'${val/1e6:.2f}M', ha='center', fontsize=9,
                 color='#374151', fontweight='bold')

val16_b = yoy[(yoy['year']==2016) & (yoy['store_name']=='Baldwin Bikes')]['revenue'].values[0]
val17_b = yoy[(yoy['year']==2017) & (yoy['store_name']=='Baldwin Bikes')]['revenue'].values[0]
var_b   = (val17_b - val16_b) / val16_b * 100
ax3.annotate(f'+{var_b:.0f}%',
             xy=(1 - width, val17_b), xytext=(1 - width - 0.05, val17_b + 120000),
             fontsize=10, color=COR_LOJAS['Baldwin Bikes'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COR_LOJAS['Baldwin Bikes'], lw=1.2))

estilo_limpo(ax3)
formatar_dolar(ax3)
ax3.set_xticks(x)
ax3.set_xticklabels(['2016', '2017'], fontsize=11)
ax3.set_title('Evolução de receita por loja — 2016 vs 2017\n'
              '(2018 excluído — dados apenas até março)',
              fontsize=13, fontweight='bold', pad=12, color='#111827')
ax3.legend(loc='upper left', frameon=False, fontsize=10)

# --- Gráfico 4: Receita mensal 2016 vs 2017 ---
ax4 = fig.add_subplot(3, 2, (5, 6))
meses_ord = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
cores_ano = {2016: '#94A3B8', 2017: '#2563EB'}

for ano in [2016, 2017]:
    df_ano = (mensal[mensal['year'] == ano]
              .set_index('month').reindex(range(1, 13)).reset_index())
    df_ano['month_name'] = meses_ord
    ax4.plot(meses_ord, df_ano['revenue'],
             marker='o', markersize=6, linewidth=2.2,
             color=cores_ano[ano], label=str(ano), zorder=3)

ax4.annotate('Pico\nJun 2017\n$375K',
             xy=(5, 375577), xytext=(6, 340000),
             fontsize=9, color=cores_ano[2017], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=cores_ano[2017], lw=1.2))

estilo_limpo(ax4)
formatar_dolar(ax4)
ax4.set_title('Receita mensal — 2016 vs 2017',
              fontsize=13, fontweight='bold', pad=12, color='#111827')
ax4.legend(frameon=False, fontsize=10)

plt.suptitle('Fase 1 — Desempenho Comercial da Rede Bikeshop',
             fontsize=15, fontweight='bold', y=1.01, color='#111827')
plt.tight_layout(h_pad=3.5, w_pad=3)
plt.savefig(PATH_OUTPUT + "fase_01_desempenho_comercial.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nGráfico 1 salvo: fase_01_desempenho_comercial.png")


# ── Gráfico separado: Dia da semana ──────────────────────────────────────────
fig2, ax5 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('white')

ordem_pt = ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo']
cores_dow = ['#2563EB' if d in ['Monday','Sunday'] else '#93C5FD'
             for d in dow['weekday']]

bars5 = ax5.bar(ordem_pt, dow['receita'], color=cores_dow, zorder=3, width=0.6)
for bar, val, ped in zip(bars5, dow['receita'], dow['pedidos']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000,
             f'${val/1e3:.0f}K\n({ped} ped.)',
             ha='center', fontsize=9, color='#374151', fontweight='bold')

estilo_limpo(ax5)
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax5.set_ylim(0, 1.3e6)
ax5.set_title('Receita por dia da semana — todos os anos',
              fontsize=13, fontweight='bold', pad=12, color='#111827')

legend_dow = [Patch(color='#2563EB', label='Maior receita'),
              Patch(color='#93C5FD', label='Demais dias')]
ax5.legend(handles=legend_dow, frameon=False, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(PATH_OUTPUT + "fase_01_dia_semana.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Gráfico 2 salvo: fase_01_dia_semana.png")

print("\n Desempenho_comercial finalizado. Próximo passo: rentabilidade_produtos")
