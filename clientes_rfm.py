# =============================================================================
# BIKESHOP — FASE 03: COMPORTAMENTO DE CLIENTES E RFM
# =============================================================================
# Autor      : Seu Nome
# Descrição  : Segmentação de clientes via análise RFM e perfil geográfico.
#
# Perguntas respondidas:
#   P6 — Quem são os clientes mais valiosos? (Análise RFM)
#   P7 — De onde vêm os clientes de maior valor? (Perfil geográfico)
#
# DECISÃO METODOLÓGICA — RFM com fonte de dados mista:
#   R (Recência)   → TODOS os pedidos (qualquer status)
#   F (Frequência) → TODOS os pedidos (qualquer status)
#   M (Monetário)  → Apenas pedidos COMPLETED
#
#   Justificativa: um cliente que tentou comprar 3 vezes (mesmo que um pedido
#   tenha sido rejeitado ou esteja pendente) demonstra maior intenção de compra
#   do que um cliente de pedido único. Excluir esses pedidos subavaliaria o
#   engajamento real. Já para valor monetário, só contabilizamos receita
#   efetivamente realizada.
#
# LIMITAÇÃO DOCUMENTADA:
#   No dataset de pedidos COMPLETED, todos os 1.445 clientes têm exatamente
#   1 pedido concluído. A diferença de frequência só aparece quando incluímos
#   pedidos com outros status (131 clientes com 2-3 pedidos totais).
#   Isso limita o poder discriminatório da dimensão F no RFM — a segmentação
#   é fortemente influenciada por R e M.
#
# Input  : master_bikeshop.csv, orders_clean.csv, customers_clean.csv
# Output : rfm_clientes.csv, rfm_segmentos.csv
#          fase_03_clientes_rfm.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

master    = pd.read_csv(PATH_INPUT + "master_bikeshop.csv",    parse_dates=['order_date'])
orders    = pd.read_csv(PATH_INPUT + "orders_clean.csv",       parse_dates=['order_date'])
customers = pd.read_csv(PATH_INPUT + "customers_clean.csv")

completed = master[master['status_label'] == 'Completed'].copy()

print("=" * 60)
print("FASE 03 — COMPORTAMENTO DE CLIENTES E RFM")
print("=" * 60)
print(f"Pedidos concluídos: {completed['order_id'].nunique()}")
print(f"Clientes únicos:    {completed['customer_id'].nunique()}")


# =============================================================================
# SEÇÃO 2 — CONSTRUÇÃO DO RFM
# =============================================================================

# Data de referência = 1 dia após o último pedido registrado (qualquer status)
data_ref = orders['order_date'].max() + pd.Timedelta(days=1)
print(f"\nData de referência RFM: {data_ref.date()}")

# R e F: baseados em TODOS os pedidos
rfm_rec_freq = (orders.groupby('customer_id')
                .agg(recencia   = ('order_date', lambda x: (data_ref - x.max()).days),
                     frequencia = ('order_id', 'nunique'))
                .reset_index())

# M: baseado apenas em pedidos COMPLETED
rfm_monetario = (completed.groupby('customer_id')['revenue']
                 .sum().round(2).reset_index()
                 .rename(columns={'revenue': 'monetario'}))

# Unir tudo
rfm = (rfm_rec_freq
       .merge(rfm_monetario, on='customer_id', how='left')
       .merge(customers[['customer_id', 'first_name', 'last_name', 'state']],
              on='customer_id', how='left'))
rfm['monetario'] = rfm['monetario'].fillna(0)

print("\n--- Distribuição RFM ---")
print(rfm[['recencia', 'frequencia', 'monetario']].describe().round(2))

print("\n--- Frequência de pedidos ---")
print(rfm['frequencia'].value_counts().sort_index()
      .rename_axis('n_pedidos').reset_index(name='clientes'))


# =============================================================================
# SEÇÃO 3 — SCORES E SEGMENTAÇÃO
# =============================================================================
# Cada dimensão é dividida em 4 quartis (score 1 a 4).
# Recência: score INVERTIDO — menor recência (mais recente) = score mais alto.
# Frequência e Monetário: score direto — maior = melhor.

rfm['r_score'] = pd.qcut(rfm['recencia'],
                          q=4, labels=[4, 3, 2, 1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequencia'].rank(method='first'),
                          q=4, labels=[1, 2, 3, 4]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetario'].rank(method='first'),
                          q=4, labels=[1, 2, 3, 4]).astype(int)

def segmentar(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    if   r >= 3 and f >= 3 and m >= 3: return 'Campeões'
    elif r >= 3 and m >= 3:             return 'Fiéis de alto valor'
    elif r >= 3 and f == 1:             return 'Novos promissores'
    elif r <= 2 and f >= 3:             return 'Em risco'
    elif r == 1 and f == 1 and m == 1:  return 'Inativos'
    else:                               return 'Regulares'

rfm['segmento'] = rfm.apply(segmentar, axis=1)

print("\n--- Segmentos RFM ---")
seg = (rfm.groupby('segmento')
       .agg(clientes       = ('customer_id', 'count'),
            receita_total  = ('monetario',   'sum'),
            ticket_medio   = ('monetario',   'mean'),
            recencia_media = ('recencia',    'mean'),
            freq_media     = ('frequencia',  'mean'))
       .round(2).reset_index()
       .sort_values('receita_total', ascending=False))
seg['pct_receita'] = (seg['receita_total'] / seg['receita_total'].sum() * 100).round(1)
print(seg.to_string(index=False))

print("\n--- Top 15 clientes por valor monetário ---")
top_cli = rfm.sort_values('monetario', ascending=False).head(15)
print(top_cli[['first_name', 'last_name', 'state', 'recencia',
               'frequencia', 'monetario', 'segmento']].to_string(index=False))


# =============================================================================
# SEÇÃO 4 — P7: PERFIL GEOGRÁFICO
# =============================================================================

print("\n--- P7: Receita por estado ---")
geo_estado = (completed.groupby('customer_state')
              .agg(receita      = ('revenue',     'sum'),
                   pedidos      = ('order_id',    'nunique'),
                   clientes     = ('customer_id', 'nunique'))
              .sort_values('receita', ascending=False).reset_index())
geo_estado['ticket_medio'] = (geo_estado['receita'] / geo_estado['pedidos']).round(2)
geo_estado['pct']          = (geo_estado['receita'] / geo_estado['receita'].sum() * 100).round(1)
print(geo_estado.to_string(index=False))

print("\n--- P7: Top 10 cidades por receita ---")
geo_cidade = (completed.groupby(['customer_city', 'customer_state'])
              .agg(receita  = ('revenue',  'sum'),
                   pedidos  = ('order_id', 'nunique'))
              .sort_values('receita', ascending=False).head(10).reset_index())
print(geo_cidade.to_string(index=False))

# OBSERVAÇÃO: os 3 estados (NY, CA, TX) correspondem exatamente às 3 lojas.
# Não há evidência de clientes comprando fora do estado da loja — isso sugere
# que a base de clientes é predominantemente local/regional.
# Uma análise de expansão geográfica precisaria de dados adicionais.


# =============================================================================
# SEÇÃO 5 — EXPORTAR TABELAS RFM
# =============================================================================

rfm.to_csv(PATH_OUTPUT + "rfm_clientes.csv",  index=False)
seg.to_csv(PATH_OUTPUT + "rfm_segmentos.csv", index=False)
print("\n✓ rfm_clientes.csv  exportado")
print("✓ rfm_segmentos.csv exportado")


# =============================================================================
# SEÇÃO 6 — VISUALIZAÇÕES
# =============================================================================

COR_SEG = {
    'Campeões':           '#2563EB',
    'Fiéis de alto valor':'#10B981',
    'Regulares':          '#94A3B8',
    'Em risco':           '#F59E0B',
    'Novos promissores':  '#60A5FA',
    'Inativos':           '#EF4444',
}
COR_ESTADO  = {'NY': '#2563EB', 'CA': '#10B981', 'TX': '#F59E0B'}
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

# --- Gráfico 1: Clientes por segmento ---
ax1 = fig.add_subplot(3, 2, 1)
seg_ord = seg.sort_values('clientes', ascending=True)
bars1 = ax1.barh(seg_ord['segmento'], seg_ord['clientes'],
                 color=[COR_SEG[s] for s in seg_ord['segmento']],
                 height=0.6, zorder=3)
for bar, n, pct in zip(bars1, seg_ord['clientes'],
                        seg_ord['clientes'] / seg_ord['clientes'].sum() * 100):
    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{n} ({pct:.0f}%)', va='center', fontsize=9.5,
             color='#374151', fontweight='bold')
estilo_limpo(ax1)
ax1.set_xlim(0, 850)
ax1.set_title('Clientes por segmento RFM', fontsize=13,
              fontweight='bold', pad=12, color='#111827')

# --- Gráfico 2: Receita por segmento ---
ax2 = fig.add_subplot(3, 2, 2)
seg_rec = seg.sort_values('receita_total', ascending=True)
bars2 = ax2.barh(seg_rec['segmento'], seg_rec['receita_total'],
                 color=[COR_SEG[s] for s in seg_rec['segmento']],
                 height=0.6, zorder=3)
for bar, val, pct in zip(bars2, seg_rec['receita_total'], seg_rec['pct_receita']):
    ax2.text(bar.get_width() + 10000, bar.get_y() + bar.get_height()/2,
             f'${val/1e3:.0f}K ({pct:.0f}%)',
             va='center', fontsize=9.5, color='#374151', fontweight='bold')
estilo_limpo(ax2)
fmt_k(ax2, 'x')
ax2.set_xlim(0, 3e6)
ax2.set_title('Receita por segmento RFM', fontsize=13,
              fontweight='bold', pad=12, color='#111827')

# --- Gráfico 3: Scatter recência × valor ---
ax3 = fig.add_subplot(3, 2, (3, 4))
for seg_nome, cor in COR_SEG.items():
    df_s = rfm[rfm['segmento'] == seg_nome]
    ax3.scatter(df_s['recencia'], df_s['monetario'],
                c=cor, label=seg_nome, alpha=0.65, s=35, zorder=3)
estilo_limpo(ax3)
ax3.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax3.set_xlabel('Recência (dias desde último pedido)',
               fontsize=10, color=CINZA_TEXTO)
ax3.set_ylabel('Valor monetário', fontsize=10, color=CINZA_TEXTO)
ax3.set_title('Mapa RFM — recência × valor por cliente',
              fontsize=13, fontweight='bold', pad=12, color='#111827')
ax3.legend(frameon=False, fontsize=9, loc='upper right')
ax3.axvline(x=400, color=CINZA_GRADE, linewidth=1.2, linestyle='--')
ax3.text(405, ax3.get_ylim()[1] * 0.95, 'limiar\nrecência',
         fontsize=8, color=CINZA_TEXTO)

# --- Gráfico 4: Ticket médio por segmento ---
ax4 = fig.add_subplot(3, 2, 5)
seg_tick = seg.sort_values('ticket_medio', ascending=True)
bars4 = ax4.barh(seg_tick['segmento'], seg_tick['ticket_medio'],
                 color=[COR_SEG[s] for s in seg_tick['segmento']],
                 height=0.6, zorder=3)
for bar, val in zip(bars4, seg_tick['ticket_medio']):
    ax4.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', fontsize=9.5,
             color='#374151', fontweight='bold')
estilo_limpo(ax4)
fmt_k(ax4, 'x')
ax4.set_xlim(0, 12000)
ax4.set_title('Ticket médio por segmento', fontsize=13,
              fontweight='bold', pad=12, color='#111827')

# --- Gráfico 5: Receita por estado ---
ax5 = fig.add_subplot(3, 2, 6)
geo_plot = geo_estado.sort_values('receita', ascending=True)
bars5 = ax5.barh(geo_plot['customer_state'], geo_plot['receita'],
                 color=[COR_ESTADO.get(s, '#94A3B8') for s in geo_plot['customer_state']],
                 height=0.4, zorder=3)
for bar, val in zip(bars5, geo_plot['receita']):
    pct = val / geo_plot['receita'].sum() * 100
    ax5.text(bar.get_width() + 20000, bar.get_y() + bar.get_height()/2,
             f'${val/1e6:.2f}M  ({pct:.0f}%)',
             va='center', fontsize=10, color='#374151', fontweight='bold')
estilo_limpo(ax5)
fmt_k(ax5, 'x')
ax5.set_xlim(0, 6.5e6)
ax5.set_title('Receita por estado (perfil geográfico)',
              fontsize=13, fontweight='bold', pad=12, color='#111827')

plt.suptitle('Fase 3 — Comportamento de Clientes e RFM',
             fontsize=15, fontweight='bold', y=1.005, color='#111827')
plt.tight_layout(h_pad=4, w_pad=3)
plt.savefig(PATH_OUTPUT + "fase_03_clientes_rfm.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nGráfico salvo: fase_03_clientes_rfm.png")

# Exportando Arquivo
rfm.to_csv(PATH_OUTPUT + "rfm_clientes.csv", index=False, decimal=',')
seg.to_csv(PATH_OUTPUT + "rfm_segmentos.csv", index=False, decimal=',')
print("Arquivo rfm_clientes.csv Salvo!")

print("\nFase 03 finalizada. Próximo passo: fase_04_equipe_vendas.py")
