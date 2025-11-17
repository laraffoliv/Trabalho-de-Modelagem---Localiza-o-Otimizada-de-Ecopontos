import pulp
import random

#Dados

# Parâmetro P (Número de Ecopontos a instalar)
P = 5  #

# Parâmetros de Simulação
NUM_PONTOS_DEMANDA = 40  #
NUM_LOCAIS_CANDIDATOS = 15  #

# Índices
pontos_demanda_idx = list(range(NUM_PONTOS_DEMANDA))
locais_candidatos_idx = list(range(NUM_LOCAIS_CANDIDATOS))

# --- DADOS FICTÍCIOS (MOCK) ---

# Coordenadas (x, y) dos 40 pontos de demanda (Centróides dos setores)
coords_demanda = {i: (random.randint(0, 100), random.randint(0, 100)) for i in pontos_demanda_idx}

# Pesos (h_u) dos 40 pontos de demanda (População / Demanda Potencial)

pesos_demanda = {i: random.randint(1, 100) for i in pontos_demanda_idx}

# Coordenadas (x, y) dos 15 locais candidatos (Terrenos viáveis)
coords_candidatos = {j: (random.randint(0, 100), random.randint(0, 100)) for j in locais_candidatos_idx}

# Função para calcular a Distância Retilínea (Manhattan)

def calcular_distancia_manhattan(coord1, coord2):
    """Calcula a distância |x1 - x2| + |y1 - y2|."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# 1. Calcular a Matriz de Distâncias (d_uj)
distancias = {
    (i, j): calcular_distancia_manhattan(coords_demanda[i], coords_candidatos[j])
    for i in pontos_demanda_idx
    for j in locais_candidatos_idx
}

# 2. Calcular a Matriz de Custo Ponderado (h_u * d_uj)
# Este é o custo que realmente queremos minimizar
custos_ponderados = {
    (i, j): pesos_demanda[i] * distancias[(i, j)]
    for i in pontos_demanda_idx
    for j in locais_candidatos_idx
}

# Modelo

# Criação do problema de minimização
modelo = pulp.LpProblem("PMP_Ecopontos_Pampulha", pulp.LpMinimize)

# --- Variáveis de Decisão ---

# x_j: 1 se o Ecoponto j for aberto, 0 caso contrário (Binária)
x = pulp.LpVariable.dicts("Ecoponto_Aberto", locais_candidatos_idx, 0, 1, pulp.LpBinary)

# y_ij: 1 se o ponto de demanda i for alocado ao Ecoponto j, 0 caso contrário (Binária)
y = pulp.LpVariable.dicts("Alocacao",
                         [(i, j) for i in pontos_demanda_idx for j in locais_candidatos_idx],
                         0, 1, pulp.LpBinary)

# --- Função Objetivo ---
# Minimizar a Soma Total das Distâncias Ponderadas
modelo += pulp.lpSum(
    custos_ponderados[(i, j)] * y[(i, j)]
    for i in pontos_demanda_idx
    for j in locais_candidatos_idx
), "Custo_Logistico_Total_Ponderado"

# --- Restrições ---

# Restrição 1: Abrir exatamente P Ecopontos

modelo += pulp.lpSum(x[j] for j in locais_candidatos_idx) == P, "Abrir_Exatamente_P_Ecopontos"

# Restrição 2: Cada ponto de demanda deve ser alocado a EXATAMENTE um Ecoponto
for i in pontos_demanda_idx:
    modelo += pulp.lpSum(y[(i, j)] for j in locais_candidatos_idx) == 1, f"Alocacao_Unica_Demanda_{i}"

# Restrição 3: Um ponto de demanda i só pode ser alocado a um Ecoponto j se ele estiver aberto (x_j=1)
for i in pontos_demanda_idx:
    for j in locais_candidatos_idx:
        # y_ij <= x_j
        modelo += y[(i, j)] <= x[j], f"Alocacao_So_Se_Aberto_{i}_{j}"

# --- Solução ---

print("Iniciando o solver (CBC)...")

modelo.solve()
print("Solução encontrada.")

# --- Impressão dos Resultados ---

print(f"\n--- Status da Solução ---")
# Aqui usamos o dicionário pulp.LpStatus para "traduzir" o status numérico para texto
print(f"Status: {pulp.LpStatus[modelo.status]}")

# *** LINHA CORRIGIDA ABAIXO ***
# Usamos a constante pulp.LpStatusOptimal (sem ponto no meio)
if modelo.status == pulp.LpStatusOptimal:
    print(f"\n--- Resultado Ótimo ---")
    custo_total = pulp.value(modelo.objective)
    print(f"Custo Logístico Total Ponderado Mínimo: {custo_total:,.2f}")

    print(f"\n--- Ecopontos Selecionados (P={P}) ---")
    ecopontos_abertos = [j for j in locais_candidatos_idx if x[j].varValue > 0.5]
    print(f"IDs dos locais candidatos selecionados: {ecopontos_abertos}")

    print(f"\n--- Alocação (Exemplo dos 5 primeiros pontos) ---")
    for i in pontos_demanda_idx[:5]: # Mostrando apenas os 5 primeiros
        for j in ecopontos_abertos:
            if y[(i, j)].varValue > 0.5:
                print(f"  Ponto de Demanda {i} (Peso: {pesos_demanda[i]}) -> Alocado ao Ecoponto {j}")
                break
else:
    print("Não foi encontrada uma solução ótima.")