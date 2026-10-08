"""
Script de verificación independiente para la Tabla Q* del laberinto (Capítulo 4).
Calcula:
1. Valores teóricos exactos de Bellman Q*(s, a) mediante Iteración de Valor.
2. Estimación empírica mediante simulación de Q-Learning (Watkins).
"""

import numpy as np

def paso_entorno(s, a):
    """
    Entorno Cuadrícula 3x3:
    0: Norte, 1: Sur, 2: Este, 3: Oeste
    S1(0) S2(1) S3(2)
    S4(3) S5(4:Muro) S6(5)
    S7(6) S8(7) S9(8:Meta)
    """
    r, c = divmod(s, 3)
    salio = False
    
    if a == 0:    # Norte
        if r > 0: r -= 1
        else: salio = True
    elif a == 1:  # Sur
        if r < 2: r += 1
        else: salio = True
    elif a == 2:  # Este
        if c < 2: c += 1
        else: salio = True
    elif a == 3:  # Oeste
        if c > 0: c -= 1
        else: salio = True

    if salio:
        return s, -10.0, True  # Game Over por salirse

    s_sig = r * 3 + c
    if s_sig == 4:  # Muro central
        s_sig = s

    if s_sig == 8:  # Meta
        return s_sig, 10.0, True
    else:
        return s_sig, -1.0, False

def calcular_q_teorico_exacto(gamma=0.90, max_iter=1000, tol=1e-8):
    """Resuelve la Ecuación de Optimalidad de Bellman para Q*(s, a)."""
    Q = np.zeros((9, 4))
    
    for _ in range(max_iter):
        Q_ant = Q.copy()
        for s in range(9):
            if s == 8 or s == 4:  # Meta o Muro
                continue
            for a in range(4):
                s_sig, r, done = paso_entorno(s, a)
                if done:
                    Q[s, a] = r
                else:
                    Q[s, a] = r + gamma * np.max(Q_ant[s_sig])
        if np.max(np.abs(Q - Q_ant)) < tol:
            break
    return Q

def simular_q_learning(episodes=30000, alfa=0.10, gamma=0.90, seed=42):
    """Simula el algoritmo de Q-Learning tabular."""
    np.random.seed(seed)
    Q = np.zeros((9, 4))
    epsilon = 1.0
    eps_min = 0.02
    decay = 0.9995

    for ep in range(episodes):
        s = 0
        done = False
        steps = 0
        while not done and steps < 60:
            if np.random.rand() < epsilon:
                a = np.random.randint(4)
            else:
                a = int(np.argmax(Q[s]))
            
            s_sig, r, done = paso_entorno(s, a)
            q_futuro = 0.0 if done else np.max(Q[s_sig])
            target = r + gamma * q_futuro
            Q[s, a] += alfa * (target - Q[s, a])
            s = s_sig
            steps += 1
        epsilon = max(eps_min, epsilon * decay)
    return Q

if __name__ == "__main__":
    nombres_estados = [
        "S1 (Inicio)", "S2 (Pasillo)", "S3 (Pasillo)",
        "S4 (Pasillo)", "S5 (Muro)   ", "S6 (Pasillo)",
        "S7 (Pasillo)", "S8 (Pasillo)", "S9 (Meta)   "
    ]
    acciones = ["Norte (^)", "Sur (v)", "Este (>)", "Oeste (<)"]

    print("=" * 65)
    print("1. VALORES TEÓRICOS EXACTOS DE BELLMAN Q*(s, a)")
    print("=" * 65)
    Q_teorico = calcular_q_teorico_exacto()
    
    header = f"{'Estado':14s} | " + " | ".join(f"{a:9s}" for a in acciones)
    print(header)
    print("-" * len(header))
    for s in range(9):
        if s == 4:
            print(f"{nombres_estados[s]:14s} | {'---':^9s} | {'---':^9s} | {'---':^9s} | {'---':^9s}")
        else:
            fila = " | ".join(f"{Q_teorico[s, a]:+9.2f}" for a in range(4))
            print(f"{nombres_estados[s]:14s} | {fila}")

    print("\n" + "=" * 65)
    print("2. POLÍTICA ÓPTIMA DERIVADA argmax Q*(s, a)")
    print("=" * 65)
    simbolos = {0: "^", 1: "v", 2: ">", 3: "<"}
    for r in range(3):
        fila_str = ""
        for c in range(3):
            s = r * 3 + c
            if s == 8:
                fila_str += "[ META ] "
            elif s == 4:
                fila_str += "[ MURO ] "
            else:
                mejor_a = np.argmax(Q_teorico[s])
                fila_str += f"[  {simbolos[mejor_a]}   ] "
        print(fila_str)
