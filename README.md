# Curso de Aprendizaje por Refuerzo: De la Probabilidad a las Redes Neuronales Profundas

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-PDF%20Book-008080?logo=latex&logoColor=white)](documents/main.tex)

> **Programa Académico Teórico–Práctico (36 Horas)**  
> **Metodología:** Lápiz, Papel y Código $\cdot$ **Caso Conductor:** OptiBot

Bienvenido al repositorio oficial del curso **Aprendizaje por Refuerzo: De la Probabilidad a las Redes Neuronales Profundas**. Este programa ofrece una formación completa y rigurosa —diseñada tanto para principiantes como para ingenieros y científicos de datos— que abarca desde los fundamentos de probabilidad estocástica hasta el aprendizaje por refuerzo profundo moderno (*Deep Q-Networks, DQN*).

---

## 🎯 Filosofía Pedagógica: "Lápiz, Papel y Código"

Cada concepto del curso se desarrolla articulando tres pilares indispensables:

1. **Intuición Conceptual:** Explicaciones claras, analogías físicas y sin jerga innecesaria (*entendible para dummies*, priorizando el razonamiento).
2. **Deducción y Cálculo Manual Paso a Paso:** Ejercicios explícitos resueltos a mano con números reales para comprender el engranaje exacto de las ecuaciones de Bellman y los gradientes.
3. **Implementación Algorítmica Reproducible:** Traducción directa a código limpio en Python (NumPy y PyTorch) en cuadernos interactivos listos para ejecutar.

### 🤖 El Hilo Conductor: OptiBot en el Almacén Inteligente
A lo largo de todo el curso, los algoritmos se prueban y contrastan en un caso industrial unificado: **OptiBot**, un robot móvil autónomo encargado del transporte de carga en un almacén logístico con muelles de carga, obstáculos infranqueables y bahías de entrega, evolucionando desde cuadrículas discretas hasta navegación en coordenadas métricas continuas.

---

## 📚 Estructura y Contenido del Curso

| Capítulo | Título | Temas Clave | Cuaderno Interactivo |
|:---:|:---|:---|:---:|
| **0** | **Probabilidad y Pensamiento Estocástico para RL** | Espacios muestrales, variables aleatorias, esperanza matemática, varianza, probabilidad condicional y Teorema de Bayes aplicado a decisiones bajo incertidumbre. | [`cap00_probabilidad.ipynb`](notebooks/cap00_probabilidad.ipynb) |
| **1** | **El Problema de los Bandidos Multi-Brazo** | Dilema exploración vs. explotación, recompensas estocásticas, regla $\varepsilon$-greedy, límite superior de confianza (*UCB*) y muestreo de Thompson. | [`cap01_bandidos.ipynb`](notebooks/cap01_bandidos.ipynb) |
| **2** | **Cadenas de Markov y Procesos Estocásticos** | Propiedad de Markov, matrices de transición de estados, grafos de transición, distribuciones estacionarias y análisis asintótico del movimiento de OptiBot. | [`cap02_markov.ipynb`](notebooks/cap02_markov.ipynb) |
| **3** | **Procesos de Decisión de Markov y Ecuaciones de Bellman** | Tupla MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, funciones de valor $V(s)$ y $Q(s, a)$, deducción analítica de las Ecuaciones de Bellman y políticas óptimas. | [`cap03_mdp_bellman.ipynb`](notebooks/cap03_mdp_bellman.ipynb) |
| **4** | **Aprendizaje por Refuerzo Tradicional: Algoritmo Q-Learning** | Métodos libres de modelo (*Model-Free*), Diferencia Temporal (TD), deducción de la regla de actualización tabular de Watkins y convergencia en cuadrícula discreta. | [`cap04_rl_tradicional.ipynb`](notebooks/cap04_rl_tradicional.ipynb) |
| **5** | **Aprendizaje por Refuerzo Profundo: Deep Q-Learning (DQN)** | Transición a coordenadas continuas ($\mathbb{R}^4$), la maldición de la dimensionalidad, generalización con redes neuronales, *Experience Replay*, Red Policy vs. Red Target congelada, truco del vector target, pérdida cuadrática (MSE) y semi-gradiente. | [`cap05_rl_profundo.ipynb`](notebooks/cap05_rl_profundo.ipynb) |

---

## 📂 Organización del Repositorio

```text
rl_course/
├── documents/                      # Libro y documentación completa en LaTeX
│   ├── main.tex                    # Documento maestro (libro del curso)
│   ├── cap00_probabilidad.tex       # Cap 0: Probabilidad y pensamiento estocástico
│   ├── cap01_bandidos.tex          # Cap 1: Bandidos multi-brazo
│   ├── cap02_markov.tex            # Cap 2: Cadenas de Markov
│   ├── cap03_mdp_bellman.tex       # Cap 3: MDP y Ecuaciones de Bellman
│   ├── cap04_rl_tradicional.tex    # Cap 4: Q-Learning tabular
│   ├── cap05_rl_profundo.tex       # Cap 5: Deep Q-Learning (DQN)
│   └── figures/                    # Diagramas y recursos gráficos vectoriales
├── notebooks/                      # Cuadernos interactivos Jupyter en Python
│   ├── cap00_probabilidad.ipynb    # Prácticas interactivas Cap 0
│   ├── cap01_bandidos.ipynb        # Prácticas interactivas Cap 1
│   ├── cap02_markov.ipynb          # Prácticas interactivas Cap 2
│   ├── cap03_mdp_bellman.ipynb     # Prácticas interactivas Cap 3
│   ├── cap04_rl_tradicional.ipynb  # Prácticas interactivas Cap 4
│   └── cap05_rl_profundo.ipynb     # Prácticas interactivas Cap 5 (DQN en PyTorch)
└── README.md                       # Presentación y guía del proyecto
```

---

## 🚀 Guía de Inicio Rápido

### 1. Clonar el Repositorio
```bash
git clone https://github.com/cuzco-t/basic-reinforcement-learning-course.git
cd basic-reinforcement-learning-course
```

### 2. Configurar el Entorno de Python
Se recomienda el uso de un entorno virtual (`venv` o `conda`):

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Linux/macOS)
source venv/bin/activate

# Activar entorno (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

Instalar las librerías necesarias:
```bash
pip install numpy matplotlib torch jupyterlab
```

### 3. Ejecutar los Cuadernos Interactivos
Inicia el servidor de Jupyter Lab o Notebook:
```bash
jupyter lab
```
Navega a la carpeta `notebooks/` y abre el capítulo que desees estudiar.

---

## 📖 Compilación del Libro en PDF (LaTeX)

El documento principal del curso está maquetado con LaTeX utilizando tipografía moderna (`helvet`), diagramas vectoriales nativos en `TikZ` y bloques estructurados con `tcolorbox`.

Para compilar el manual completo a PDF:

```bash
cd documents
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # Segunda pasada para generar índice y referencias cruzadas
```

El resultado se generará en `documents/main.pdf`.

---

## 🛠️ Tecnologías Empleadas

* **Lenguaje:** Python 3.10+
* **Cálculo Numérico y Tensores:** NumPy, PyTorch
* **Visualización y Animación:** Matplotlib, IPython Display
* **Edición y Tipografía Matemática:** LaTeX, TikZ, PGF, BibTeX
* **Entorno Interactivo:** Jupyter Lab / Notebooks

---

## 📄 Licencia

Este material se distribuye bajo la licencia **MIT**. Eres libre de usarlo, modificarlo y compartirlo con fines académicos y profesionales citando la fuente original.
