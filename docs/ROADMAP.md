# ROADMAP — Proyecto CoilStellaration (`coilgen`)

> Documento vivo compartido entre Miguel y Claude. Versión 1.1 — 2026-11-12.
> Idioma: este documento y `docs/BITACORA.md` en español; código, commits, `README.md` y `CLAUDE.md` en inglés.
> Las afirmaciones marcadas con **(a verificar)** son hipótesis de planificación, no hechos medidos.

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Análisis estratégico del reto](#2-análisis-estratégico-del-reto)
3. [Visión general de fases](#3-visión-general-de-fases)
4. [Fases en detalle](#4-fases-en-detalle)
5. [Asignación de modelos Claude](#5-asignación-de-modelos-claude)
6. [Prácticas transversales](#6-prácticas-transversales)
7. [Hitos, calendario y decisiones go/no-go](#7-hitos-calendario-y-decisiones-gono-go)
8. [Portfolio y carrera](#8-portfolio-y-carrera)
9. [Supuestos y preguntas abiertas](#9-supuestos-y-preguntas-abiertas)
10. [Cómo usamos este documento](#10-cómo-usamos-este-documento)
11. [Registro de cambios](#11-registro-de-cambios)

---

## 1. Resumen ejecutivo

### 1.1 Objetivo

Atacar el benchmark **CoilStellaration** (Proxima Fusion, arXiv:2611.12689, NeurIPS 2026 E&D) — generar, a partir de una frontera de plasma y unos requisitos de ingeniería, un conjunto de bobinas filamentarias que los cumpla — con rigor de investigación, aprendiendo física de fusión y ML por el camino, y dejando un portfolio público que sirva de carta de presentación ante la industria de fusión + IA (objetivo aspiracional: Proxima Fusion).

### 1.2 Definición de éxito (por capas)

Cada capa tiene valor por sí misma; no alcanzar la siguiente no invalida las anteriores.

| Capa | Éxito | Evidencia medible |
|---|---|---|
| L1 Aprendizaje | Miguel puede explicar sin notas el problema de stage 2, las 4 métricas, el scoring y por qué fallan los baselines | Notas propias en `docs/notes/`, "test de Feynman" al cierre de cada fase registrado en BITACORA |
| L2 Baseline reproducible | Baselines del paper reproducidos y evaluados con un harness propio | Resultados dentro de la variación entre seeds del paper (Tbl. 1–2); `make reproduce-baseline` funciona en una máquina limpia |
| L3 Single-pass puro | Un modelo de **un solo forward pass sin física en el grafo** supera a todos los baselines publicados en el leaderboard | Fixed-shape: S medio y strict por encima de los baselines de Proxima (≈0.177 / ≈5.1 % en el board en vivo) **y** del baseline de círculos (6.6 % strict). Objetivo tentativo: S̄ ≥ 0.30 y strict ≥ 10 %; stretch: strict > 0 % en el estrato tight |
| L4 Híbrido competitivo | Un enfoque aprendido + refinamiento, claramente etiquetado, competitivo con la mejor entrada híbrida (0.961 / 78.8 %) | S̄ ≥ 0.9 y strict ≥ 70 % en fixed-shape, con coste de inferencia reportado; idealmente igual calidad con menos cómputo por instancia |
| L5 Difusión pública | Repo público, informe técnico estilo arXiv y posts | Repo con CI verde y resultados reproducibles; informe en `docs/reports/final/`; contacto respetuoso con los autores |

Los umbrales numéricos de L3/L4 son tentativos y se revisan en cada hito, porque el leaderboard se mueve.

### 1.3 Principios rectores

1. **Rigor antes que velocidad.** Ningún número entra en un informe sin seeds, intervalo de confianza y commit asociado.
2. **Reproducibilidad.** Todo resultado se regenera con un comando a partir de un config versionado, una revisión fijada del dataset y un entorno bloqueado.
3. **Reporte honesto.** Se publican también los resultados negativos; el tipo de método (puro vs. híbrido) se etiqueta siempre; no se entrena con el split de evaluación sin declararlo.
4. **Conciencia de cómputo.** Tres niveles: máquina local para desarrollo, Colab para prototipos y las GPUs del grupo de investigación de la universidad para el trabajo serio (§6.1). El hardware local no limita el alcance de los modelos ni de los experimentos; aun así, cada barrido caro lleva una estimación de GPU-horas antes de lanzarse, porque el recurso es compartido.
5. **Entender, no solo delegar.** Claude acelera; Miguel debe poder defender cada decisión en una entrevista técnica.
6. **Física primero.** El score es física evaluada con DESC; cualquier atajo de ML que no respete esa física no sirve.

---

## 2. Análisis estratégico del reto

### 2.1 Anatomía del score: dónde se pierde y se gana

El score por instancia es `S = (∏ s_i)^(1/4)` con `s_i = exp(−v_i/α_i)`, es decir:

```
−ln S = ¼ · Σ_i v_i / α_i        α = 0.10 (d_CC, d_CP, κ) ; α = 0.25 (ε_max)
```

Consecuencias directas:

- **El score es una función de pérdida ya escrita.** `Σ v_i/α_i` es una suma de hinges normalizados, diferenciable casi en todas partes. Si tenemos las 4 métricas en JAX, podemos entrenar directamente contra el criterio del benchmark en lugar de contra un proxy (MSE en coeficientes).
- **Umbral de S ≥ 0.9:** `Σ v_i/α_i ≤ 0.42`. Si solo falla el error de campo, se tolera un exceso de ~10.5 % sobre el objetivo; si solo falla una métrica geométrica, ~4.2 %.
- **Por qué los baselines puntúan ~0.1:** con violación mediana de error de campo 2.1–2.9, `s_ε ≈ 1e−5…2e−4`; aunque las otras tres métricas sean perfectas, `S ≈ 0.06–0.12`. El S medio es, en la práctica, "fracción de instancias casi factibles": la mediana es ~1e−3.
- **Factibilidad estricta = dominar a la referencia.** Los objetivos son las métricas que alcanzó el coilset de referencia de DESC, así que la referencia está *exactamente* en la frontera en las 4 métricas. Ser estrictamente factible (v_i ≤ 1e−3) exige igualar o mejorar a un coilset optimizado en las 4 a la vez. Una regresión que "apunta a la referencia" cae a un lado u otro de la frontera por azar; hay que apuntar **dentro** de la región factible, con margen.
- **El leaderboard ordena por strict y luego por media**; el paper, por media. Optimizamos para ambos, pero el entrenamiento debe incluir margen explícito para que el strict no dependa del ruido numérico.

### 2.2 Dónde está el margen de mejora

| Observación | Implicación |
|---|---|
| El error de campo domina el fallo; ensembles ya clavan d_CP y curvatura; d_CC se viola a menudo | El trabajo de mayor palanca es bajar ε sin romper d_CC. Las dos son las que hay que vigilar juntas |
| El problema inverso es no único y el MSE promedia modos | La media de dos coilsets válidos no es un coilset válido: ε es extremadamente sensible a la geometría. Hace falta (a) eliminar la multimodalidad artificial y (b) modelar o evitar la real |
| Estrato loose casi trivial: círculos planos sin aprendizaje logran 6.6 % strict | Hay que medir una "escalera de baselines triviales" antes de atribuir mérito a un modelo; y reportar siempre por estrato |
| Estrato tight con 0 % strict para todo modelo puro | Es donde un resultado no nulo de un modelo single-pass sería noticia |
| El optimizador LM in-graph alcanza 78.8 % strict | Las instancias son alcanzables desde un buen punto inicial con ~150 pasos de un método de segundo orden: el paisaje local es benigno; el reto es la precisión, no la existencia |
| La entrada no incluye equilibrio ni campo objetivo, solo la frontera | El modelo debe inferir implícitamente la física del campo vacío a partir de la geometría; la pérdida física le da esa señal |

**Multimodalidad artificial (gauge) vs. real.** Parte de la no-unicidad es puramente de representación y se puede eliminar gratis:

- permutación del orden de las bobinas;
- origen del parámetro `t` de cada curva (desplazar `t` cambia todos los coeficientes de Fourier sin cambiar la geometría);
- orientación de la curva (invertirla y cambiar el signo de la corriente);
- escala global de corrientes: `ε = |B·n|/|B|` es invariante a ella, solo importan las corrientes relativas **(a verificar** contra el evaluador, que podría aplicar su propia normalización).

Comprobar si el dataset ya es canónico en estas simetrías es una tarea de EDA con consecuencias directas para el diseño de pérdida y de salida.

### 2.3 Apuestas técnicas

Ordenadas por relación valor/riesgo. Cada una tiene un experimento barato que la valida o la descarta.

| # | Apuesta | Por qué puede funcionar | Experimento de validación barato | Riesgo principal |
|---|---|---|---|---|
| A1 | **Métricas del benchmark en JAX, rápidas y diferenciables** (`coilgen.physics`), validadas contra las métricas DESC de upstream | Habilita todo lo demás: pérdida física, evaluación local barata, selección de candidatos, refinamiento. DESC ya es diferenciable, pero no está pensado para `vmap` sobre miles de instancias ni para exportarse a ONNX | Paridad numérica en 200 instancias; throughput en CPU/GPU | Paridad de ε_max y de mínimos/máximos a nivel 1e−4; coste de memoria de Biot–Savart |
| A2 | **Entrenamiento physics-informed contra el propio score**, con margen | Ataca el modo de fallo dominante y no sufre el promediado de modos: la pérdida física acepta cualquier coilset válido | Fine-tune del baseline reproducido con pérdida física; medir ΔS por estrato | Mínimos locales, gradientes mal condicionados (min/max), coste por paso |
| A3 | **Canonicalización y pérdidas invariantes a la reparametrización** | Quita multimodalidad artificial sin coste en inferencia | Varianza de coeficientes entre coilsets "iguales"; MSE geométrico (curvas remuestreadas) vs. MSE en coeficientes | Que el dataset ya sea canónico (apuesta sin ganancia, pero barata) |
| A4 | **Hindsight relabeling y aumento de datos por física** | Cualquier coilset `c` con métricas `g(c)` es un par de entrenamiento válido `(b, r = g(c)) → c`. Permite reusar los coilsets REGCOIL y los DESC filtrados, y generar pares con el propio modelo | Entrenar con y sin los pares reetiquetados | Desplazamiento de distribución respecto a los requisitos de evaluación |
| A5 | **Modelos condicionales multimodales** (multi-hipótesis con selector aprendido, flow matching / difusión destilada a pocos pasos) | Modelan la multimodalidad real en lugar de promediarla | *Oráculo best-of-K*: si elegir el mejor de K candidatos baratos sube mucho el score, hay margen | Complejidad, coste de entrenamiento, exportación a ONNX, y que A2 ya resuelva el problema |
| A6 | **Parametrizaciones de salida con sesgo físico** (residuo sobre una plantilla; curvas sobre una superficie de bobinado a distancia d_CP) | Algunas restricciones se cumplen por construcción y el modelo solo aprende correcciones | Ajustar los coilsets de referencia con la parametrización y medir el error de representación | Pérdida de expresividad; curvatura mal controlada |
| A7 | **Retrieval** de coilsets de entrenamiento cercanos como condición o punto de partida | Los vecinos más próximos son un prior fuerte y multimodal; caben en 50 MiB en fp16 (ya demostrado por terceros) | Score del vecino más próximo sin modificar (parte de la escalera de baselines) | Etiquetado: un dataset embebido en el grafo ya no es "solo pesos" |
| A8 | **Refinamiento amortizado / unrolled** (K pasos fijos de Gauss–Newton/LM con hiperparámetros aprendidos, entrenado extremo a extremo) | Combina un buen punto inicial con unas pocas correcciones informadas por la física; el estudio de warm-start del paper ya muestra que el inicio aprendido acelera a DESC | Curva score vs. nº de pasos de refinamiento desde distintos inicios, offline en JAX | Ambigüedad de reglas; escribir Biot–Savart y jacobianos en ONNX |

**Hilo conductor para el write-up:** la contribución no es un punto, sino la **frontera de Pareto calidad vs. cómputo de inferencia** — desde un forward pass puro hasta N pasos de refinamiento — con el mismo harness y las mismas instancias. Esto convierte la ambigüedad de las reglas en un eje de análisis en lugar de en un problema.

### 2.4 Ambigüedad de las reglas y postura del proyecto

El paper exige "un solo forward pass, sin optimización costosa en inferencia", pero el contrato real es solo "un `model.onnx` ≤ 50 MiB". Una entrada de terceros embebe retrieval + un optimizador Levenberg–Marquardt de ~150 pasos con `Loop` de ONNX (1.5–2.7 s/instancia en CPU) y logra 0.961 / 78.8 %. Cumple la letra del contrato y contradice el espíritu del paper. Es previsible que los organizadores aclaren las reglas.

**Postura:** perseguir dos tracks, etiquetados sin ambigüedad en nombres de repos, model cards e informes.

| Track | Definición operativa | Repos HF (propuesta) |
|---|---|---|
| **A — Amortizado puro** | El grafo de inferencia es una red de profundidad fija. **No evalúa métricas físicas** (ni Biot–Savart, ni distancias) y **no contiene datos de entrenamiento embebidos** más allá de los pesos | `<user>/coilgen-a-*` |
| **B — Híbrido** | Cualquier cosa que evalúe física en el grafo, itere sobre un objetivo o embeba datos de entrenamiento. Se reporta siempre el coste: nº de evaluaciones de Biot–Savart, pasos y tiempo por instancia | `<user>/coilgen-b-*` |

Zonas grises y cómo las tratamos:

- *Selector aprendido entre K hipótesis* (sin física en el grafo) → Track A.
- *Muestreador generativo de pocos pasos sin física* (p. ej. flow destilado a 1–4 evaluaciones de la red) → Track A, declarando el número de evaluaciones.
- *Selector que evalúa Biot–Savart en el grafo para elegir entre K* → Track B.
- *Retrieval con base de datos embebida* → Track B.
- *Entrenar con las entradas del split de evaluación usando la pérdida física* (no requiere etiquetas; aprendizaje transductivo) → **no se hace** en ningún track salvo que los organizadores lo permitan explícitamente; si se explora, es un experimento aparte y etiquetado.

**Acción:** Miguel publica una pregunta breve y respetuosa en la discusión del Space del leaderboard (borrador en §8.3) durante la Fase 0–1. La respuesta puede reordenar las fases 6 y 7. Si el Track B se prohíbe, su trabajo se reconvierte en un estudio offline "score vs. iteraciones de refinamiento" análogo al estudio de warm-start del paper, que sigue siendo publicable.

---

## 3. Visión general de fases

| Fase | Nombre | Esfuerzo (tiempo parcial, ~8–10 h/sem) | Depende de | Modelo principal |
|---|---|---|---|---|
| 0 | Setup y fundamentos | 1.5 semanas | — | Sonnet |
| 1 | EDA y anatomía del score | 2 semanas | 0 | Sonnet (+ Opus para análisis) |
| 2 | Reproducción de baselines, harness y primer envío | 3 semanas | 1 | Opus / Sonnet |
| 3 | Revisión de literatura (en paralelo con 0–2) | ~6 sesiones repartidas | — | Fable |
| 4 | Métricas diferenciables en JAX | 4 semanas | 2 | Opus (+ Fable para numérica sutil) |
| 5 | Track A v1: entrenamiento physics-informed | 5 semanas | 4 | Opus |
| 6 | Track A v2: modelos avanzados y multimodalidad | 6 semanas | 3, 5 | Fable (diseño) / Opus (implementación) |
| 7 | Track B: retrieval y refinamiento | 4 semanas | 4, 5 | Opus (+ Fable) |
| 8 | Evaluación final, ablations y envíos definitivos | 2 semanas | 5–7 | Opus / Sonnet |
| 9 | Write-up y difusión | 3 semanas | 8 | Fable (revisión) / Sonnet (redacción) |

Una "sesión" = 2–3 h de trabajo conjunto. Las fases 6 y 7 pueden intercambiarse según la respuesta de los organizadores y los resultados de la fase 5.

---

## 4. Fases en detalle

### Fase 0 — Setup y fundamentos

**Objetivo.** Entorno reproducible, repo profesional y base conceptual mínima para leer el paper con criterio.

**Por qué importa.** Los errores de entorno (DESC, JAX, REGCOIL, NetCDF) y de comprensión del problema son los que más tiempo hacen perder después.

**Tareas**

- [ ] Entorno `uv` con Python 3.12; `uv.lock` commiteado; upstream como submódulo fijado en `external/coilstellaration` e instalado como dependencia editable.
- [ ] Verificar que `import desc`, `import coilstellaration` y un cálculo de métricas sobre una fila funcionan **en Colab y en el cluster**. En local está comprobado que no es posible: la CPU (Intel Celeron N2840) no tiene AVX, así que jaxlib no arranca (`RuntimeError` en el import y SIGILL si se fuerza). Todo lo que use JAX/DESC se ejecuta fuera de la máquina local. Documentar tiempos y pico de memoria.
- [ ] **Acceso al cómputo del grupo de la universidad:** averiguar y documentar en `docs/COMPUTE.md` el modo de acceso (SLURM o SSH directo), GPUs disponibles, cuotas y normas de uso, almacenamiento (home/scratch), salida a internet desde los nodos y si se admiten contenedores (Apptainer/Docker). Confirmar con el responsable del grupo que este proyecto puede usar el recurso.
- [ ] Replicar el entorno en el cluster (`uv sync` desde el lock, o imagen de contenedor a partir del Dockerfile de upstream si `uv` no es viable) y ejecutar allí `scripts/check_env.py` en GPU.
- [ ] Plantillas de lanzamiento: `scripts/cluster/{train.sbatch,eval_array.sbatch}` (o equivalentes `ssh` + `tmux` si no hay planificador) y un notebook de arranque para Colab que clona el repo e instala el entorno.
- [ ] Decidir si se compila REGCOIL en local (gfortran/cmake presentes) o se pospone: solo es necesario para regenerar warm starts, no para el benchmark.
- [ ] `pyproject.toml` con `ruff`, `pytest`, `pre-commit`; estructura `src/coilgen/` con paquete importable y un test trivial.
- [ ] CI mínima en GitHub Actions: lint + tests rápidos (sin DESC pesado).
- [ ] `README.md` (inglés) con objetivo, estado y cómo reproducir; `CLAUDE.md` con convenciones del proyecto; `docs/BITACORA.md` iniciada.
- [ ] Cuentas: Hugging Face (token de lectura/escritura), Weights & Biases, GitHub (repo público desde el principio o privado hasta M2: decisión de Miguel).
- [ ] Lectura completa del paper (23 páginas) con notas propias; lista de dudas en `docs/notes/00_paper_dudas.md`.
- [ ] Publicar la pregunta sobre reglas en el Space del leaderboard (§8.3).
- [ ] Capturar el estado actual del leaderboard en `docs/reports/leaderboard_snapshots/2026-11.md`.

**Entregables.** `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `README.md`, `CLAUDE.md`, `docs/BITACORA.md`, `docs/COMPUTE.md`, `docs/notes/00_paper_resumen.md`, `scripts/check_env.py`, `scripts/cluster/`, `notebooks/00_colab_bootstrap.ipynb`.

**Definition of Done.** `uv sync && uv run pytest` pasa en una máquina limpia; CI verde; `scripts/check_env.py` calcula las 4 métricas de un coilset del dataset usando upstream, en local y en el cluster (o, si el acceso aún no está listo, en Colab, con el acceso al cluster como tarea abierta con fecha); Miguel explica en 5 minutos, sin notas, qué entra y qué sale del modelo y cómo se puntúa.

**Aprendizaje para Miguel.**
- Stellarators: por qué campos 3D, superficies de flujo, transformada rotacional, stage 1 vs. stage 2, omnigenidad/QI. Lecturas: Imbert-Gérard, Paul & Wright, *An Introduction to Stellarators* (arXiv:1908.05360), caps. iniciales; Helander (2014), *Theory of plasma confinement in non-axisymmetric magnetic fields* (lectura selectiva); charlas introductorias de M. Landreman.
- Ejercicio a mano: campo en el eje de una espira circular por Biot–Savart, y comprobarlo numéricamente en un notebook propio.
- Herramientas: `uv`, `git` con ramas, Conventional Commits.

**Riesgos y mitigación.** JAX/DESC no corren en local (CPU sin AVX, confirmado) → todas las comprobaciones con JAX/DESC van a Colab o al cluster desde el primer día; en local solo código, tests sin JAX y EDA con pandas/pyarrow. El acceso al cluster se retrasa o tiene restricciones (sin internet en nodos, sin `uv`) → Colab cubre las fases 0–2; preparar caché de HF y wheels en el nodo de login, o contenedor. Instalación de `constellaration`/`vmecpp` falla → no es necesaria para el benchmark si solo se usan fronteras ya calculadas; aislarla como extra opcional.

**Modelos.** Sonnet para scaffolding, CI y docs. Haiku para búsquedas en el repo upstream. Miguel lee el paper él mismo; Claude (Opus) responde dudas, no resume en su lugar.

---

### Fase 1 — EDA y anatomía del score

**Objetivo.** Conocer el dataset y el benchmark mejor que nadie fuera de Proxima: distribuciones, simetrías, convenciones, y qué fracción del score es "gratis".

**Por qué importa.** Varias apuestas (A3, A4, A6, A7) se validan o descartan aquí con coste casi nulo, y la escalera de baselines triviales es la vara de medir honesta del resto del proyecto.

**Tareas**

- [ ] Descargar el subset `results` (0.3 GB) y `requirements`; fijar la revisión de HF (commit hash) en `configs/data.yaml`. No descargar `metrics` (213 GB).
- [ ] Loader `coilgen.data` que une con las fronteras de `proxima-fusion/constellaration` por `constellaration_boundary_id` y cachea en parquet/npz local.
- [ ] Reconstruir exactamente los tracks: fixed (53,206 / 2,771) y variable (155,850 / 8,058) y los estratos tight/medium/loose (umbrales 0.31 % y 1.25 %). Los recuentos deben coincidir con el paper.
- [ ] Documentar convenciones: layout de coeficientes `[sin N..sin 1, const, cos 1..cos N]`, simetría de periodo y de stellarator, normalización R = 1 m y B₀ = 1 T, definición del radio menor `a`.
- [ ] Distribuciones y correlaciones de las 4 métricas y de los requisitos por N_FP, N_C y estrato; reproducir cualitativamente Fig. 3–4 del paper.
- [ ] **Simetrías gauge:** ¿orden de bobinas canónico (ángulo toroidal)? ¿origen de `t` consistente? ¿orientación consistente? ¿corrientes iguales dentro de un coilset? Cuantificar.
- [ ] **Multimodalidad real:** para fronteras con varios coilsets, distancia geométrica entre ellos en función de la distancia entre sus requisitos. ¿Requisitos parecidos ⇒ bobinas parecidas?
- [ ] Espectro de Fourier: energía por modo de fronteras y bobinas; motivar (o cuestionar) el escalado espectral 1.2^q de los baselines.
- [ ] **Escalera de baselines triviales** sobre un subconjunto estratificado del split de validación propio: círculos planos (réplica de la entrada de terceros), coilset medio del train, vecino más próximo en (frontera, requisitos), coilset REGCOIL de la misma muestra. Score por estrato de cada uno.
- [ ] Sensibilidad: perturbar coilsets de referencia con ruido gaussiano en coeficientes y medir cuánto cae S. Da la **precisión objetivo** que necesita un modelo (¿error relativo 1e−2? ¿1e−3?).
- [ ] Definir el **split de validación propio**: 5 % de las fronteras de train por hash, estratificado; el split oficial de evaluación queda como test y se usa con moderación.

**Entregables.** `src/coilgen/data/{loading,tracks,splits}.py`; `notebooks/01_dataset_overview.ipynb`, `02_symmetries_gauge.ipynb`, `03_multimodality.ipynb`, `04_trivial_baselines.ipynb`, `05_sensitivity.ipynb`; `docs/reports/01_eda.md`; `configs/data.yaml`.

**Definition of Done.** Recuentos de tracks y estratos idénticos al paper; tabla de la escalera trivial con score por estrato; respuesta escrita (sí/no + evidencia) a cada pregunta de simetría; curva de sensibilidad; decisión registrada en BITACORA sobre A3 (canonicalizar o no).

**Aprendizaje para Miguel.** Representación de Fourier de superficies y curvas (`FourierRZToroidalSurface`, `FourierXYZCoil` en la documentación de DESC); simetría de stellarator; qué significan físicamente d_CP, d_CC y κ (acceso, blanket, límites de HTS). Lectura: Kappel, Landreman & Malhotra (2024) sobre la longitud de escala del gradiente magnético y por qué ciertos plasmas exigen bobinas cercanas; secciones 2 y apéndices D–E del paper. Miguel hace él mismo al menos dos de los notebooks (01 y 05), con Claude como revisor.

**Riesgos y mitigación.** Memoria local insuficiente para joins → procesar por lotes con `pyarrow`/`polars` en streaming, o hacer el join una vez en Colab o en el cluster y guardar el artefacto. Convenciones mal entendidas → test de ida y vuelta: reconstruir un coilset con DESC y comparar métricas con las del dataset.

**Modelos.** Sonnet para loaders y notebooks; Opus para interpretar los resultados de multimodalidad y sensibilidad y decidir consecuencias; Haiku para inspección masiva de esquemas.

---

### Fase 2 — Reproducción de baselines, harness de evaluación y primer envío

**Objetivo.** Reproducir los baselines del paper, tener un harness local equivalente al evaluador oficial y completar un envío de extremo a extremo.

**Por qué importa.** Sin baseline reproducido no hay comparación creíble; sin harness fiable se optimiza a ciegas; el primer envío descubre los problemas de exportación ONNX cuando todavía son baratos.

**Tareas**

- [ ] Ejecutar `run_model_training` de upstream para MLP y ResMLP (fixed) y PerCoilMLP (variable), 3 seeds, en el cluster del grupo (Colab como alternativa para la primera prueba). Registrar en W&B.
- [ ] Evaluar con `run_benchmark_scoring` de upstream y con el checkpoint incluido; comparar con Tbl. 1–2.
- [ ] `coilgen.eval`: harness propio que envuelve el scoring de upstream, añade desglose por estrato y por métrica, bootstrap de intervalos de confianza y comparación pareada entre modelos.
- [ ] **Medir el coste de la evaluación con DESC** por instancia (tiempo y RAM, CPU local vs. nodo del cluster, CPU vs. GPU) y montar la evaluación de un split completo como *job array* paralelo. Este número fija la planificación de evaluaciones de todo el proyecto.
- [ ] Definir conjuntos de evaluación: `dev-300` (300 instancias estratificadas del split de validación propio, para uso diario), `val-full`, y `test` (split oficial, uso contado).
- [ ] Caché de métricas por hash de (coilset, instancia) en `outputs/cache/`.
- [ ] Exportación a ONNX: estudiar cómo lo hace upstream; test de ida y vuelta JAX ↔ `onnxruntime` (diferencia < 1e−5); validar el contrato de entradas/salidas y metadatos (`supported_tracks`).
- [ ] **Primer envío ("plumbing")** de un baseline reproducido al leaderboard, etiquetado como reproducción. Comprobar que el score oficial coincide con el local.
- [ ] Reentrenar un baseline con el pipeline propio (`coilgen.training`) y confirmar que iguala al de upstream: a partir de aquí se itera sobre código propio.
- [ ] Diagnóstico del baseline: desglose de violaciones por métrica y estrato, casos peores, visualización 3D de aciertos y fallos.

**Entregables.** `src/coilgen/eval/{harness,stats,cache}.py`, `src/coilgen/export/onnx.py`, `src/coilgen/training/`, `src/coilgen/models/baselines.py`, `scripts/{train,evaluate,export_onnx,submit}.py`, `configs/baselines/*.yaml`, `notebooks/06_baseline_diagnostics.ipynb`, `docs/reports/02_baselines.md`, tests de exportación.

**Definition of Done.** S̄ y strict reproducidos dentro de ±2σ entre seeds del paper para al menos un modelo por track; score oficial del envío vs. local con diferencia < 0.005 en S̄; coste por instancia de DESC documentado; `dev-300` evaluable de principio a fin con un comando.

**Aprendizaje para Miguel.** JAX (transformaciones `jit`/`grad`/`vmap`, PRNG explícito) y Flax NNX; tutoriales de DESC sobre optimización de bobinas; cómo funciona REGCOIL (Landreman 2017) y por qué deja bobinas bien separadas pero cerca del plasma; ONNX y `onnxruntime`. Flujo de trabajo en cluster (SLURM, job arrays, entornos reproducibles). Miguel lanza él mismo los entrenamientos en el cluster y hace el envío.

**Riesgos y mitigación.** No se reproducen los números → comparar primero con el checkpoint incluido para separar "entrenamiento" de "evaluación". Evaluación DESC demasiado cara → es precisamente la motivación de la Fase 4; mientras tanto, `dev-300` y caché. Exportación JAX→ONNX frágil → alternativa: reescribir solo el grafo de inferencia con `onnxscript` o portar pesos a un módulo equivalente; decidirlo aquí, no en la Fase 5.

**Modelos.** Opus para el harness, la estadística y la exportación; Sonnet para scripts, configs y el informe; Haiku para resumir logs de entrenamiento.

---

### Fase 3 — Revisión de literatura (en paralelo con las fases 0–2)

**Objetivo.** Un mapa del estado del arte que alimente decisiones de diseño concretas en las fases 5–7, no una bibliografía decorativa.

**Por qué importa.** Evita reinventar lo ya resuelto en optimización de bobinas y en optimización amortizada, y da el vocabulario para el informe y las entrevistas.

**Tareas**

- [ ] Bloque 1 — Bobinas de stellarator: NESCOIL (Merkel 1987), REGCOIL (Landreman 2017), FOCUS (Zhu et al. 2018), optimización con SIMSOPT (Landreman et al. 2021; Wechsung et al. 2022), QUASR (Giuliani et al.), DESC (Dudt & Kolemen 2020; Panici/Conlin et al. 2023), optimización single-stage, bobinas de sección finita.
- [ ] Bloque 2 — Contexto de Proxima: ConStellaration (Cadena et al. 2025), concepto Stellaris, y el propio CoilStellaration con apéndices.
- [ ] Bloque 3 — Optimización amortizada y learning-to-optimize: tutorial de Amos sobre amortized optimization; survey L2O (Chen et al. 2022); deep unrolling (Monga et al. 2021); restricciones duras en redes (DC3, Donti et al. 2021); surrogates auto-supervisados por objetivo en problemas inversos de ingeniería.
- [ ] Bloque 4 — Modelos generativos condicionales: DDPM, flow matching (Lipman et al. 2023), rectified flow y destilación a pocos pasos, classifier-free guidance, guiado por física (diffusion posterior sampling), multiple-choice learning / winner-takes-all.
- [ ] Bloque 5 — Geometría y simetrías: redes equivariantes y sobre conjuntos (Deep Sets, Set Transformer), representaciones de curvas, operadores neuronales.
- [ ] Bloque 6 — ML en diseño de fusión: surrogates de equilibrio y de bobinas, trabajos previos de generación de bobinas con ML.
- [ ] Ficha por paper (problema, método, resultado, relevancia para nosotros, idea accionable) en `docs/literature/`.
- [ ] Síntesis final: tabla "idea → apuesta → fase → experimento mínimo" y revisión de la lista de apuestas de §2.3.

**Entregables.** `docs/literature/<bloque>/<paper>.md`, `docs/literature/references.bib`, `docs/reports/03_literature_review.md`.

**Definition of Done.** ≥ 30 fichas; cada apuesta A1–A8 tiene al menos dos referencias de apoyo o de advertencia; la síntesis propone como mínimo dos ideas nuevas o descarta con argumentos alguna de las existentes; todas las citas verificadas (título, autores, año, identificador).

**Aprendizaje para Miguel.** Leer en profundidad, él mismo, cinco papers ancla: REGCOIL, FOCUS, Wechsung et al. 2022, Kappel–Landreman, y el tutorial de Amos. Para el resto bastan las fichas.

**Riesgos y mitigación.** Citas inventadas o mal atribuidas → toda referencia se comprueba contra arXiv/DOI antes de entrar en `references.bib`; las de este roadmap también están pendientes de esa verificación. Dispersión → timebox de 6 sesiones y foco en ideas accionables.

**Modelos.** **Fable** para la lectura cruzada y la síntesis (es donde más rinde); subagentes Sonnet/Haiku para localizar PDFs y extraer metadatos; el MCP de literatura científica para búsquedas en arXiv.

---

### Fase 4 — Métricas diferenciables en JAX (`coilgen.physics`)

**Objetivo.** Implementación propia, rápida, vectorizable y diferenciable de las 4 métricas del benchmark, con paridad numérica demostrada frente a las métricas DESC de upstream.

**Por qué importa.** Es la pieza de la que dependen las apuestas A2, A4, A5, A7 y A8, y el componente más reutilizable y enseñable del portfolio.

**Decisión previa (primera sesión de la fase).** DESC ya es JAX y diferenciable. Antes de reimplementar, evaluar si `metrics_utils_v2` de upstream admite `jit` + `vmap` sobre lotes con coste aceptable. Se reimplementa solo lo necesario; el criterio es throughput, memoria y exportabilidad, no purismo.

**Tareas**

- [ ] `curves.py`: evaluación de `FourierXYZCoil` (posición, tangente, segunda derivada) y curvatura; expansión de las bobinas únicas al toro completo por simetría de periodo y de stellarator, con los signos correctos de corriente.
- [ ] `surface.py`: puntos, normales y elementos de área de la frontera a partir de los coeficientes R/Z; radio menor `a` con la misma definición que upstream.
- [ ] `biot_savart.py`: campo de filamentos por cuadratura; versión por bloques para acotar memoria; opción float32/float64.
- [ ] `metrics.py`: d_CC (entre todas las bobinas del toro completo), d_CP, a·κ_max, ε_max y ε_mean, en versión **exacta** (para evaluar) y **suave** (soft-min/soft-max con temperatura, para entrenar).
- [ ] `score.py`: violaciones, s_i, S y strict en JAX, idénticos a `benchmark/scoring.py`.
- [ ] **Tests de paridad** contra upstream en ≥ 200 instancias estratificadas (todas las N_FP y N_C): error relativo por métrica.
- [ ] Estudio de convergencia en resolución: puntos de cuadratura por bobina y malla de la frontera vs. error, e igualación de las mallas que usa el evaluador.
- [ ] Tests de gradiente (diferencias finitas vs. autodiff) y de invariancias (reparametrización, permutación, escala de corriente).
- [ ] Casos analíticos: espira circular, dos espiras coaxiales, toro con bobinas toroidales ideales.
- [ ] Benchmark de rendimiento: instancias/segundo en CPU local, GPU de Colab y GPUs del cluster; memoria por tamaño de lote.
- [ ] Validación de extremo a extremo: recalcular el score de los baselines de la Fase 2 con `coilgen.physics` y compararlo con el de DESC, instancia a instancia.

**Entregables.** `src/coilgen/physics/{curves,surface,symmetry,biot_savart,metrics,score}.py`, `tests/physics/`, `notebooks/07_metric_parity.ipynb`, `notebooks/08_resolution_convergence.ipynb`, `docs/reports/04_differentiable_metrics.md`, `docs/notes/derivations.md` (derivaciones de Miguel).

**Definition of Done.**
- Error relativo mediano ≤ 1e−5 y máximo ≤ 1e−4 en las 3 métricas geométricas; ≤ 1e−3 máximo en ε_max, con el objetivo de bajar a 1e−4 (la tolerancia strict del benchmark es 1e−3, así que el desajuste restante se convierte en margen de entrenamiento explícito).
- Acuerdo en la decisión strict/no-strict ≥ 99 % de las instancias respecto al evaluador DESC.
- Throughput ≥ 100× el de la evaluación DESC por instancia en GPU **(objetivo a confirmar con la medida de la Fase 2)**.
- Gradientes validados; cobertura de tests del módulo ≥ 90 %.

**Aprendizaje para Miguel.** Ley de Biot–Savart para filamentos y su cuadratura (convergencia espectral de la regla trapezoidal en curvas periódicas); geometría diferencial de curvas (curvatura, torsión) y superficies (normal, primera forma fundamental); por qué B·n = 0 caracteriza una superficie de flujo y qué mide el flujo cuadrático; aritmética de punto flotante y cancelación. Lecturas: Zhu et al. 2018 (FOCUS), documentación de `desc.coils` y `desc.objectives`, sección de métricas del paper (Apéndice E). **Miguel deriva a mano** la expresión del campo y de la curvatura y escribe la primera versión de `curves.py`; Claude revisa.

**Riesgos y mitigación.**
- *ε_max no alcanza paridad* (malla, interpolación, definición del máximo) → leer el código de upstream línea a línea; si no se resuelve, usar las funciones de DESC para evaluar y las propias solo para entrenar, con margen.
- *float32 insuficiente* (ε ~ 1e−3 de |B| implica cancelaciones) → float64 en evaluación; medir si float32 basta para gradientes de entrenamiento.
- *Memoria de Biot–Savart* (bobinas × puntos de cuadratura × puntos de superficie × lote) → evaluar solo medio periodo de superficie, bloques con `lax.map`, lotes pequeños con acumulación de gradiente.
- *Mín/máx no suaves* → versiones suaves con temperatura en curriculum; verificar siempre con las exactas.

**Modelos.** Opus para la implementación y los tests; **Fable** para depurar discrepancias numéricas sutiles frente a DESC y para revisar las derivaciones; Sonnet para notebooks de paridad e informe.

---

### Fase 5 — Track A v1: entrenamiento physics-informed en un solo forward pass

**Objetivo.** Superar con claridad a todos los baselines publicados con una red de un solo paso, sin física en el grafo de inferencia.

**Por qué importa.** Es el resultado central del proyecto y el que responde a la pregunta del paper tal como está formulada.

**Tareas**

- [ ] Experimento mínimo: fine-tune del baseline reproducido con `L = Σ max(0, v_i + m_i)/α_i` (margen `m_i`), y medir ΔS por estrato. Es la prueba de la apuesta A2.
- [ ] Estudio de pérdidas: solo supervisada, solo física, mixta con annealing (supervisada → física), con y sin margen, suave vs. exacta. Términos de regularización (longitud, suavidad espectral).
- [ ] Canonicalización de targets y/o pérdida geométrica invariante a reparametrización (según la decisión de la Fase 1).
- [ ] Hindsight relabeling: añadir coilsets REGCOIL y DESC filtrados con `r = g(c)`; relabeling online de las salidas del propio modelo.
- [ ] Aumento de requisitos: para una frontera, entrenar con requisitos más laxos que los de una referencia factible (siguen siendo factibles).
- [ ] Arquitectura: mejoras incrementales sobre ResMLP/PerCoil (capacidad, normalización, condicionamiento de requisitos tipo FiLM, embeddings de N_FP/N_C, parametrización de corrientes relativas).
- [ ] Parametrización de salida con sesgo físico (A6): residuo sobre plantilla; evaluar el error de representación antes de entrenar.
- [ ] Estabilidad: clipping, ponderación de las 4 violaciones, detección de colapsos (bobinas que se cruzan o se alejan al infinito).
- [ ] Evaluación: `dev-300` con `coilgen.physics` en cada experimento; `val-full` con DESC solo para candidatos; 3 seeds para los finalistas.
- [ ] Envío al leaderboard del mejor modelo Track A (fixed y variable), con model card.

**Entregables.** `src/coilgen/training/{losses,loop,curriculum}.py`, `src/coilgen/models/`, `configs/track_a/*.yaml`, `notebooks/09_track_a_analysis.ipynb`, `docs/reports/05_track_a_v1.md`, repos HF `coilgen-a-*`.

**Definition of Done.** En `val-full` y en el leaderboard, fixed-shape: S̄ y strict por encima de los mejores baselines de Proxima y del baseline de círculos, con intervalos de confianza que no se solapan (3 seeds). Tabla de ablations completa. Informe con análisis de fallos por estrato.

**Aprendizaje para Miguel.** Physics-informed learning y aprendizaje auto-supervisado por objetivo; optimización con restricciones (penalización, Lagrangiano aumentado — el mismo método que usa DESC en el pipeline del dataset); optimización amortizada (Amos); buenas prácticas de experimentación (una variable por experimento, ablations, seeds). Miguel diseña y ejecuta él mismo al menos una ablation completa.

**Riesgos y mitigación.**
- *La pérdida física converge a mínimos malos o a soluciones degeneradas* → warm start supervisado, annealing, regularización de longitud, margen progresivo.
- *Coste por paso alto* → perfilar primero; resolución reducida en entrenamiento con verificación a resolución completa solo si hace falta; estimar GPU-horas antes de cada barrido.
- *Sobreajuste al leaderboard* → decisiones solo con el split de validación propio; envíos contados y registrados.
- *Mejora en loose pero no en tight* → reportarlo tal cual; es la motivación de la Fase 6.

**Modelos.** Opus para pérdidas, bucle de entrenamiento y análisis de experimentos; **Fable** para una revisión crítica del diseño antes del barrido grande y para interpretar resultados inesperados; Sonnet para configs, scripts de lanzamiento e informe; Haiku para resumir logs.

---

### Fase 6 — Track A v2: modelos avanzados y multimodalidad

**Objetivo.** Atacar los estratos medium y tight con modelos que representen la multimodalidad real y aprovechen la estructura del problema, manteniendo la etiqueta de Track A.

**Por qué importa.** Es donde el proyecto puede aportar algo original; también es la fase con más riesgo de dispersión, por eso empieza con un experimento de oráculo.

**Tareas**

- [ ] **Oráculo best-of-K** (puerta de entrada): con K candidatos baratos (miembros del ensemble sin promediar, perturbaciones, seeds), ¿cuánto sube el score eligiendo el mejor con la física? Si la ganancia es pequeña, la multimodalidad no es el cuello de botella y se prioriza la Fase 7.
- [ ] Multi-hipótesis: K cabezas con pérdida winner-takes-all y **selector aprendido** (predice el score de cada hipótesis); todo dentro del grafo, sin física.
- [ ] Generativo condicional: flow matching sobre coeficientes canónicos, condicionado a (frontera, requisitos, N_FP, N_C); guiado por la pérdida física durante el entrenamiento; destilación a 1–4 pasos para la inferencia.
- [ ] Arquitectura estructurada: tokens por bobina y por modo de frontera, atención cruzada frontera–bobinas, pesos compartidos entre bobinas; equivariancia/invariancia a las simetrías identificadas.
- [ ] Destilación del Track B (si la Fase 7 va por delante): usar coilsets refinados como nuevos targets de alta calidad para el modelo puro.
- [ ] Generalización: resultados por N_FP y N_C en variable-shape; combinaciones poco pobladas.
- [ ] Envío del mejor Track A v2.

**Entregables.** `src/coilgen/models/{multihead,flow,transformer}.py`, `configs/track_a_v2/*.yaml`, `notebooks/10_best_of_k_oracle.ipynb`, `docs/reports/06_advanced_models.md`.

**Definition of Done.** Mejora estadísticamente significativa sobre Track A v1 en S̄ o en strict de los estratos medium/tight; o bien un informe negativo bien argumentado (oráculo + ablations) que justifique cerrar la línea. Timebox estricto de 6 semanas.

**Aprendizaje para Miguel.** Difusión y flow matching (Ho et al. 2020; Lipman et al. 2023), guidance, destilación; atención y redes sobre conjuntos; equivariancia. Recurso práctico: implementar un flow matching 2D de juguete antes de tocar el problema real.

**Riesgos y mitigación.** Complejidad sin retorno → oráculo primero, timebox, una línea cada vez. Exportación a ONNX de modelos iterativos → desenrollar los pasos en el grafo y probarlo pronto. Sobreajuste de modelos grandes con ~150k muestras → el tamaño del modelo se decide por validación, no por hardware (el cluster permite barrer capacidad); regularización, aumento por física (A4) y early stopping.

**Modelos.** **Fable** para el diseño del método y la revisión crítica de resultados; Opus para la implementación; Sonnet para el resto.

---

### Fase 7 — Track B: retrieval y refinamiento híbrido

**Objetivo.** Un sistema aprendido + refinamiento competitivo con la mejor entrada híbrida, y la curva completa calidad vs. cómputo de inferencia.

**Por qué importa.** Es el uso realista (el paper lo propone como warm start) y, hoy, donde están los scores altos. Se etiqueta siempre como híbrido.

**Tareas**

- [ ] Revisar la respuesta de los organizadores sobre reglas y ajustar el alcance (envío vs. estudio offline).
- [ ] Refinamiento offline en JAX desde la salida del Track A: Gauss–Newton / Levenberg–Marquardt sobre los residuos del score. Curva S̄ y strict vs. nº de pasos, por estrato y por punto de partida (modelo, vecino más próximo, REGCOIL, círculos).
- [ ] Comparar con el estudio de warm-start del paper usando DESC (`--desc-maxiters`) en un subconjunto del estrato tight.
- [ ] Retrieval: índice de coilsets de train (fp16), métrica de similitud aprendida o diseñada; ¿ayuda como inicio frente al modelo?
- [ ] **Refinamiento unrolled aprendido:** K pasos fijos con amortiguamiento y precondicionador aprendidos, entrenado extremo a extremo; objetivo: misma calidad con muchos menos pasos que un LM genérico.
- [ ] Implementación en ONNX: Biot–Savart, métricas y jacobianos con `onnxscript`/`Loop`; test de equivalencia con la versión JAX; tamaño ≤ 50 MiB; tiempo por instancia en una CPU.
- [ ] Envío Track B con model card que detalle el coste de inferencia.

**Entregables.** `src/coilgen/refine/{lm,unrolled,retrieval}.py`, `src/coilgen/export/onnx_physics.py`, `configs/track_b/*.yaml`, `notebooks/11_refinement_curves.ipynb`, `docs/reports/07_hybrid.md`, repos HF `coilgen-b-*`.

**Definition of Done.** Curva calidad–cómputo completa en `val-full`; un punto con S̄ ≥ 0.9 y strict ≥ 70 % en fixed-shape; equivalencia ONNX–JAX verificada; si hay envío, score oficial coherente con el local. Si las reglas lo prohíben: el estudio offline completo sustituye al envío.

**Aprendizaje para Miguel.** Mínimos cuadrados no lineales (Gauss–Newton, LM, regiones de confianza); diferenciación implícita y unrolling; cómo optimiza DESC/SIMSOPT bobinas en la práctica; operadores de ONNX. Lectura: Nocedal & Wright, capítulo de mínimos cuadrados no lineales; Monga et al. 2021.

**Riesgos y mitigación.** Reglas cambian a mitad de fase → el valor está en el estudio, no en el envío. ONNX con bucles y jacobianos es laborioso → empezar por el caso sin `Loop` (K pasos desenrollados) y medir. Límite de tiempo del evaluador desconocido → preguntar y medir con envíos pequeños.

**Modelos.** Opus para LM, unrolling y ONNX; **Fable** para el diseño del refinamiento aprendido y para depurar divergencias numéricas ONNX/JAX; Sonnet para informes.

---

### Fase 8 — Evaluación final, ablations y envíos definitivos

**Objetivo.** Congelar resultados con el máximo rigor estadístico y dejar los envíos definitivos.

**Tareas**

- [ ] Congelar código (`v1.0.0-rc`) y configs de los modelos finales de ambos tracks.
- [ ] Reentrenar finalistas con 3–5 seeds; evaluación con DESC en `val-full` y en test.
- [ ] Tabla maestra: S̄, mediana, p10, soft, strict, por track, estrato, N_FP y N_C, con IC por bootstrap y tests pareados.
- [ ] Ablations finales: cada componente retirado de uno en uno.
- [ ] Análisis de fallos: qué instancias siguen sin resolverse y por qué (longitud de escala del gradiente, N_FP alto, requisitos extremos).
- [ ] Verificación independiente: clonar el repo en una máquina limpia y regenerar una fila de la tabla maestra.
- [ ] Envíos definitivos y snapshot del leaderboard.
- [ ] Revisión crítica externa simulada (Fable como "revisor 2") de todas las afirmaciones.

**Entregables.** `docs/reports/08_final_evaluation.md`, `outputs/final/` (no versionado; tablas exportadas a `docs/reports/assets/`), tag `v1.0.0`.

**Definition of Done.** Cada número del informe final es regenerable con un comando documentado; ninguna afirmación sin IC; revisión crítica resuelta punto por punto en BITACORA.

**Modelos.** Opus para análisis; Fable para la revisión crítica; Sonnet para tablas e informe; Haiku para agregación de logs.

**Esfuerzo.** 2 semanas. **Riesgo:** descubrir tarde una fuga de datos o un error del harness → la verificación independiente se hace antes de escribir el informe.

---

### Fase 9 — Write-up y difusión

**Objetivo.** Convertir el trabajo en piezas públicas que demuestren capacidad técnica y criterio.

**Tareas**

- [ ] Informe técnico estilo arXiv (inglés, 8–12 páginas): problema, análisis del benchmark, métodos, resultados, frontera calidad–cómputo, limitaciones, trabajo futuro.
- [ ] README final con figuras, tabla de resultados y reproducción en 3 comandos.
- [ ] Post técnico largo (blog o GitHub Pages) y versión corta para LinkedIn.
- [ ] Model cards completas en HF; opcionalmente una demo (Space) que visualice bobinas generadas.
- [ ] Extraer `coilgen.physics` como componente documentado y reutilizable.
- [ ] Mensaje a los autores con resultados y feedback sobre el benchmark (§8.3).
- [ ] Retrospectiva del proyecto en BITACORA: qué funcionó, qué no, qué haría distinto.

**Entregables.** `docs/reports/final/coilgen_report.{tex,pdf}`, `README.md`, post(s), model cards.

**Definition of Done.** Informe revisado por Miguel línea a línea; todas las figuras generadas por scripts versionados; repo público con CI verde y licencia; mensaje enviado.

**Aprendizaje para Miguel.** Escritura científica; presentar resultados con honestidad; preparar una charla de 10 minutos sobre el proyecto.

**Modelos.** Sonnet para borradores y figuras; Fable para la revisión de argumentación y de afirmaciones; Miguel escribe él mismo la introducción y las conclusiones.

**Esfuerzo.** 3 semanas.

---

## 5. Asignación de modelos Claude

Precios por MTok (entrada/salida): Fable $8/$40 · Opus $4/$20 · Sonnet $2/$10 · Haiku $0.10/$0.50. Fable cuesta 2× Opus, 4× Sonnet y 80× Haiku: se reserva para donde la profundidad cambia el resultado.

### 5.1 Por tipo de tarea

| Tipo de tarea | Modelo | Justificación |
|---|---|---|
| Revisión de literatura y síntesis entre muchos papers | **Fable** | Requiere mantener muchas fuentes y extraer ideas accionables |
| Diseño de métodos nuevos (pérdidas, refinamiento aprendido, generativos) | **Fable** | Decisiones de alto apalancamiento y difícil reversión |
| Derivaciones matemáticas/físicas difíciles; depuración numérica sutil (paridad con DESC, ONNX vs. JAX) | **Fable** | Errores caros de detectar, razonamiento largo |
| Revisión crítica de resultados y de afirmaciones antes de publicar | **Fable** | "Revisor 2" exigente |
| Implementación de algoritmos núcleo (`physics`, pérdidas, LM, export ONNX) | **Opus** | Código difícil con buena relación calidad/coste |
| Diseño de arquitectura de software; code review de PRs importantes | **Opus** | |
| Análisis de experimentos y decisiones de siguiente paso | **Opus** | Escalar a Fable si el resultado es inesperado |
| Código rutinario, scripts, configs, loaders, tests unitarios | **Sonnet** | |
| Notebooks de EDA, figuras, informes de fase, entradas de BITACORA, README | **Sonnet** | |
| Búsqueda en ficheros, formateo, renombrados, resumen de logs, extracción de metadatos | **Haiku** | Volumen alto, bajo riesgo |
| Mensajes de commit, changelog | Haiku / Sonnet | |

### 5.2 Por fase

| Fase | Sesión principal | Subagentes típicos | Uso de Fable |
|---|---|---|---|
| 0 Setup | Sonnet | Haiku (explorar upstream) | No |
| 1 EDA | Sonnet | Haiku (esquemas), Opus (interpretación) | No, salvo hallazgo extraño |
| 2 Baselines | Opus | Sonnet (scripts), Haiku (logs) | No |
| 3 Literatura | Fable | Sonnet/Haiku (fichas, metadatos) | **Sí, principal** |
| 4 Métricas JAX | Opus | Sonnet (tests, notebooks) | Depuración de paridad y revisión de derivaciones |
| 5 Track A v1 | Opus | Sonnet (configs), Haiku (logs) | Revisión de diseño antes del barrido; resultados anómalos |
| 6 Track A v2 | Opus | Sonnet | **Diseño del método** y revisión crítica |
| 7 Track B | Opus | Sonnet | Diseño del refinamiento aprendido; depuración ONNX |
| 8 Evaluación final | Opus | Sonnet, Haiku | Revisión crítica final |
| 9 Write-up | Sonnet | Haiku (formato) | Revisión de argumentación |

### 5.3 Sesión principal vs. subagentes

- **Sesión principal:** trabajo interactivo donde Miguel necesita seguir el razonamiento (diseño, depuración, análisis), y todo lo que modifica código núcleo.
- **Subagentes:** tareas autocontenidas y paralelizables cuyo detalle intermedio no hace falta en el contexto principal — leer y fichar un paper, explorar el repo upstream, resumir logs, generar tests para un módulo ya especificado. Cada subagente arranca en frío: darle siempre objetivo, ficheros relevantes, criterio de terminado y formato de respuesta; `CLAUDE.md` y este roadmap son su briefing.
- **No usar subagentes** para tareas pequeñas (el arranque cuesta más que la tarea) ni para decisiones de diseño (se pierde el contexto compartido).
- **Regla de escalado:** empezar con el modelo más barato razonable; subir un nivel si falla dos veces o si el resultado afecta a una decisión go/no-go.

### 5.4 Lo que Miguel hace él mismo

| Actividad | Por qué no delegarla |
|---|---|
| Leer el paper y los cinco papers ancla | Es el conocimiento que se examina en una entrevista |
| Derivar a mano Biot–Savart, curvatura y el score | Entender la física que se optimiza |
| Escribir la primera versión de `curves.py` y de al menos dos notebooks de EDA | Fluidez con JAX y con los datos |
| Lanzar entrenamientos y envíos | Dominar el ciclo experimental completo |
| Diseñar y ejecutar una ablation completa | Criterio experimental |
| Revisar cada PR antes de fusionar | Responsabilidad sobre el código |
| "Test de Feynman" al cierre de cada fase (explicación propia en BITACORA) | Detectar lagunas |
| Publicar en HF/LinkedIn y escribir a los autores | Voz propia |

---

## 6. Prácticas transversales

### 6.1 Estrategia de cómputo

El hardware local **no** es el límite del proyecto. Hay tres niveles, cada uno con su papel:

| Nivel | Recurso | Papel | No usar para |
|---|---|---|---|
| 1 | **Máquina local** (2 CPU sin AVX, 3 GB RAM, sin GPU) | Desarrollo, tests unitarios sin JAX, EDA ligero sobre `results` con pandas/pyarrow, sesiones con Claude Code | Cualquier cosa con JAX/DESC (jaxlib exige AVX), entrenamiento, evaluación |
| 2 | **Google Colab** (GPU) | Experimentos interactivos y prototipos: probar una pérdida, depurar en GPU, notebooks de paridad, primeras curvas | Barridos, trabajos largos o que deban sobrevivir a una desconexión |
| 3 | **GPUs del grupo de investigación de la universidad** | **Recurso principal:** entrenamientos serios, barridos de hiperparámetros, multi-seed, evaluaciones DESC de splits completos, refinamiento a gran escala | — |
| Opcional | Cloud de pago (vast.ai, RunPod, Lambda, Modal) | Solo como respaldo si el cluster no está disponible o hay un pico puntual | Uso por defecto |

**Flujo de trabajo en el cluster** (los detalles concretos se fijan en `docs/COMPUTE.md` en la Fase 0):

- **Acceso y planificación:** SLURM (`sbatch`, job arrays) si existe; si es SSH directo, `tmux` + un script lanzador que registre qué GPU ocupa cada run. En ambos casos, los mismos puntos de entrada (`scripts/train.py`, `scripts/evaluate.py`) que en local.
- **Entorno:** `uv sync --frozen` desde `uv.lock` como primera opción; contenedor (Apptainer a partir del Dockerfile de upstream) si faltan dependencias de sistema (NetCDF, BLAS/LAPACK, gfortran) o `uv` no está permitido. La versión de JAX con CUDA se fija por extra/grupo de dependencias para no romper el entorno local de CPU.
- **Código:** se sincroniza **solo por git**. Cada job registra el hash de commit y falla si el árbol de trabajo está sucio (salvo flag explícito de depuración).
- **Datos:** vía caché de Hugging Face (`HF_HOME` en scratch), con la revisión fijada en `configs/data.yaml`. Si los nodos de cómputo no tienen internet, se descarga en el nodo de login y se usa `HF_HUB_OFFLINE=1`. El subset `metrics` (213 GB) solo se plantea, y parcialmente, en el cluster.
- **Resultados:** W&B (modo offline + `wandb sync` si no hay red en los nodos); checkpoints y cachés en scratch, con copia de los finalistas a HF/W&B Artifacts. Nunca depender de scratch como almacenamiento único.
- **Evaluación DESC masiva:** job array que trocea el split en bloques, escribe resultados por bloque en la caché y es reanudable. Es lo que permite evaluar `val-full` y test con DESC de forma rutinaria.
- **Buen uso de un recurso compartido:** conocer y respetar las normas del grupo, estimar GPU-horas antes de cada barrido, no acaparar nodos, y agradecer el recurso en el informe final (confirmar con el responsable del grupo cómo citarlo).

Pautas generales:

- **JAX en GPU** para entrenamiento y para `coilgen.physics`. El coste dominante será previsiblemente Biot–Savart en la pérdida física, no la red **(a verificar perfilando)**.
- **Alcance sin restricciones artificiales:** tamaño de modelo, resolución de la pérdida física, número de seeds y amplitud de los barridos se deciden por criterio científico y por la cuota real del cluster.
- **Evaluación en dos velocidades:** `coilgen.physics` para iterar (`dev-300` en cada experimento, segundos); DESC para candidatos (`val-full`) y finalistas (test). El coste por instancia se mide en la Fase 2 y cada evaluación completa se anota en BITACORA.
- **Caché:** métricas por hash de (coilset, instancia, versión del evaluador); datasets procesados como artefactos reutilizables entre local, Colab y cluster.
- **Portabilidad:** todo experimento debe poder ejecutarse en Colab a escala reducida (`--debug`/subconjunto) y en el cluster a escala completa, con el mismo config.
- **Presupuesto económico:** el plan completo es viable sin gasto en cómputo; el cloud de pago queda como contingencia a aprobar por Miguel caso por caso.

### 6.2 Seguimiento de experimentos

**Elección: Weights & Biases.** Motivos: es un servicio alojado (no hay que mantener un servidor, lo cual importa cuando el cómputo se reparte entre local, Colab y el cluster de la universidad; admite modo offline con sincronización posterior si los nodos no tienen red), gratuito para uso personal, con artefactos, barridos e informes públicos que sirven directamente para el portfolio. MLflow sería preferible si se necesitara auto-alojamiento; no es el caso. Para no depender del servicio, cada run guarda además en `outputs/<run_id>/` su config resuelto, métricas en JSON y el hash de commit.

Cada run registra: commit, config completo, revisión del dataset, seed, hardware, duración, métricas por estrato y enlace al checkpoint.

### 6.3 Configuración y reproducibilidad

- **Configs** en YAML bajo `configs/`, validados con modelos tipados (pydantic/dataclasses); un config = un experimento; overrides por CLI; el config resuelto se guarda con el run.
- **Seeds** explícitas (PRNG de JAX) para datos, inicialización y entrenamiento; resultados finales con ≥ 3 seeds.
- **Entorno** bloqueado con `uv.lock`; submódulo upstream fijado por commit; versiones de JAX/DESC anotadas en cada run.
- **Datos** versionados por revisión de HF (commit hash) en `configs/data.yaml`; nunca se commitean (`data/`, `outputs/` y la caché de HF están en `.gitignore`).
- **Splits** deterministas por hash de frontera; el fichero de índices del split de validación propio se versiona.
- **Estadística:** IC por bootstrap sobre instancias, comparación pareada entre modelos, media ± σ entre seeds.

### 6.4 Calidad de código

- `ruff` (lint + formato) y `pre-commit`; type hints en todo `src/coilgen/`; docstrings con unidades y shapes.
- `pytest` con marcadores: rápidos (CI en cada push), `slow` y `desc` (manuales o programados, requieren DESC completo).
- Tests de física obligatorios: casos analíticos, invariancias, gradientes, paridad con upstream.
- **CI (GitHub Actions):** lint + tests rápidos con caché de `uv`. Los tests pesados no bloquean el merge pero se ejecutan antes de cada tag.
- Notebooks: salidas limpiadas antes de commitear (o emparejados con `jupytext`); la lógica reutilizable vive en `src/`, no en notebooks.

### 6.5 Flujo de git

- `main` siempre verde; ramas `feat/…`, `fix/…`, `exp/…`, `docs/…`; merge por PR (aunque sea de uno mismo) con descripción y checklist.
- **Conventional Commits** en inglés (`feat(physics): add Biot-Savart quadrature`).
- **Tags** en hitos: `v0.1.0` (M1) … `v1.0.0` (M8).
- **Cada commit se registra en `docs/BITACORA.md`** (fecha, hash, qué y por qué).
- Nada de datos, checkpoints, PDFs de papers ni secretos en el repo.

### 6.6 Documentación

- `docs/BITACORA.md`: diario cronológico — commits, decisiones (formato ADR breve: contexto, opciones, decisión, consecuencias), resultados de experimentos clave, tests de Feynman.
- `docs/reports/NN_<fase>.md`: un informe por fase con objetivo, método, resultados, limitaciones y siguientes pasos.
- `docs/notes/`: apuntes de aprendizaje y derivaciones de Miguel.
- `docs/literature/`: fichas de papers y `references.bib`.
- Este roadmap: plan y estado.

### 6.7 Protocolo de envíos al leaderboard

- Solo se envía lo que ya ha sido evaluado localmente con DESC en `val-full`.
- Nombre del repo y model card indican track (A/B), commit, config, datos de entrenamiento y coste de inferencia.
- Cada envío se anota en `docs/reports/submissions.md` con score local vs. oficial.
- No se ajustan hiperparámetros contra el score oficial.

---

## 7. Hitos, calendario y decisiones go/no-go

Calendario tentativo a tiempo parcial (~8–10 h/semana) desde el 2026-11-12. Se revisa en cada hito.

| Hito | Fecha tentativa | Contenido | Tag |
|---|---|---|---|
| M0 | 2026-11-23 | Entorno, repo, CI, paper leído, pregunta de reglas publicada | — |
| M1 | 2026-12-07 | EDA completo, escalera trivial, split de validación | `v0.1.0` |
| M2 | 2026-12-28 | Baselines reproducidos, harness, primer envío | `v0.2.0` |
| M3 | 2027-01-10 | Revisión de literatura y síntesis | — |
| M4 | 2027-01-31 | `coilgen.physics` validado contra DESC | `v0.3.0` |
| M5 | 2027-03-07 | Track A v1 por encima de los baselines, enviado | `v0.4.0` |
| M6 | 2027-04-18 | Track A v2 (o informe negativo) | `v0.5.0` |
| M7 | 2027-05-16 | Track B: curva calidad–cómputo (y envío si procede) | `v0.6.0` |
| M8 | 2027-05-30 | Evaluación final congelada | `v1.0.0` |
| M9 | 2027-06-20 | Informe, posts y contacto con autores | `v1.0.1` |

### Decisiones go/no-go

| Punto | Cuándo | Criterio | Si no se cumple |
|---|---|---|---|
| G1 | Tras M2 | Baseline reproducido dentro de ±2σ; score local ≈ oficial | No avanzar: depurar harness o exportación |
| G2 | Tras M4 | Paridad de métricas en los umbrales de la Fase 4; acuerdo strict ≥ 99 % | Usar DESC para evaluar y las métricas propias solo como pérdida con margen; revaluar el alcance del Track B en ONNX |
| G3 | Mitad de la Fase 5 | El fine-tune con pérdida física mejora S̄ en `dev-300` de forma significativa | Sesión de diagnóstico con Fable; probar refinamiento offline para separar "pérdida" de "capacidad del modelo" |
| G4 | Inicio de la Fase 6 | El oráculo best-of-K muestra margen sustancial | Saltar a la Fase 7 y reducir la 6 a destilación |
| G5 | Inicio de la Fase 7 | Respuesta de los organizadores sobre reglas | Prohibido → estudio offline; permitido → envío etiquetado |
| G6 | M5 y M7 | Revisión de alcance y calendario frente al leaderboard actual | Recalibrar los objetivos de L3/L4 |

Punto de control permanente: en cada hito se captura el leaderboard y se releen las reglas del Space.

---

## 8. Portfolio y carrera

### 8.1 Activos públicos

| Activo | Cuándo | Contenido |
|---|---|---|
| Repo de GitHub | Desde M0 o M2 (decisión de Miguel) | Código limpio, CI, README con resultados, licencia compatible con la MIT de upstream |
| Modelos en HF | M2, M5, M6, M7 | Model cards con track, método, resultados y limitaciones |
| Informes W&B públicos | M5, M7 | Curvas y ablations navegables |
| Post 1 | M1–M2 | "Anatomía de un benchmark de bobinas de stellarator": EDA, escalera trivial, qué mide realmente el score |
| Post 2 | M4 | "Métricas de bobinas diferenciables en JAX validadas contra DESC" |
| Post 3 | M5–M7 | Resultados: entrenamiento contra la física y frontera calidad–cómputo |
| Informe técnico | M9 | Estilo arXiv; decidir si se sube a arXiv según la solidez de los resultados |

Cada hito va acompañado de una actualización breve en LinkedIn: qué se hizo, un resultado, una figura, enlace al repo. Tono técnico y sobrio.

### 8.2 Qué demuestra el proyecto a un empleador de fusión + IA

- Comprensión del problema físico (stage 2, restricciones de ingeniería) y no solo de ML.
- Ingeniería de investigación: reproducibilidad, tests de física, evaluación estadística.
- Dominio del stack del sector: JAX, DESC, ONNX, Hugging Face.
- Criterio: análisis honesto del benchmark, etiquetado claro de métodos, resultados negativos documentados.

### 8.3 Relación con Proxima y la comunidad

Principio: aportar antes de pedir. Interacciones públicas, técnicas, breves y respetuosas.

- **Pregunta sobre reglas** (Fase 0–1) en la discusión del Space. Borrador:

  > Hi, and thanks for releasing the benchmark. A clarification question on the "single forward pass" rule: the current contract only constrains the ONNX interface and size, and some entries embed iterative optimizers and retrieval databases inside the graph. Are in-graph iterative refinement, embedded training data, and physics evaluation at inference considered in scope? Is there an inference-time budget per instance? I plan to submit to both a strictly amortized setting and a clearly labelled hybrid one, and would like to label them correctly.

- **Feedback sobre el benchmark** cuando haya evidencia: p. ej., la sensibilidad del ranking strict a instancias triviales del estrato loose, o problemas reproducibles en el código (issues/PRs pequeños y bien documentados en el repo upstream).
- **Contacto con los autores** en M9 (o antes, si hay un resultado claro en el Track A): mensaje corto con enlace al informe, tres resultados y una pregunta genuina. Sin pedir trabajo en el primer mensaje.
- **Comunidad:** seguir los repos de DESC y SIMSOPT (issues y discusiones), seminarios abiertos de la comunidad de stellarators, y conferencias/workshops de ML para ciencias físicas como posibles destinos del informe.
- Vigilar las ofertas de Proxima y de otras empresas del sector, y alinear el informe con las competencias que piden.

---

## 9. Supuestos y preguntas abiertas

**Supuestos de este plan**

1. Dedicación a tiempo parcial de ~8–10 h/semana; el calendario escala proporcionalmente.
2. El clon local de upstream no estaba disponible al redactar el plan (se estaba convirtiendo en submódulo); los detalles de `metrics_utils_v2`, de la exportación ONNX y de las mallas de evaluación se han tomado del resumen de contexto y están por verificar en las fases 0–2.
3. Las referencias bibliográficas citadas están pendientes de verificación formal en la Fase 3.
4. Los objetivos numéricos de L3/L4 se basan en el leaderboard del 2026-11-11/12.
5. JAX + Flax NNX como framework (coherencia con upstream y DESC).
6. Miguel dispone de GPUs en Colab y en el cluster de su grupo de investigación; el modo de acceso al cluster (SLURM/SSH, contenedores, red) está por documentar en la Fase 0.

**Preguntas para Miguel**

- [ ] ¿Cuántas horas semanales reales y hay fecha límite externa (p. ej., candidaturas)?
- [ ] Cluster del grupo: ¿SLURM o SSH? ¿Qué GPUs y cuotas? ¿Internet en los nodos? ¿Contenedores permitidos? ¿Hay que pedir permiso o citar al grupo por un proyecto personal?
- [ ] ¿Presupuesto mensual para uso de Claude (y, como contingencia, para cloud de pago)?
- [ ] ¿Repo público desde el día 1 o a partir de M2?
- [ ] ¿Nombre de usuario de HF/GitHub para los repos de modelos?
- [ ] Nivel de partida en JAX, ML generativo y electromagnetismo (para ajustar las tareas de aprendizaje).
- [ ] ¿Prioridad si hay que elegir: Track A (más original) o Track B (scores más altos)? Propuesta por defecto: A primero.

**Preguntas para los organizadores**

- [ ] Alcance de "single forward pass": iteración en el grafo, datos embebidos, física en inferencia.
- [ ] Límite de tiempo y de memoria por instancia en el evaluador; opset de ONNX soportado.
- [ ] ¿Se permite usar las entradas del split de evaluación (sin etiquetas) durante el entrenamiento?
- [ ] ¿La escala global de corrientes se renormaliza en la evaluación?

---

## 10. Cómo usamos este documento

- **Es un documento vivo.** Es la fuente de verdad del plan; `BITACORA.md` es la fuente de verdad de lo ocurrido.
- **Al empezar una sesión:** Claude lee este roadmap y las últimas entradas de la BITACORA, y propone el objetivo de la sesión a partir de las tareas pendientes de la fase activa.
- **Al terminar una tarea:** se marca su casilla `[x]` en el mismo commit que la completa (o en el siguiente), añadiendo entre paréntesis el hash o el informe que la acredita.
- **Al cerrar una fase:** se comprueba la Definition of Done punto por punto, se escribe el informe de fase, Miguel hace su test de Feynman, y se revisan las fases siguientes a la luz de lo aprendido.
- **Cambios de plan:** cualquier cambio de alcance, orden u objetivos se anota en el registro de cambios de abajo con fecha y motivo, y la decisión razonada va a la BITACORA. No se borran tareas: se marcan como `~~descartada~~` con el motivo.
- **Hipótesis:** cuando un "(a verificar)" se confirma o se refuta, se actualiza el texto y se enlaza la evidencia.
- **Fase activa actual:** Fase 0.

---

## 11. Registro de cambios

| Fecha | Versión | Cambio | Autor |
|---|---|---|---|
| 2026-11-12 | 1.0 | Versión inicial del roadmap | Claude (Fable) + Miguel |
| 2026-11-12 | 1.1 | Estrategia de cómputo en tres niveles: local (desarrollo), Colab (prototipos), GPUs del grupo de la universidad (recurso principal); cloud de pago solo como respaldo. Tareas de cluster añadidas a las fases 0 y 2 | Claude (Fable) + Miguel |
| 2026-11-12 | 1.2 | Hallazgo del setup: la CPU local no tiene AVX y JAX/DESC no corren en local. Todo lo que use JAX pasa a Colab o al cluster (fase 0, riesgos, tabla de §6.1) | Claude (Opus) + Miguel |
