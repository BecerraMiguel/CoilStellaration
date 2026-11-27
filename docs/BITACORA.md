# BITÁCORA — Proyecto CoilStellaration (`coilgen`)

Diario de trabajo del proyecto. Registra **qué se hizo, qué se decidió (y por qué), qué se discutió,
qué ideas surgieron y qué queda abierto**, en orden cronológico y vinculado a commits.
Lo usamos Miguel y Claude en cada sesión. Complementa a `docs/ROADMAP.md`: el roadmap es el plan
(fuente de verdad de lo que queremos hacer); la bitácora es la historia (fuente de verdad de lo que
ocurrió y de las razones).

## Cómo usar este documento

- **Orden:** cronológico, **las entradas nuevas van al final** (se lee de arriba abajo como una historia).
  Para ver lo último, ir al final del archivo.
- **Una entrada por sesión de trabajo** o por grupo de commits con sentido propio. Las entradas sin
  commits (investigación, discusión) también valen.
- **Cada entrada lista los hashes de los commits que cubre** (7 caracteres). Frase tipo: "después del
  commit `abc1234` se hizo tal cosa".
- **Convención de auto-referencia:** el commit que añade o modifica una entrada de la bitácora no
  puede contener su propio hash. Por eso el hash de ese commit se cita en la **entrada siguiente**
  (campo "Commits", marcado como "(commit de la bitácora de la entrada N)").
- **Decisiones con ID** `D-NNN` (correlativo, nunca se reutiliza). ROADMAP y CLAUDE.md pueden
  referenciarlas por ID. Si una decisión se revoca, se añade una nueva decisión que la sustituye y se
  anota "sustituye a D-NNN"; no se edita la original.
- **Al empezar una sesión:** leer las últimas entradas y sus "Próximos pasos" y "Preguntas abiertas".
- **Al terminar una sesión (o tras un grupo de commits):** añadir la entrada, con hashes reales
  (`git log --oneline`).
- Idioma: español. Código, commits y README en inglés (ver D-001).

### Plantilla de entrada

```markdown
## Entrada N — AAAA-MM-DD — <título corto>

**Fecha:** AAAA-MM-DD
**Sesión / Fase:** <nº de sesión, fase del roadmap>
**Commits:** `hash` mensaje; `hash` mensaje (o "ninguno")

### Qué se hizo
- ...

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-NNN | ... | ... | ... |

### Discusiones y hallazgos
- ...

### Ideas (backlog)
- ...

### Preguntas abiertas
- [ ] ...

### Próximos pasos
- [ ] ...

### Modelo(s) de Claude usados
- ...
```

---

## Entrada 0 — 2026-11-11 — Descubrimiento

**Fecha:** 2026-11-11 (la sesión de investigación se prolonga hasta el 2026-11-12)
**Sesión / Fase:** Previa al proyecto (descubrimiento)
**Commits:** ninguno (el repositorio aún no existía)

### Qué se hizo
- Miguel encontró el post de LinkedIn sobre CoilStellaration (Proxima Fusion).
- Leímos la nota de prensa de Proxima, el paper completo (arXiv:2611.12689), la dataset card de
  Hugging Face y el repositorio de GitHub (`proximafusion/coilstellaration`).
- Se produjo una explicación detallada del problema, el dataset, el benchmark, el scoring y los
  baselines (resumida en `CLAUDE.md`).
- Se analizó el leaderboard en vivo (2026-11-11/12) para entender dónde está el margen real.

### Hallazgos clave del análisis del leaderboard
- **El baseline de círculos supera a los baselines de Proxima en factibilidad estricta.**
  `phanerozoic/coilstellaration-circle-baseline` (círculos planos de radio 0.3·R0, sin aprendizaje)
  logra 6.6 % de factibilidad estricta en `fixed_shape`, frente a ~5.1 % del mejor baseline de
  Proxima (media 0.117 vs ~0.177). Hay instancias triviales en el estrato `loose` y el ranking
  strict-first las premia.
- **Optimizador dentro del grafo + retrieval.** `phanerozoic/coilstellaration-coil-optimizer`:
  recuperación de coilsets de entrenamiento + semilla del ensemble de Proxima + Levenberg–Marquardt
  con operaciones `Loop` de ONNX (~150 pasos, jacobianos analíticos), ~2–3 s por instancia en una CPU.
  Resultado: media 0.961, factibilidad estricta 78.8 % (tight 65 %).
- **Ambigüedad de reglas:** "single forward pass" solo se hace cumplir mediante el contrato ONNX, así
  que hoy un optimizador en el grafo es válido. Probablemente se aclararán las reglas.
- **Criterio de ranking distinto:** el leaderboard ordena por factibilidad estricta y luego por media;
  el paper ordena por media.
- **Discrepancia HF vs paper:** el rango de d_CC/a es 0.1–0.7 en la dataset card y 0.3–0.7 en el paper.

### Motivación de Miguel
Entrar en la industria de fusión + IA, idealmente en Proxima Fusion. El proyecto debe ser a la vez
un trabajo técnico serio, un portfolio público y una vía de aprendizaje.

### Preguntas abiertas
- [ ] Ver las preguntas consolidadas en la Entrada 1.

### Próximos pasos
- Inicializar el repositorio y elaborar un plan (Entrada 1).

### Modelo(s) de Claude usados
- Opus 5.5 (sesión principal).

---

## Entrada 1 — 2026-11-12 — Fase 0: Inicialización del repositorio

**Fecha:** 2026-11-12
**Sesión / Fase:** Sesión 1, Fase 0 (Setup y fundamentos)
**Commits:**

| Hash | Mensaje |
|---|---|
| `dc0a2f4` | chore: initialize repository structure and tooling |
| `9cf503a` | chore: add upstream coilstellaration as submodule |
| `b4cbdf2` | docs: add CLAUDE.md with project and challenge context |
| `b12d6ed` | docs: describe compute resources (Colab, university GPUs) in CLAUDE.md |
| `b397ce7` | chore: lock environment |
| `9c34499` | test: skip upstream import when CPU lacks AVX; document local JAX blocker |
| `5bfa3f1` | docs: add project roadmap (v1.2) |
| (este) | docs: add project journal (bitácora) with entries 0–1 — su hash se cita en la Entrada 2 |
| (condicional) | docs: reference bitácora workflow in CLAUDE.md — su hash se cita en la Entrada 2 |

### Qué se hizo
- Estructura del repo y herramientas (`dc0a2f4`): `.gitignore`, pre-commit, `.python-version`,
  LICENSE (MIT), README, `pyproject.toml`, paquete `src/coilgen/`, test de humo y carpetas
  `configs/`, `notebooks/`, `scripts/`, `data/`, `docs/papers/`.
- Upstream añadido como submódulo en `external/coilstellaration`, fijado en `5b925cf` (`9cf503a`).
- `CLAUDE.md` con todo el contexto del proyecto y del reto (`b4cbdf2`), luego ampliado con la
  estrategia de cómputo (`b12d6ed`).
- Entorno bloqueado con `uv.lock` (`b397ce7`).
- Después de `b397ce7` se detectó que JAX no funciona en la CPU local (D-008) y en `9c34499` se
  adaptó el test de humo y se documentó el bloqueo.
- Descarga del subconjunto `results` del dataset (0.309 GB; train 253,250 / eval 13,299; 178
  columnas) a la caché de HF, fuera de git.
- Después de `9c34499` se redactó el roadmap (v1.2) y se commiteó como `5bfa3f1`.
- Esta bitácora.

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-001 | **Idiomas:** código, comentarios, commits, README y CLAUDE.md en inglés; ROADMAP y BITACORA en español. | Repo público, audiencia profesional (Proxima, comunidad); los documentos de trabajo en español facilitan el aprendizaje de Miguel. | Todo en inglés (peor para aprender y reflexionar); todo en español (menos alcance y profesionalidad en el código). |
| D-002 | **Upstream como submódulo git** en `external/coilstellaration`, fijado en `5b925cf` (upstream main, 2026-11-12), instalado como dependencia editable por ruta vía `[tool.uv.sources]`. | Reproducibilidad exacta, código legible e inspeccionable, bump del pin deliberado y registrado. | Dependencia git de pip (menos legible/inspeccionable); copia en el repo (carga de licencia y mantenimiento); fork (divergencia innecesaria). |
| D-003 | **Paquete propio `coilgen`** con layout `src/`; backend de build `hatchling`; versión fija 0.1.0. | Layout `src/` evita importar por accidente el árbol de trabajo; hatchling es simple y es el que usa upstream. | Layout plano; setuptools/poetry. |
| D-004 | **uv + Python 3.12**; `uv.lock` commiteado; dependencias de desarrollo en `[dependency-groups]`. | Upstream exige >=3.12 y fija `desc-opt==0.17.1`; uv da entornos reproducibles y rápidos. | pip/conda/poetry. |
| D-005 | **Los datos nunca se commitean.** Caché de HF; solo se descargó el subconjunto `results` (0.309 GB); `coilsets` (2.3 GB) se difiere; `metrics` (213 GB) jamás completo. El PDF del paper va en `docs/papers/` ignorado por git. | Tamaño, buenas prácticas y licencia arXiv (no redistribuir). | Git LFS / DVC (innecesario: el dataset es público en HF); descargar todo (inviable en local). |
| D-006 | **Tooling:** ruff (línea 100; reglas E,F,I,UP,B,PTH,NPY), configuración de pre-commit (aún no instalada como hook), pytest; licencia MIT; Conventional Commits; todo commit se registra aquí. | Calidad y trazabilidad desde el inicio, con coste mínimo. | black+flake8+isort por separado; commits libres. |
| D-007 | **Cómputo en tres niveles** (aporte de Miguel): local = desarrollo, tests y EDA ligero; Colab = prototipos; GPUs del grupo de investigación de la universidad = recurso principal; cloud de pago = respaldo. | La máquina local (2 núcleos, 3 GB, sin GPU) no es el límite del proyecto; hay acceso a GPUs. | Planificar todo para la máquina local; depender de cloud de pago. |
| D-008 | **HALLAZGO:** la CPU local (Intel Celeron N2840) no tiene AVX. `jaxlib 0.9.2` se niega a importarse (`RuntimeError`) y da `SIGILL` si se salta la comprobación; compilar jaxlib desde fuente es irreal con 3 GB de RAM. **Consecuencia:** todo lo que use JAX/DESC/`coilstellaration` corre en Colab o en el cluster; `tests/test_smoke.py` omite el import de upstream si no hay AVX. ROADMAP actualizado a v1.2. | Es una limitación de hardware, no un fallo de configuración. | Compilar jaxlib desde fuente; forzar el import; cambiar de framework (perdería coherencia con upstream y DESC). |
| D-009 | **Roadmap elaborado con Claude Fable** (tarea de planificación profunda). Estructura: 10 fases (0 setup y fundamentos; 1 EDA y anatomía del score; 2 reproducción de baselines + harness de evaluación + primer envío de plomería; 3 revisión de literatura en paralelo; 4 métricas diferenciables en JAX; 5 Track A v1 physics-informed de un solo pase; 6 Track A v2 avanzado/multimodal; 7 Track B retrieval + refinamiento; 8 evaluación final y ablations; 9 write-up y difusión). Apuestas A1–A8 (métricas JAX diferenciables; entrenamiento physics-informed contra el score; canonicalización; hindsight relabeling; modelos multimodales; parametrizaciones con sesgo físico; retrieval; refinamiento amortizado/unrolled). **Track A** = single pass puro (sin evaluación física ni datos de entrenamiento embebidos en el grafo de inferencia); **Track B** = híbrido, etiquetado con su coste de inferencia. Hitos M0 2026-11-23 → M9 2027-06-20, con puntos go/no-go G1–G6. Política de modelos: Fable para investigación profunda, diseño novedoso, numérica sutil y revisión crítica; Opus para algoritmos centrales, arquitectura, revisión de código y análisis; Sonnet para código rutinario, EDA, tests, informes y bitácora; Haiku para tareas masivas/mecánicas; escalar tras 2 fallos. | Un plan por fases con criterios de salida y puntos de decisión reduce el riesgo; separar A y B mantiene la honestidad del resultado ante la ambigüedad de reglas (hilo conductor: frontera de Pareto calidad vs. cómputo de inferencia). | Plan informal sin hitos; centrarse solo en retrieval + refinamiento (el enfoque que ya domina el leaderboard, menos original y dependiente de la aclaración de reglas). |
| D-010 | **Modelos por subagente en esta sesión:** setup → Sonnet; roadmap → Fable; bitácora → Sonnet; sesión principal Opus 5.5. | Ajustar coste y capacidad a la tarea (ver política en D-009 y ROADMAP §5). | Usar un único modelo para todo. |

### Discusiones y hallazgos
- Durante el setup se reescribió una vez el historial local de git para separar el commit del
  submódulo del commit de estructura (antes de cualquier otra cosa y sin remoto, así que no hay
  consecuencias). Desde ahora el historial no se reescribe.
- Las revisiones de los hooks de pre-commit no se verificaron contra las últimas versiones.
- La configuración de pytest de upstream no aplica desde la raíz de nuestro repo: hay que pasar
  rutas explícitas (`uv run pytest external/coilstellaration/src/.../scoring_test.py`).
- REGCOIL (binario Fortran) no está compilado: solo se necesita para generar datos.
- El roadmap anota que los detalles de `metrics_utils_v2`, la exportación ONNX y las mallas de
  evaluación se tomaron de resúmenes y están por verificar en las fases 0–2.

### Ideas (backlog)
- A1 `coilgen.physics`: métricas del benchmark en JAX, rápidas y diferenciables (habilita todo lo demás).
- A4 Hindsight relabeling: cualquier coilset con métricas `g(c)` es un par válido `(b, r=g(c)) → c`.
- A5 Oráculo best-of-K como test barato para decidir si merece la pena invertir en modelos multimodales.
- A8 Refinamiento amortizado/unrolled (K pasos de Gauss–Newton/LM con hiperparámetros aprendidos).
- Write-up: contribución como frontera de Pareto calidad vs. cómputo de inferencia, con el baseline
  de círculos siempre como referencia.
- Escalera trivial de baselines (círculos, vecino más próximo) antes de entrenar nada.

### Preguntas abiertas
Para Miguel:
- [ ] ¿Crear el repo en GitHub: público desde ya o a partir de M2? ¿Nombre de usuario de GitHub/HF?
- [ ] Detalles del cluster: ¿SLURM o SSH? GPUs y cuotas, internet en nodos, contenedores, permiso del grupo.
- [ ] Horas semanales reales y fechas límite externas.
- [ ] Presupuesto para Claude (y cloud de pago como contingencia).
- [ ] Nivel de partida en JAX, ML generativo y electromagnetismo.
- [ ] Prioridad: Track A (más original) o Track B (scores más altos); propuesta por defecto: A primero.

Para los organizadores:
- [ ] Alcance de "single forward pass" (iteración en el grafo, datos embebidos, física en inferencia).
- [ ] Límites de tiempo y memoria por instancia; opset de ONNX soportado.
- [ ] ¿Se pueden usar las entradas sin etiquetas del split de evaluación durante el entrenamiento?

### Próximos pasos (Fase 0)
- [ ] Notebook de Colab que verifique JAX/DESC/coilstellaration y calcule las métricas de una fila.
- [ ] Documento de acceso al cluster: `docs/COMPUTE.md`.
- [ ] Remoto en GitHub + CI.
- [ ] Lecturas de fundamentos (ver ROADMAP, Fase 0).
- [ ] Publicar la pregunta de reglas a los organizadores (M0, 2026-11-23).

### Modelo(s) de Claude usados
- Opus 5.5 (sesión principal y revisión).
- Sonnet 5.5 (subagente de setup del repo y de esta bitácora).
- Fable 5.2 (subagente del roadmap).

---

## Entrada 2 — 2026-11-26 — Repositorio publicado en GitHub

**Fecha:** 2026-11-26
**Sesión / Fase:** sesión 2, Fase 0
**Commits:** `af950ce` docs: add project journal (bitácora) with entries 0–1 (commit de la bitácora
de la Entrada 1); `b8591ea` docs: reference bitácora workflow in CLAUDE.md (citado en la Entrada 1
como "condicional"); más el commit que añade esta entrada (su hash se cita en la Entrada 3).

### Qué se hizo
- Miguel creó el repositorio público `BecerraMiguel/CoilStellaration` en GitHub.
- Después de `b8591ea` se añadió el remoto `origin` por SSH
  (`git@github.com:BecerraMiguel/CoilStellaration.git`, clave `~/.ssh/id_ed25519_github` ya
  configurada) y se hizo `git push -u origin main` con los 9 commits de la Entrada 1.

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-011 | **Repo público en GitHub desde el inicio**, remoto por SSH, rama única `main` por ahora. | El historial y la bitácora muestran el proceso, no solo el resultado (valor de portafolio); facilita CI y llevar el código al cluster/Colab con `git clone --recurse-submodules`. Datos, PDFs y checkpoints ya están en `.gitignore`. | Privado hasta M2 (se pierde la trazabilidad pública del proceso); HTTPS con token (SSH ya estaba configurado). |

### Discusiones y hallazgos
- El email de los commits (identidad global de git) queda visible en el historial público; Miguel
  lo aceptó al crear el repo. Si se quiere ocultar en el futuro: email `noreply` de GitHub para los
  commits nuevos (el historial ya publicado no se reescribe).
- El hito M0 (2026-11-23) ya pasó: la pregunta de reglas a los organizadores sigue pendiente.

### Próximos pasos (Fase 0)
- [ ] CI con GitHub Actions (ruff + pytest; en los runners de GitHub sí hay AVX, así que también
  corre el import de upstream).
- [ ] Notebook de Colab que verifique JAX/DESC/coilstellaration y calcule las métricas de una fila.
- [ ] `docs/COMPUTE.md` con el acceso al cluster.
- [ ] Publicar la pregunta de reglas a los organizadores.

### Modelo(s) de Claude usados
- Opus 5.5 (sesión principal; tarea corta, sin subagente).

---

## Entrada 3 — 2026-11-26 — Autoría de los commits

**Fecha:** 2026-11-26
**Sesión / Fase:** sesión 2, Fase 0
**Commits:** `4a151ec` docs(bitacora): add entry 2 (GitHub remote) (commit de la bitácora de la
Entrada 2); más el commit que añade esta entrada (su hash se cita en la Entrada 4).

### Qué se hizo
- Miguel vio a Claude como contribuidor en GitHub. En un primer momento se interpretó que no quería
  aparecer con Claude: se reescribió el historial para quitar el trailer `Co-Authored-By` y se hizo
  push forzado.
- Miguel aclaró que sí quiere a Claude como **coautor**, siempre que él sea el **autor**. Esa ya era
  la situación original (autor de todos los commits: Miguel Becerra; Claude solo en el trailer), así
  que se restauró el historial original desde el tag local `backup/pre-rewrite` y se volvió a hacer
  push forzado. Los hashes vuelven a ser los originales; las Entradas 1–2 siguen siendo válidas.
  La versión sin trailer queda solo como tag local `backup/no-trailer` (no publicado).

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-012 | **Autoría:** Miguel es siempre el autor de los commits; Claude figura solo como coautor mediante el trailer `Co-Authored-By: <modelo> <noreply@anthropic.com>`. Además, cada entrada de la bitácora registra qué modelo hizo qué. | Proyecto de portafolio de Miguel, con transparencia sobre el uso de IA. | Quitar el trailer (se probó y se revirtió: Miguel prefiere la coautoría visible). |

### Discusiones y hallazgos
- GitHub muestra a los coautores en la lista de contribuidores junto al autor; no se controla el
  orden. La autoría real de cada commit (campo *author*) es de Miguel en todos los casos.
- Desde ahora el historial publicado no se vuelve a reescribir.

### Modelo(s) de Claude usados
- Opus 5.5 (sesión principal; tarea corta, sin subagente).

---

## Entrada 4 — 2026-11-26 — Segundo PC de trabajo (Ryzen, con AVX)

**Fecha:** 2026-11-26
**Sesión / Fase:** sesión 3, Fase 0
**Commits:** `bc9b0dd` docs: clarify commit authorship convention; add journal entry 3 (commit de la
bitácora de la Entrada 3); más el commit que añade esta entrada y actualiza `CLAUDE.md` (su hash se
cita en la Entrada 5).

### Qué se hizo
- Se preparó un segundo PC de trabajo (AMD Ryzen 5 3500U, 4 núcleos / 8 hilos, 5.7 GB RAM, sin GPU,
  Ubuntu 26.04) clonando el repo desde GitHub e inicializando el submódulo (sigue en `5b925cf`).
- Instalación de uv (`~/.local/bin`) y `uv sync`. `booz-xform` (dependencia de `constellaration`)
  no tiene wheel y se compila desde fuente: hizo falta instalar por apt `build-essential cmake
  ninja-build libnetcdf-dev`, y limpiar su caché (`uv cache clean booz-xform`) porque un intento
  fallido dejó un `CMakeCache` apuntando a un `ninja` temporal inexistente.
- **Esta CPU sí tiene AVX/AVX2:** JAX 0.9.2 (CPU), DESC 0.17.1 e `import coilstellaration` funcionan.
  Pasan nuestros tests (2) y `scoring_test.py` de upstream (41).
- Hook de pre-commit instalado (todos los checks pasan) y kernel de Jupyter "Python (coilgen)".
- Git: identidad `Miguel Becerra <miguelfer.becerra@gmail.com>` a nivel de repo (no global), clave
  SSH propia de este PC registrada en GitHub, remoto `origin` cambiado de HTTPS a SSH,
  `pull.rebase=true`.
- Primer `git pull` desde este PC: trajo `bc9b0dd` (hecho en el PC Celeron) por fast-forward.
- `CLAUDE.md`, sección *Compute resources*: ahora describe ambos PCs, cómo distinguirlos y el flujo
  de sincronización.

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-013 | **Dos PCs locales sincronizados solo vía GitHub.** Cada sesión empieza con `git pull` y termina con commit + `git push`; el trabajo a medias se commitea (`wip:` o rama) antes de cambiar de PC. `pull.rebase=true`. Cada PC tiene su propia clave SSH. Lo no versionado (`.venv`, datos, caché de HF, outputs, memoria de Claude) es local de cada PC; el contexto compartido va en CLAUDE.md, ROADMAP y BITACORA. **Matiza D-008:** el bloqueo de JAX aplica solo al PC Celeron; el PC Ryzen sirve para desarrollo y pruebas pequeñas con JAX/DESC (el entrenamiento real sigue en Colab/cluster). | El PC Ryzen permite iterar en local con JAX/DESC sin depender de Colab; git ya es la fuente de verdad del proyecto. | Sincronizar carpetas (Dropbox/rsync: rompe `.git` y `.venv`); trabajar solo en el Ryzen (Miguel quiere usar ambos). |

### Discusiones y hallazgos
- Reparto sugerido: Celeron para documentación, bitácora, lectura, escritura de código y EDA con
  pandas/pyarrow; Ryzen para todo lo que ejecute JAX/DESC. El código JAX escrito en el Celeron se
  prueba en el Ryzen tras un pull.
- El conflicto más probable entre PCs es `docs/BITACORA.md` (se toca en cada commit): hacer pull
  antes de editarla.

### Próximos pasos (Fase 0)
- [ ] Activar `git config pull.rebase true` también en el PC Celeron.
- [ ] Siguen pendientes los de la Entrada 2 (CI, notebook de Colab, `docs/COMPUTE.md`, pregunta de
  reglas a los organizadores). El notebook de verificación de JAX/DESC puede prototiparse ya en el
  PC Ryzen.

### Modelo(s) de Claude usados
- Sonnet 5.5 (sesión principal en el PC Ryzen; sin subagentes).

---

## Entrada 5 — 2026-11-26 — Fase 0: verificación del pipeline, CI y roadmap v1.3

**Fecha:** 2026-11-26
**Sesión / Fase:** sesión 3, Fase 0
**Commits:** `6d9df58` docs: describe both local PCs and sync workflow; add journal entry 4 (commit
de la bitácora de la Entrada 4); `c0bb355` feat(scripts): add environment check and in-process
quick-check scoring; `39f5ce3` ci: add GitHub Actions workflow with lint and test jobs; más el commit
que añade esta entrada, ROADMAP v1.3 y cambios de `CLAUDE.md` (su hash se cita en la Entrada 6).

### Qué se hizo
- **Revisión completa del repo** con Miguel (qué es el reto, dataset, scoring, plan) y aclaraciones:
  estructura del dataset, "un solo forward pass", tracks del benchmark vs. tracks del proyecto.
- **Dataset `results` descargado en el PC Ryzen** (revisión fijada por upstream `f505cc8`). Recuentos
  verificados: 266,549 muestras = pares (frontera, requisitos) sobre 14,496 fronteras; **todas**
  tienen coilset REGCOIL; 177,341 (66.5 %) tienen además coilset DESC (~12.3 por frontera);
  138,711 coilsets DESC cumplen `meets_relative_error_thresholds`. Retención DESC por N_FP:
  2 → 99 %, 3 → 86 %, 4 → 52 %, 5 → 36 %. Track fijo 53,206 / 2,771 (coincide con el paper).
- **`scripts/check_env.py`**: evalúa con DESC (mismo código que el leaderboard) el coilset de
  referencia de filas del split eval y comprueba que reproduce sus objetivos. Resultado en 2 filas
  tight de `fixed_shape`: error relativo ≤ 8e−15 en las 4 métricas, S = 1, strict. **El entorno y el
  pipeline de evaluación son exactos.** Coste en el PC Ryzen: primera instancia ~23 s (reconstruir
  equilibrio) + ~76 s (métricas, incluye compilación JIT); siguientes en el mismo proceso ~3 s + ~6 s;
  pico de memoria ~3.7 GB.
- **`examples/run_benchmark_scoring.py` de upstream está roto**: el subproceso de cada instancia
  hace `from coilstellaration.examples import ...`, módulo que no existe en el paquete instalado,
  así que todas las instancias fallan y puntúan 0 en silencio (el resumen muestra 0.0 sin error).
  Sigue igual en el `main` de upstream a 2026-11-26.
- **`scripts/quickcheck_score.py`**: sustituto propio, en un solo proceso, que sigue los 6 pasos del
  README de upstream (cargar filas, predecir, métricas DESC, score). Primera instancia tight con el
  checkpoint MLP incluido: **S = 5.8e−3**, violaciones d_CC 0, d_CP 0, κ 0, **ε 5.15**: el modo de
  fallo dominante descrito en el paper. Las instancias siguientes no se pudieron evaluar por memoria
  (ver hallazgos).
- **CI en GitHub Actions** (`.github/workflows/ci.yml`): job `lint` (pre-commit con `uvx`, sin
  construir el entorno) y job `tests` (submódulo, dependencias de sistema, `uv sync --locked`,
  nuestros tests y `scoring_test.py` de upstream). Acciones fijadas por hash como en upstream.
- **ROADMAP v1.3**: PC Ryzen en §6.1 y Fase 0; casillas de la Fase 0 marcadas con sus commits;
  primera medida del coste de DESC en la Fase 2; calendario recalibrado; supuesto 2 parcialmente
  verificado; respuestas a preguntas ya resueltas (repo público, usuario de GitHub).
- `CLAUDE.md`: comandos nuevos, aviso del bug de upstream, cambio de ranking del leaderboard, límite
  de memoria del PC Ryzen.

### Decisiones
| ID | Decisión | Motivo | Alternativas descartadas |
|---|---|---|---|
| D-014 | **Scoring local con script propio en un solo proceso** (`scripts/quickcheck_score.py`) en lugar de `run_benchmark_scoring.py` de upstream; no se parchea el submódulo. Es el germen del harness de la Fase 2. | El script de upstream no evalúa nada en el pin actual; además, un solo proceso reutiliza la compilación JIT (~10 s/instancia en vez de ~100 s). | Parchear upstream en el submódulo (prohibido por D-002); subir el pin (el bug sigue en `main`). |
| D-015 | **Calendario recalibrado +2 semanas** (M0 → 2026-12-06, M9 → 2027-07-04), conservando las fechas originales entre paréntesis. | M0 venció el 2026-11-23 con tareas abiertas; mejor un plan realista que uno ignorado. | Mantener fechas (dejarían de servir para decidir); replanificar a fondo sin conocer aún las horas semanales reales. |
| D-016 | **CI con dos jobs** (lint rápido + tests con entorno completo) en cada push a `main` y en cada PR; versiones de acciones fijadas por hash. | Detecta en minutos errores de formato, de dependencias o de tests en una máquina limpia, independiente de los dos PCs; los runners de GitHub tienen AVX, así que también prueban upstream. | Solo lint (no detectaría entornos rotos); ejecutar DESC pesado en CI (lento y caro para cada push). |

### Discusiones y hallazgos
- **Cambio de ranking del leaderboard** (upstream `c60811b`, 2026-11-16; `36212bb`, 2026-11-13): la
  portada ordena ahora por **frente de Pareto de score (strict o media) vs. coste (tiempo medio de
  inferencia por instancia o nº de parámetros)**; la tabla solo por score pasa a una pestaña aparte.
  Encaja con el hilo conductor del ROADMAP (frontera calidad vs. cómputo) y quita ventaja automática
  a los optimizadores lentos dentro del grafo. Nuestro pin (`5b925cf`) aún no incluye ese código;
  subir el pin es una decisión pendiente (solo afecta a `leaderboard/`).
- **Memoria del PC Ryzen**: con el escritorio abierto quedan ~3.6 GB libres y el swap es de 512 MB;
  la evaluación DESC (~3.7–3.9 GB) provoca que el sistema mate el proceso (OOM) **sin mensaje de
  error en la salida**: la salida simplemente se corta (comprobar con `journalctl -k | grep -i oom`).
  Ocurrió 3 veces. Solución propuesta: ampliar el swapfile a 8 GB.
- Pendientes para el EDA: 102,641 filas sin estrato (89,208 sin coilset DESC + 13,433 con DESC);
  el track variable da 168,576 / 8,765 filas por la columna `benchmark/variable_shape_track` frente a
  155,850 / 8,058 del paper (probablemente por el filtro `meets_relative_error_thresholds`).
- Posible contribución a upstream: abrir un issue sobre el bug de `run_benchmark_scoring.py`
  ("aportar antes de pedir", ROADMAP §8.3). Pendiente de que Miguel decida.

### Próximos pasos (Fase 0, M0 → 2026-12-06)
- [ ] Ampliar el swap del PC Ryzen y repetir `quickcheck_score.py --n 4` para tener el score de las
  4 instancias y el coste en caliente.
- [ ] Comprobar que la CI pasa en GitHub.
- [ ] Cuenta y token de Hugging Face; cuenta de W&B.
- [ ] Lectura del paper con notas propias (`docs/notes/00_paper_dudas.md`).
- [ ] `docs/COMPUTE.md` (acceso al cluster) y notebook de arranque en Colab.
- [ ] Publicar la pregunta de reglas (revisarla a la luz del ranking por Pareto) y, si Miguel quiere,
  el issue del bug.
- [ ] Captura del leaderboard en `docs/reports/leaderboard_snapshots/`.

### Modelo(s) de Claude usados
- Sonnet 5.5 (sesión principal en el PC Ryzen; sin subagentes).
