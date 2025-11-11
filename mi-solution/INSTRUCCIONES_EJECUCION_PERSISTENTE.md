# Instrucciones para Ejecutar el Notebook de Forma Persistente
# Los resultados se guardarán directamente en el notebook (.ipynb)

## ⭐ Opción 1: Ejecutar Notebook con SCREEN (Recomendado)

Esta opción ejecuta el notebook y guarda TODOS los resultados (outputs, gráficos, métricas) directamente en el archivo `solution.ipynb`.

### Iniciar una sesión screen:
```bash
screen -S ejecucion_notebook
```

### Dentro de screen, ejecutar el notebook:
```bash
cd /home/estudiante/punto2/mi-solution
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
jupyter nbconvert --to notebook --execute solution.ipynb --inplace
```

**Importante:** El flag `--inplace` hace que los resultados se guarden en el mismo archivo `solution.ipynb`. Cuando termine, podrás abrir el notebook y ver todos los outputs.

### Para desconectarte de screen (sin detener la ejecución):
Presiona: `Ctrl+A` luego `D` (detach)

### Para reconectarte a la sesión:
```bash
screen -r ejecucion_notebook
```

### Para ver todas las sesiones screen:
```bash
screen -ls
```

### Para matar una sesión screen:
```bash
screen -X -S ejecucion_notebook quit
```

---

## Opción 2: Ejecutar Notebook con TMUX (Alternativa a screen)

### Iniciar una sesión tmux:
```bash
tmux new -s ejecucion_notebook
```

### Dentro de tmux, ejecutar el notebook:
```bash
cd /home/estudiante/punto2/mi-solution
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
jupyter nbconvert --to notebook --execute solution.ipynb --inplace
```

### Para desconectarte de tmux:
Presiona: `Ctrl+B` luego `D` (detach)

### Para reconectarte a la sesión:
```bash
tmux attach -t ejecucion_notebook
```

### Para ver todas las sesiones tmux:
```bash
tmux ls
```

### Para matar una sesión tmux:
```bash
tmux kill-session -t ejecucion_notebook
```

---

## Opción 3: Ejecutar Notebook con NOHUP (Más simple, pero menos control)

### Ejecutar directamente con nohup:
```bash
cd /home/estudiante/punto2/mi-solution
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
nohup jupyter nbconvert --to notebook --execute solution.ipynb --inplace > ejecucion.log 2>&1 &
```

### Ver el progreso en tiempo real:
```bash
tail -f ejecucion.log
```

### Ver el proceso:
```bash
ps aux | grep nbconvert
```

### Detener el proceso:
```bash
# Encontrar el PID
ps aux | grep nbconvert
# Matar el proceso (reemplaza PID con el número)
kill PID
```

---

## Opción 4: Crear una copia del notebook antes de ejecutar (Recomendado para seguridad)

Si quieres mantener el notebook original intacto y crear uno con resultados:

```bash
cd /home/estudiante/punto2/mi-solution
cp solution.ipynb solution_ejecutado.ipynb
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
jupyter nbconvert --to notebook --execute solution_ejecutado.ipynb --inplace
```

O con screen:
```bash
screen -S ejecucion_notebook
cd /home/estudiante/punto2/mi-solution
cp solution.ipynb solution_ejecutado.ipynb
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
jupyter nbconvert --to notebook --execute solution_ejecutado.ipynb --inplace
```

---

## Recomendación

**Usa SCREEN con `jupyter nbconvert` (Opción 1)** porque:
- ✅ Ejecuta el notebook completo y guarda TODOS los resultados en el archivo .ipynb
- ✅ Puedes reconectarte y ver el progreso en tiempo real
- ✅ Puedes interactuar si es necesario
- ✅ Cuando termine, simplemente abres el notebook y ves todos los outputs guardados
- ✅ No necesitas redirigir output manualmente

### Comandos rápidos de SCREEN:
- `Ctrl+A` luego `D` = Desconectar (detach) - **La ejecución continúa**
- `Ctrl+A` luego `C` = Crear nueva ventana
- `Ctrl+A` luego `N` = Siguiente ventana
- `Ctrl+A` luego `[` = Modo scroll (usar flechas, `q` para salir)

# 1. Crear sesión screen
screen -S ejecucion_notebook

# 2. Dentro de screen, ejecutar:
cd /home/estudiante/punto2/mi-solution
source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
jupyter nbconvert --to notebook --execute solution.ipynb --inplace

# 3. Desconectarte (Ctrl+A luego D) - la ejecución continúa
# 4. Reconectarte después: screen -r ejecucion_notebook
# Eliminar una sesión específica
screen -X -S <nombre_o_pid> quit

### Después de la ejecución:
Una vez que termine la ejecución, simplemente abre `solution.ipynb` en Jupyter y verás:
- ✅ Todos los outputs de las celdas
- ✅ Gráficos y visualizaciones
- ✅ Resultados de métricas
- ✅ Todo guardado directamente en el notebook

