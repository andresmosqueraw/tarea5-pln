#!/bin/bash
# Script para verificar si el notebook se está ejecutando en screen
# sin necesidad de entrar a la sesión de screen
# Uso: ./verificar_ejecucion.sh

echo "=========================================="
echo "VERIFICANDO ESTADO DE EJECUCIÓN"
echo "=========================================="
echo ""

# 1. Verificar sesiones de screen activas
echo "1. SESIONES DE SCREEN:"
echo "----------------------"
SCREEN_SESSIONS=$(screen -ls 2>/dev/null | grep ejecucion_notebook)
if [ -z "$SCREEN_SESSIONS" ]; then
    echo "   ✗ No hay sesiones de screen 'ejecucion_notebook' activas"
else
    echo "   ✓ Sesiones encontradas:"
    echo "$SCREEN_SESSIONS" | while read line; do
        echo "     $line"
    done
fi
echo ""

# 2. Verificar si hay procesos de nbconvert corriendo
echo "2. PROCESOS DE NBCONVERT:"
echo "-------------------------"
NB_CONVERT_PROC=$(ps aux | grep "nbconvert.*solution.ipynb" | grep -v grep)
if [ -z "$NB_CONVERT_PROC" ]; then
    echo "   ✗ No hay procesos de nbconvert ejecutándose"
else
    echo "   ✓ Proceso encontrado:"
    echo "$NB_CONVERT_PROC" | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        TIME=$(echo "$line" | awk '{print $10}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        echo "     PID: $PID | CPU: $TIME%"
        echo "     Comando: $CMD"
        
        # Ver tiempo transcurrido del proceso
        ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
        if [ ! -z "$ELAPSED" ]; then
            echo "     Tiempo transcurrido: $ELAPSED"
        fi
    done
fi
echo ""

# 3. Verificar última modificación del notebook
echo "3. ESTADO DEL NOTEBOOK:"
echo "-----------------------"
NOTEBOOK_FILE="/home/estudiante/punto2/mi-solution/solution.ipynb"
if [ -f "$NOTEBOOK_FILE" ]; then
    LAST_MODIFIED=$(stat -c %y "$NOTEBOOK_FILE" 2>/dev/null)
    FILE_SIZE=$(du -h "$NOTEBOOK_FILE" | cut -f1)
    echo "   ✓ Archivo existe"
    echo "   Última modificación: $LAST_MODIFIED"
    echo "   Tamaño: $FILE_SIZE"
    
    # Verificar si se modificó recientemente (últimos 5 minutos)
    MODIFIED_SECONDS=$(stat -c %Y "$NOTEBOOK_FILE" 2>/dev/null)
    CURRENT_SECONDS=$(date +%s)
    DIFF=$((CURRENT_SECONDS - MODIFIED_SECONDS))
    
    if [ $DIFF -lt 300 ]; then
        echo "   ⚠ Archivo modificado hace menos de 5 minutos - proceso probablemente activo"
    else
        echo "   ⚠ Archivo no modificado recientemente - proceso puede estar detenido o en celda larga"
    fi
else
    echo "   ✗ Archivo no encontrado: $NOTEBOOK_FILE"
fi
echo ""

# 4. Verificar procesos de Python relacionados
echo "4. PROCESOS DE PYTHON RELACIONADOS:"
echo "------------------------------------"
PYTHON_PROCS=$(ps aux | grep -E "python.*jupyter|python.*nbconvert" | grep -v grep)
if [ -z "$PYTHON_PROCS" ]; then
    echo "   ✗ No hay procesos de Python/Jupyter relacionados"
else
    echo "   ✓ Procesos encontrados:"
    echo "$PYTHON_PROCS" | head -3 | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}' | cut -c1-80)
        ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
        echo "     PID: $PID | Tiempo: $ELAPSED"
        echo "     $CMD..."
    done
fi
echo ""

# 5. Resumen
echo "=========================================="
echo "RESUMEN:"
echo "=========================================="

# Determinar estado
if [ ! -z "$NB_CONVERT_PROC" ]; then
    echo "✓ PROCESO ACTIVO - El notebook se está ejecutando"
    PID=$(echo "$NB_CONVERT_PROC" | head -1 | awk '{print $2}')
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
    echo "  Tiempo transcurrido: $ELAPSED"
    echo ""
    echo "Para ver el progreso, reconéctate con:"
    echo "  screen -r ejecucion_notebook"
elif [ ! -z "$SCREEN_SESSIONS" ]; then
    echo "⚠ SCREEN ACTIVO pero no se detecta proceso nbconvert"
    echo "  Puede estar en pausa o esperando input"
    echo ""
    echo "Para verificar, reconéctate con:"
    echo "  screen -r ejecucion_notebook"
else
    echo "✗ NO HAY PROCESO ACTIVO"
    echo "  El notebook no se está ejecutando actualmente"
fi
echo "=========================================="


