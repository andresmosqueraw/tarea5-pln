#!/bin/bash
# Script para ejecutar el notebook de forma persistente
# Los resultados se guardarán directamente en solution.ipynb
# Muestra errores en consola y tiempo transcurrido
# Uso: ./ejecutar_persistente.sh

cd /home/estudiante/punto2/mi-solution

# Activar el entorno virtual si existe
if [ -d "/home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv" ]; then
    source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
fi

# Mostrar tiempo de inicio
echo "=========================================="
echo "INICIANDO EJECUCIÓN DEL NOTEBOOK"
echo "=========================================="
echo "Fecha/Hora de inicio: $(date)"
echo "Los resultados se guardarán en solution.ipynb"
echo "Modo debug activado - los errores aparecerán en consola"
echo "=========================================="
echo ""

# Guardar tiempo de inicio para cálculo
START_TIME=$(date +%s)

# Ejecutar el notebook con debug y mostrar tiempo
# El flag --debug muestra información detallada y errores en consola
time jupyter nbconvert --to notebook --execute solution.ipynb --inplace --debug 2>&1

# Capturar código de salida
EXIT_CODE=$?

# Calcular tiempo transcurrido
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ EJECUCIÓN COMPLETADA EXITOSAMENTE"
else
    echo "✗ EJECUCIÓN TERMINÓ CON ERRORES (código: $EXIT_CODE)"
fi
echo "=========================================="
echo "Fecha/Hora de finalización: $(date)"
echo "Tiempo total transcurrido: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

# Salir con el código de error del comando
exit $EXIT_CODE

