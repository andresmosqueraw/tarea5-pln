#!/bin/bash
# Script para ejecutar el notebook de forma persistente
# Los resultados se guardarán directamente en solution.ipynb
# Uso: ./ejecutar_persistente.sh

cd /home/estudiante/punto2/mi-solution

# Activar el entorno virtual si existe
if [ -d "/home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv" ]; then
    source /home/estudiante/tldr-uniandes/encoders-vs-decoders-classification/venv/bin/activate
fi

# Ejecutar el notebook - los resultados se guardan en el mismo archivo
echo "Iniciando ejecución del notebook..."
echo "Los resultados se guardarán en solution.ipynb"
jupyter nbconvert --to notebook --execute solution.ipynb --inplace
echo "Ejecución completada. Revisa solution.ipynb para ver los resultados."

