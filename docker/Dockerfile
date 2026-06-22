# Imagen base oficial de Python, versión slim para reducir tamaño.
# "slim" excluye herramientas de compilación y documentación que no necesitamos.
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor.
# Todos los COPY y CMD se ejecutan relativos a esta ruta.
WORKDIR /app

# Copiamos primero solo el requirements.txt para aprovechar la caché de Docker.
# Si el código cambia pero las dependencias no, Docker no reinstala todo.
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la API y los modelos serializados.
COPY api/ ./api/
COPY models/ ./models/

# Puerto que expone la API dentro del contenedor.
EXPOSE 8000

# Comando que se ejecuta al arrancar el contenedor.
# --host 0.0.0.0 es necesario para que el puerto sea accesible desde fuera del contenedor.
# Sin esto, uvicorn escucha solo en localhost interno y el contenedor no responde.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]