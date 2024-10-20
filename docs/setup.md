
# Guía de Configuración

## 1. Clonar el Repositorio
Primero, clona el repositorio a tu máquina local:

```bash
git clone https://github.com/usuario/proyecto-mlops.git
cd proyecto-mlops
```

## 2. Crear un Entorno Virtual
Si estás usando `virtualenv` para manejar el entorno virtual, sigue estos pasos:

```bash
python3 -m venv env
source env/bin/activate   # En Windows: env\Scripts\activate
```

Si prefieres `conda`, puedes crear el entorno usando:

```bash
conda create --name mlops-env python=3.8
conda activate mlops-env
```

## 3. Instalar las Dependencias
Instala las dependencias necesarias para ejecutar el proyecto. Si usas `pip`, puedes ejecutar:

```bash
pip install -r requirements.txt
```

Si estás utilizando `conda`, puedes instalar las dependencias desde `environment.yml`:

```bash
conda env create -f environment.yml
```

## 4. Configuración de Herramientas
### DVC
Asegúrate de tener `dvc` instalado:

```bash
pip install dvc
```
Para inicializar DVC y conectarlo con el almacenamiento remoto:

```bash
dvc init
dvc remote add -d <nombre_remoto> <ruta_remota>
```

### MLflow
Si estás utilizando `MLflow`, asegúrate de instalarlo y configurarlo:

```bash
pip install mlflow
mlflow ui
```

Esto abrirá la interfaz de MLflow para rastrear los experimentos.
