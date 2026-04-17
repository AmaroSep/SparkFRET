# Instrucciones para publicar SparkFRET en GitHub

## Paso 1 — Crear cuenta en GitHub

1. Ve a https://github.com/signup
2. Elige un nombre de usuario (ej. `bhattlab` o tu nombre)
3. Confirma tu email

---

## Paso 2 — Crear el repositorio en GitHub

1. Inicia sesion en https://github.com
2. Click en el boton verde **New** (esquina superior izquierda)
3. Configura el repo:
   - **Repository name:** `SparkFRET`
   - **Description:** `Automated FRET sparkle detection and quantification pipeline`
   - **Visibility:** Public (para compartir con colaboradores) o Private
   - **IMPORTANTE:** NO marques ninguna opcion de inicializacion (sin README, sin .gitignore, sin licencia) — ya los tenemos listos
4. Click **Create repository**

---

## Paso 3 — Subir el codigo desde tu PC

Abre PowerShell o CMD y ejecuta estos comandos uno por uno:

```bat
cd D:\Cellpose\SparkFRET

git remote add origin https://github.com/TU_USUARIO/SparkFRET.git

git branch -M main

git push -u origin main
```

Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub.

GitHub te pedira usuario y contrasena la primera vez. Para la contrasena usa un **Personal Access Token** (ver Paso 3b).

---

## Paso 3b — Crear Personal Access Token (si pide contrasena)

GitHub ya no acepta la contrasena normal para git push. Necesitas un token:

1. GitHub → tu foto (esquina superior derecha) → **Settings**
2. Menu izquierdo → **Developer settings** (hasta abajo)
3. **Personal access tokens** → **Tokens (classic)**
4. **Generate new token (classic)**
5. Configuracion:
   - Note: `SparkFRET push`
   - Expiration: `No expiration` o 1 año
   - Scopes: marca solo **repo** (primer checkbox grande)
6. Click **Generate token**
7. **Copia el token** — solo se muestra una vez
8. Cuando git pida contrasena, pega el token

---

## Paso 4 — Subir el modelo entrenado como Release

Los modelos son demasiado grandes para git. Se distribuyen como Release:

1. En GitHub, abre tu repo `SparkFRET`
2. Click en **Releases** (columna derecha) → **Create a new release**
3. Configura:
   - **Tag:** `v1.0`
   - **Release title:** `SparkFRET v1.0 — sparkle_fret_v9`
   - **Description:**
     ```
     Modelo entrenado: sparkle_fret_v9
     - 500 epochs, loss final 0.0061
     - Parametros de inferencia: flow=0.8, cellprob=-3, upper_percentile=99
     ```
4. En **Attach binaries**: arrastra el archivo del modelo
   - Ruta: `D:\Cellpose\sparkle_fret_v9` (sin extension)
5. Click **Publish release**

---

## Paso 5 — Compartir con colaboradores

Para dar acceso a alguien al repo privado:
1. Tu repo → **Settings** → **Collaborators** → **Add people**
2. Ingresa su email o usuario de GitHub

Para repo publico, simplemente comparte la URL:
```
https://github.com/TU_USUARIO/SparkFRET
```

---

## Instrucciones para el colaborador (nuevo usuario)

Cuando alguien quiera instalar SparkFRET:

```bat
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/SparkFRET.git
cd SparkFRET

# 2. Instalar todo automaticamente
install.bat

# 3. Descargar el modelo desde Releases y copiarlo a la carpeta models/

# 4. Lanzar el hub
launch_hub.bat
```

---

## Actualizaciones futuras

Cada vez que hagas cambios y quieras subirlos a GitHub:

```bat
cd D:\Cellpose\SparkFRET

git add .
git commit -m "Descripcion del cambio"
git push
```

---

## Resumen rapido

| Paso | Accion |
|------|--------|
| 1 | Crear cuenta en github.com/signup |
| 2 | Crear repo `SparkFRET` (sin inicializar) |
| 3 | `git remote add origin ...` + `git push` |
| 3b | Crear Personal Access Token si pide contrasena |
| 4 | Subir modelo como Release v1.0 |
| 5 | Compartir URL con colaboradores |
