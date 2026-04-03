# ModelGenerator

Aplicacion local para generar modelos 3D a partir de prompts de texto. Pipeline completo: texto → imagen → modelo 3D → texturizado → exportacion GLB → visor web.

## Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Frontend      │────▶│   Backend API    │────▶│   Worker       │
│   Next.js       │     │   FastAPI        │     │   Pipeline ML  │
│   :3000         │     │   :8000          │     │                │
└─────────────────┘     └──────────────────┘     └────────────────┘
                              │                        │
                              ▼                        ▼
                        ┌──────────┐           ┌──────────────┐
                        │  SQLite  │           │   Storage    │
                        │  (jobs)  │           │  (archivos)  │
                        └──────────┘           └──────────────┘
```

### Pipeline de generacion

```
1. Prompt de texto
2. Text-to-Image (SDXL por defecto)
3. Image-to-3D (TripoSR)
4. Texturizado basico (proyeccion UV)
5. Exportacion a GLB
6. Visor web con Three.js
```

### Estructura del proyecto

```
modelgenerator/
├── backend/
│   └── app/
│       ├── api/           # Endpoints REST
│       │   ├── jobs.py    # CRUD de jobs
│       │   ├── files.py   # Servir assets
│       │   └── health.py  # Health check
│       ├── core/          # Config y logging
│       ├── db/            # SQLAlchemy + SQLite
│       ├── models/        # Modelos de DB
│       ├── schemas/       # Pydantic schemas
│       ├── services/      # Interfaces + implementaciones ML
│       │   ├── base.py    # ABCs (TextToImageService, etc.)
│       │   ├── text_to_image.py  # SDXL + Mock
│       │   ├── image_to_3d.py    # TripoSR + Mock
│       │   ├── texturing.py      # UV projection + passthrough
│       │   ├── export.py         # OBJ/GLB/STL via trimesh
│       │   ├── storage.py        # Asset storage local
│       │   └── factory.py        # Factory con autodeteccion
│       ├── workers/       # Worker de procesamiento
│       │   ├── orchestrator.py   # Pipeline completo
│       │   └── runner.py         # Loop principal del worker
│       └── main.py        # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── app/           # Next.js App Router pages
│       │   ├── page.tsx   # Home + formulario
│       │   ├── jobs/      # Historial
│       │   └── job/[id]/  # Detalle + visor 3D
│       ├── components/    # React components
│       │   ├── ModelViewer.tsx    # Three.js GLB viewer
│       │   ├── PromptForm.tsx    # Formulario de generacion
│       │   ├── JobCard.tsx       # Tarjeta de job
│       │   └── StatusBadge.tsx   # Badge de estado
│       ├── lib/           # API client
│       └── types/         # TypeScript types
├── storage/               # Archivos generados
│   ├── images/
│   ├── models/
│   └── exports/
├── scripts/               # Scripts de arranque
├── docker-compose.yml
├── .env.example
└── README.md
```

## Requisitos previos

- **Python 3.11+**
- **Node.js 20+**
- **NVIDIA GPU** con CUDA 12.x (RTX 3090/4090/5090 recomendada)
- **~20GB de espacio** para modelos ML
- **~16GB+ VRAM** para SDXL + TripoSR

## Instalacion

### 1. Clonar y configurar

```bash
cd modelgenerator
cp .env.example .env
# Editar .env si necesitas cambiar algo
```

### 2. Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar PyTorch con CUDA (ajustar version de CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

### 4. TripoSR (opcional pero recomendado)

```bash
# Clonar TripoSR
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -e .
```

Sin TripoSR, el sistema usara un mock que genera cubos. La imagen se genera con SDXL si esta disponible, o con un mock si no.

## Ejecucion local

Necesitas 3 terminales:

```bash
# Terminal 1: Backend API
./scripts/start-backend.sh
# o manualmente: cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Worker
./scripts/start-worker.sh
# o manualmente: cd backend && python -m app.workers.runner

# Terminal 3: Frontend
./scripts/start-frontend.sh
# o manualmente: cd frontend && npm run dev
```

Acceder a: **http://localhost:3000**

API docs: **http://localhost:8000/docs**

### Con Docker

```bash
docker compose up --build
```

> Requiere nvidia-docker/nvidia-container-toolkit para acceso a GPU.

## Configuracion de entorno

Todas las variables estan en `.env`. Las mas importantes:

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `TEXT_TO_IMAGE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | Modelo HuggingFace para text-to-image |
| `TEXT_TO_IMAGE_DEVICE` | `cuda` | Device (cuda/cpu) |
| `TRIPOSR_MODEL` | `stabilityai/TripoSR` | Modelo para image-to-3D |
| `TRIPOSR_MC_RESOLUTION` | `256` | Resolucion del marching cubes |
| `TEXTURING_ENABLED` | `true` | Habilitar texturizado basico |
| `EXPORT_FORMAT` | `glb` | Formato de exportacion |
| `IMAGE_NUM_STEPS` | `30` | Steps de difusion (default) |

## API REST

Base: `http://localhost:8000/api/v1`

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| `GET` | `/health` | Health check + info GPU |
| `POST` | `/jobs` | Crear job de generacion |
| `GET` | `/jobs` | Listar jobs (paginado) |
| `GET` | `/jobs/{id}` | Detalle de un job |
| `DELETE` | `/jobs/{id}` | Eliminar job completado/fallido |
| `POST` | `/jobs/{id}/retry` | Reintentar job fallido |
| `GET` | `/files/{path}` | Descargar archivo generado |

### Crear job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A medieval sword with gemstones", "num_steps": 30}'
```

### Respuesta

```json
{
  "id": 1,
  "prompt": "A medieval sword with gemstones",
  "status": "pending",
  "image_url": null,
  "export_url": null,
  "created_at": "2024-01-01T00:00:00"
}
```

## Flujo de trabajo

1. El usuario escribe un prompt en el frontend
2. Se crea un job via `POST /api/v1/jobs`
3. El worker detecta el job pendiente
4. Pipeline: `pending → generating_image → image_ready → generating_model → model_ready → texturing → exporting → completed`
5. El frontend hace polling cada 3s para actualizar el estado
6. Cuando el job se completa, se muestra la imagen y el visor 3D
7. El usuario puede descargar los archivos

## Como cambiar el modelo text-to-image

1. **Cambiar variable de entorno:**
   ```
   TEXT_TO_IMAGE_MODEL=runwayml/stable-diffusion-v1-5
   ```

2. **O crear un nuevo servicio:** Implementar `TextToImageService` de `app/services/base.py`:
   ```python
   from app.services.base import TextToImageService

   class MyCustomService(TextToImageService):
       def load_model(self): ...
       def generate(self, prompt, ...): ...
       def unload_model(self): ...
   ```

3. **Registrar en factory:** Editar `app/services/factory.py` para usar tu servicio.

## Como integrar TripoSR

1. Clonar e instalar:
   ```bash
   git clone https://github.com/VAST-AI-Research/TripoSR.git
   cd TripoSR && pip install -e .
   ```

2. El factory lo detecta automaticamente. Si `import tsr` funciona y hay CUDA, usara TripoSR.

3. La primera ejecucion descargara los pesos (~1GB) de HuggingFace.

4. Si la API de TripoSR cambia, ajustar `app/services/image_to_3d.py`.

**Punto de integracion exacto:** `TripoSRImageTo3DService.generate()` en `app/services/image_to_3d.py`. La clase `TSR` se importa de `tsr.system`. Si tu version tiene una API diferente, este es el unico archivo que necesitas tocar.

## Problemas comunes

### "CUDA out of memory"
- Reducir `IMAGE_WIDTH`/`IMAGE_HEIGHT` a 768 o 512
- Reducir `TRIPOSR_MC_RESOLUTION` a 128
- No ejecutar ambos modelos a la vez (el worker los carga secuencialmente)
- Con RTX 5090 (32GB VRAM) no deberia pasar con config por defecto

### "Mock service" en los logs
- Normal si no tienes GPU o las dependencias ML no estan instaladas
- El sistema funciona con mocks para desarrollo/testing

### El frontend no conecta con el backend
- Verificar que el backend esta en `localhost:8000`
- El proxy se configura en `frontend/next.config.ts`
- CORS esta habilitado para `localhost:3000`

### TripoSR no se detecta
- Verificar: `python -c "import tsr; print('OK')"`
- Instalar desde el repo: `pip install -e .` dentro del directorio de TripoSR

### El visor 3D no carga el modelo
- Verificar que el export es GLB (default)
- Probar acceder directamente a la URL del archivo en el navegador
- Abrir consola del navegador para ver errores de Three.js

## Siguientes pasos

- [ ] SSE (Server-Sent Events) para updates en tiempo real en vez de polling
- [ ] Cola de jobs con prioridad (Redis/RQ)
- [ ] Batch processing
- [ ] Galeria publica de modelos generados
- [ ] Texturizado avanzado con TEXTure o Text2Tex
- [ ] Soporte para Flux, Kandinsky, u otros modelos text-to-image
- [ ] Comparacion side-by-side de diferentes generaciones
- [ ] Export a formatos adicionales (USDZ para iOS, FBX)
- [ ] Estimacion de tiempo de procesamiento
- [ ] Tests automatizados (pytest backend, vitest frontend)
- [ ] CI/CD pipeline
