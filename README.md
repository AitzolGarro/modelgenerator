# ModelGenerator

Genera modelos 3D desde texto. Un solo comando para arrancar.

```
./setup.sh    # primera vez
./start.sh    # abrir http://localhost:8000
```

## Que hace

Escribes un prompt → genera una imagen → la convierte en modelo 3D → le aplica textura → la exporta a GLB → la ves en el navegador.

```
"A medieval sword" → [SDXL] → imagen → [TripoSR] → mesh 3D → [textura] → model.glb → visor web
```

## Inicio rapido

### Requisitos

- Python 3.11+
- Node.js 20+
- GPU NVIDIA con CUDA 12.x (RTX 3090/4090/5090)
- ~20GB disco para modelos ML

### Instalar y arrancar

```bash
git clone https://github.com/AitzolGarro/modelgenerator.git
cd modelgenerator
./setup.sh     # instala todo (python venv, npm, build frontend)
./start.sh     # arranca en http://localhost:8000
```

Eso es todo. `setup.sh` instala dependencias, `start.sh` arranca la app. Un solo proceso, un solo puerto.

### Con Docker

```bash
docker compose up --build
# → http://localhost:8000
```

### Con Make

```bash
make setup     # instalar todo
make run       # arrancar
make dev       # modo desarrollo (hot-reload frontend en :3000)
make build     # rebuild frontend
make clean     # limpiar archivos generados
```

## Arquitectura

```
┌─────────────────────────────────────────┐
│          http://localhost:8000           │
│                                         │
│  ┌──────────┐  ┌─────┐  ┌───────────┐  │
│  │ Frontend  │  │ API │  │  Worker   │  │
│  │ (static)  │  │REST │  │ (thread)  │  │
│  └──────────┘  └─────┘  └───────────┘  │
│                   │            │        │
│              ┌────┴────┐  ┌───┴─────┐  │
│              │ SQLite  │  │ Storage │  │
│              └─────────┘  └─────────┘  │
└─────────────────────────────────────────┘
```

Todo corre en **un solo proceso** FastAPI:
- El frontend se sirve como archivos estaticos
- La API REST gestiona los jobs
- El worker ML corre como thread daemon en background
- No hay Redis, ni Celery, ni procesos extra

### Pipeline

| Paso | Estado del job | Que pasa |
|------|---------------|----------|
| 1 | `pending` | Job creado, en cola |
| 2 | `generating_image` | SDXL genera imagen de referencia |
| 3 | `image_ready` | Imagen lista |
| 4 | `generating_model` | TripoSR convierte imagen a mesh 3D |
| 5 | `model_ready` | Mesh generado |
| 6 | `texturing` | Proyeccion UV basica |
| 7 | `exporting` | Export a GLB |
| 8 | `completed` | Listo para ver y descargar |

### Estructura

```
modelgenerator/
├── setup.sh              ← instalar todo
├── start.sh              ← arrancar la app
├── Makefile              ← make setup / make run
├── Dockerfile            ← contenedor unico
├── docker-compose.yml
├── .env.example
├── backend/
│   └── app/
│       ├── main.py           ← entry point (API + worker + frontend)
│       ├── api/              ← endpoints REST
│       ├── core/             ← config, logging
│       ├── db/               ← SQLAlchemy + SQLite
│       ├── models/           ← modelos de DB
│       ├── schemas/          ← Pydantic
│       ├── services/         ← interfaces ML desacopladas
│       │   ├── base.py       ← ABCs (swap cualquier modelo)
│       │   ├── factory.py    ← autodeteccion GPU/deps
│       │   ├── text_to_image.py  ← SDXL + mock
│       │   ├── image_to_3d.py    ← TripoSR + mock
│       │   ├── texturing.py      ← UV projection
│       │   ├── export.py         ← GLB/OBJ via trimesh
│       │   └── storage.py        ← filesystem
│       └── workers/
│           ├── background.py     ← thread daemon
│           ├── orchestrator.py   ← pipeline completo
│           └── runner.py         ← modo standalone (opcional)
└── frontend/
    └── src/
        ├── app/              ← paginas Next.js
        ├── components/       ← ModelViewer, PromptForm, etc.
        ├── lib/api.ts        ← cliente API
        └── types/            ← TypeScript types
```

## API

Base: `http://localhost:8000/api/v1`

| Metodo | Endpoint | Que hace |
|--------|----------|----------|
| `GET` | `/health` | Estado + info GPU |
| `POST` | `/jobs` | Crear job |
| `GET` | `/jobs` | Listar jobs |
| `GET` | `/jobs/{id}` | Detalle de job |
| `DELETE` | `/jobs/{id}` | Eliminar job |
| `POST` | `/jobs/{id}/retry` | Reintentar job fallido |
| `GET` | `/files/{path}` | Descargar archivo |

Docs interactivos: http://localhost:8000/docs

```bash
# Crear un job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A medieval sword with gemstones"}'
```

## Cambiar modelos ML

### Text-to-Image

Opcion 1 — cambiar modelo de HuggingFace en `.env`:
```
TEXT_TO_IMAGE_MODEL=runwayml/stable-diffusion-v1-5
```

Opcion 2 — crear tu propio servicio:
```python
# backend/app/services/text_to_image.py
from app.services.base import TextToImageService

class MyCustomService(TextToImageService):
    def load_model(self): ...
    def generate(self, prompt, ...): ...
    def unload_model(self): ...
```

Registrarlo en `backend/app/services/factory.py`.

### Image-to-3D (TripoSR)

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR && pip install -e .
```

Se detecta automaticamente. Sin el, se usa un mock (genera cubos).

Punto de integracion: `TripoSRImageTo3DService.generate()` en `backend/app/services/image_to_3d.py`.

## Configuracion

Todo en `.env`. Los defaults funcionan bien con una RTX 5090.

| Variable | Default | Notas |
|----------|---------|-------|
| `TEXT_TO_IMAGE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | Cualquier modelo de diffusers |
| `TRIPOSR_MC_RESOLUTION` | `256` | Bajar a 128 si falta VRAM |
| `IMAGE_NUM_STEPS` | `30` | Mas steps = mejor calidad, mas lento |
| `EXPORT_FORMAT` | `glb` | `glb`, `obj` |
| `TEXTURING_ENABLED` | `true` | Desactivar si solo quieres el mesh |

## Problemas comunes

**"CUDA out of memory"** — Bajar `IMAGE_WIDTH`/`IMAGE_HEIGHT` a 512, `TRIPOSR_MC_RESOLUTION` a 128.

**"Mock service"** — Normal sin GPU. Todo funciona, solo genera imagenes y cubos de prueba.

**Frontend no carga** — Verificar que se hizo build: `cd frontend && npm run build`. Si falta `/frontend/out/`, el backend muestra JSON en vez del UI.

**TripoSR no detectado** — `python -c "import tsr"` debe funcionar. Si no, instalar desde el repo.
