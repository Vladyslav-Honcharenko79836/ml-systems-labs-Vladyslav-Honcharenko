# ML Systems Labs

Individual lab work for *Introduction to Machine Learning Systems* (V. J. Reddi).

| Folder  | Topic                                        |
|---------|----------------------------------------------|
| `lab01` | Environment and first system measurements   |
| `lab02` | A neural network from scratch (NumPy)       |
| `lab03` | Architectures and their costs (MLP/CNN/attention) |

## Environment

Python **3.11.8**, packages pinned in `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
```

### Known conflict in the pinned set

`mlflow==2.14.1` declares `pyarrow<16`, but the course pins `pyarrow==16.1.0`,
so a plain `pip install -r requirements.txt` fails with `ResolutionImpossible`.
To keep every pinned version exactly, install mlflow without dependency
resolution and install its other dependencies alongside the rest:

```bash
grep -v '^mlflow' requirements.txt > req_no_mlflow.txt
pip install -r req_no_mlflow.txt "Flask<4" "alembic!=1.10.0,<2" "cachetools>=5.0.0,<6" \
  "click>=7.0,<9" "cloudpickle<4" "docker>=4.0.0,<8" "entrypoints<1" "gitpython>=3.1.9,<4" \
  "graphene<4" "importlib-metadata>=3.7.0,!=4.7.0,<8" "markdown>=3.3,<4" \
  "opentelemetry-api>=1.9.0,<3" "opentelemetry-sdk>=1.9.0,<3" "packaging<25" \
  "protobuf>=3.12.0,<5" "pytz<2025" "pyyaml>=5.1,<7" "querystring-parser<2" \
  "sqlalchemy>=1.4.0,<3" "sqlparse>=0.4.0,<1" "Jinja2>=3.0,<4" "waitress<4"
pip install --no-deps mlflow==2.14.1
```

Verify with `python lab01/src/print_versions.py` — every package must show `OK`.

## Data

Datasets are downloaded by the scripts into `data/` (git-ignored).
