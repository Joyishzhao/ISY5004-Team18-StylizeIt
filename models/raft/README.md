# RAFT optical flow

Place the pretrained weights here:

```
models/raft/
├── raft-things.pth       # FlyingThings3D — best generalization
└── raft-sintel.pth       # Sintel — best on movie-like footage
```

Two ways to get them:

**Option A — official release**:

```bash
curl -L -o models/raft/raft-things.pth \
  https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth
```

**Option B — torchvision** (no manual download, recommended):

```python
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=True).eval()
```

Used by: `backend/pipeline/temporal.py` to estimate frame-to-frame flow
for the warping-based temporal stabilization (kills flicker).

Paper: Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow", ECCV 2020.

License: BSD 3-Clause.
