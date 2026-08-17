# Shape-Aware Graph Distance
## Installation
### Clone the repository

```
git clone https://github.com/msarrias/SAGD.git
cd SAGD
```

### Install package

```
pip install -e .
```

## Usage
```python
import numpy as np
from SAGD import SAGD

# Load your adjacency/weight matrices
W1 = np.random.rand(100, 100) 
W2 = np.random.rand(100, 100)

# Calculate distance
distance = SAGD(W1, W2, laplacian_type="unnormalized", norm_type="norm_wrt_avg_ctd")
print(f"SAGD: {distance}")
```
