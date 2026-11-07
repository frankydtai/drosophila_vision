from pathlib import Path
import json
import select
import numpy
import torch
O=Path(__file__).resolve().parent
assert not Path(select.__file__).is_relative_to(O),select.__file__
result=dict(select_module=select.__file__,numpy_version=numpy.__version__,torch_version=torch.__version__,imports_ok=True)
(O/'import_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
