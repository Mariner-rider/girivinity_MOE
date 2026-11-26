"""Bootstrap mocks for running unit tests in lightweight environments without external heavy ML/DB packages."""

import pathlib
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, "app_vendor")

# Torch mock (if real torch isn't fully loadable)
if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "utils"):
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__path__ = []  # Mark as package so submodules can be imported
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn_utils = types.ModuleType("torch.nn.utils")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.__path__ = []
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_testing = types.ModuleType("torch.testing")
    torch_cuda = types.ModuleType("torch.cuda")
    torch_cuda_amp = types.ModuleType("torch.cuda.amp")
    torch_optim = types.ModuleType("torch.optim")
    torch_optim.__path__ = []
    torch_optim_lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
    torch_distributed = types.ModuleType("torch.distributed")
    torch_distributed.is_available = lambda: False
    torch_distributed.is_initialized = lambda: False

    class DummyModule:
        def __init__(self, *args, **kwargs):
            self.weight = DummyTensor([1.0])

        def parameters(self):
            return []

        def gradient_checkpointing_enable(self):
            pass

        def to(self, *args, **kwargs):
            return self

    torch_nn.init = types.ModuleType("torch.nn.init")
    torch_nn.init.constant_ = lambda tensor, val: None
    torch_nn.init.kaiming_uniform_ = lambda tensor, **kwargs: None
    torch_nn.Module = DummyModule
    torch_nn.Linear = DummyModule
    torch_nn.Embedding = DummyModule
    torch_nn.Parameter = lambda data: data
    torch_nn.ModuleList = lambda *args, **kwargs: []
    torch_nn.Dropout = DummyModule
    torch_nn.functional = torch_nn_functional
    torch_nn_utils.clip_grad_norm_ = lambda parameters, max_norm: None
    torch.arange = lambda start, end, step=1, **kwargs: DummyTensor(list(range(start, end, step)))
    torch.outer = lambda a, b: DummyTensor([[1.0]])
    torch.polar = lambda a, b: DummyTensor([1.0])
    torch.cos = lambda a: DummyTensor([1.0])
    torch.sin = lambda a: DummyTensor([1.0])
    torch_optim.AdamW = lambda *args, **kwargs: None
    torch_optim.lr_scheduler = torch_optim_lr_scheduler
    torch_optim_lr_scheduler.LambdaLR = lambda *args, **kwargs: None

    class DummyDataset:
        def __class_getitem__(cls, item):
            return cls

    class DummyDataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size

    class DummyTensor:
        def __init__(self, data, dtype=None):
            if isinstance(data, DummyTensor):
                self.data = list(data.data)
            elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                self.data = [d.data if isinstance(d, DummyTensor) else d for d in data]
            else:
                self.data = [data]
            self.dtype = dtype
            self.grad = None

        @property
        def shape(self):
            if self.data and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def size(self, dim=None):
            s = self.shape
            return s[dim] if dim is not None else s

        def numel(self):
            return len(self.data)

        def item(self):
            return self.data[0] if self.data else 0

        def to(self, *args, **kwargs):
            return self

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                row, col = idx
                val = self.data[row]
                if isinstance(col, slice):
                    return DummyTensor(val[col])
                if isinstance(val, list):
                    return DummyTensor([val[col]])
                return DummyTensor([val])
            if isinstance(idx, slice):
                return DummyTensor(self.data[idx])
            return DummyTensor([self.data[idx]])

        def __setitem__(self, idx, value):
            if isinstance(idx, tuple):
                row, col = idx
                val = value.data if isinstance(value, DummyTensor) else value
                if isinstance(col, slice):
                    start = col.start or 0
                    stop = col.stop if col.stop is not None else len(self.data[row])
                    if isinstance(val, list):
                        self.data[row][start : start + len(val)] = val
                    else:
                        self.data[row][start:stop] = [val] * (stop - start)
                else:
                    self.data[row][col] = val[0] if isinstance(val, list) else val
            elif isinstance(idx, slice):
                val = value.data if isinstance(value, DummyTensor) else value
                start = idx.start or 0
                stop = idx.stop if idx.stop is not None else len(self.data)
                self.data[start:stop] = val
            else:
                val = value.data if isinstance(value, DummyTensor) else value
                self.data[idx] = val[0] if isinstance(val, list) else val

        def __len__(self):
            return len(self.data)

        def sum(self):
            return DummyTensor([sum(self.data)])

        def abs(self):
            return DummyTensor([abs(x) for x in self.data])

        def mul_(self, factor):
            self.data = [x * factor for x in self.data]
            return self

        def float(self):
            return self

        def backward(self):
            pass

    torch_nn_init = types.ModuleType("torch.nn.init")
    torch_nn_init.constant_ = lambda tensor, val: None
    torch_nn.init = torch_nn_init
    torch_nn.Module = DummyModule
    torch_nn.Linear = DummyModule
    torch_nn.functional = torch_nn_functional
    torch_utils.data = torch_utils_data
    torch_utils_data.Dataset = DummyDataset
    torch_utils_data.DataLoader = DummyDataLoader
    torch_utils_data.random_split = lambda dataset, lengths, generator=None: (dataset, dataset)
    torch.Tensor = DummyTensor
    torch.tensor = lambda d, dtype=None: DummyTensor(d, dtype)
    def _make_zeros(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if len(shape) == 2:
            return DummyTensor([[0] * shape[1] for _ in range(shape[0])])
        return DummyTensor([0] * (shape[0] if shape else 0))

    torch.zeros = _make_zeros
    torch.full = lambda shape, val, **kwargs: DummyTensor([[val] * shape[1] for _ in range(shape[0])]) if len(shape) == 2 else DummyTensor([val] * shape[0])
    torch.empty = lambda *shape, **kwargs: DummyTensor([])
    torch.long = "long"
    torch.no_grad = lambda: (lambda f: f)

    def assert_close(a, b):
        assert a.data == b.data, f"{a.data} != {b.data}"

    class DummyDevice:
        def __init__(self, type_="cpu"):
            self.type = type_

    torch.device = lambda d="cpu": DummyDevice(d)
    torch.save = lambda obj, f, *args, **kwargs: pathlib.Path(f).write_text("dummy_tensor_bytes")
    torch.load = lambda f, *args, **kwargs: {}
    torch_cuda.is_available = lambda: False
    torch_cuda.get_device_name = lambda i=0: "Mock GPU"
    torch_testing.assert_close = lambda a, b, **kwargs: None

    torch.nn = torch_nn
    torch.utils = torch_utils
    torch.utils.data = torch_utils_data
    torch.testing = torch_testing
    torch.cuda = torch_cuda
    torch.cuda.amp = torch_cuda_amp
    torch.optim = torch_optim
    torch.distributed = torch_distributed
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.nn.utils"] = torch_nn_utils
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data
    sys.modules["torch.testing"] = torch_testing
    sys.modules["torch.cuda"] = torch_cuda
    sys.modules["torch.cuda.amp"] = torch_cuda_amp
    sys.modules["torch.optim"] = torch_optim
    sys.modules["torch.optim.lr_scheduler"] = torch_optim_lr_scheduler
    sys.modules["torch.distributed"] = torch_distributed

# DB (psycopg2) mock
if "psycopg2" not in sys.modules:
    p = types.ModuleType("psycopg2")
    p.extras = types.ModuleType("psycopg2.extras")
    p.pool = types.ModuleType("psycopg2.pool")
    sys.modules["psycopg2"] = p
    sys.modules["psycopg2.extras"] = p.extras
    sys.modules["psycopg2.pool"] = p.pool

# RAG / ML deps mocks
for mod in ["chromadb", "sentence_transformers", "transformers"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
