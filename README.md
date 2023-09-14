# PfO3

PfO3 is a python module written in rust to process numpy ndarrays (e.g. the "call_genotype" variable of the `malariagen-data` callset) quickly.

## Requirements

To build and install PfO3 using the source code in this repository, you must first install [rust](https://www.rust-lang.org/tools/install) and [maturin](https://www.maturin.rs/). 

Then you can install the python package localling using the following commands:

```sh
❯ git clone https://github.com/bjeight/PfO3.git
❯ cd PfO3
❯ maturin build --release
📦 Built wheel for CPython 3.11 to /path/to/wheels/PfO3-0.1.0-cp311-cp311-macosx_11_0_arm64.whl
❯ python3 -m pip install /path/to/wheels/PfO3-0.1.0-cp311-cp311-macosx_11_0_arm64.whl
```

Then you can simply import PfO3 in your python scripts

```py
import PfO3

```