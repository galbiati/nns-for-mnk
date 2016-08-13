# Neural network models for (4, 9, 4) game data

Main dependencies: Python 3.x, pandas, numpy, Theano, lasagne

I use Jupyter notebook through Anaconda. Training and Analysis.ipynb is where you can find usage examples.

The code used here was developed by imitating the Lasagne tutorial and expanding as needed.

archs.py contains functions that define architectures for different networks.

load_data.py contains functions for loading data.

network.py implements a container class for networks.

train.py contains functions that implement alternative training schemes. (This and network should be cleaned up so default training scheme is moved out of class and into independent function).

util.py contains miscellaneous functions. (These can be moved into separate scripts as variety requires, eg various export functions).

Results.ipynb is for analyzing network output.

Network Examination.ipynb is old and retiring soon.