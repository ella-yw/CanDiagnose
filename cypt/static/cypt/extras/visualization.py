import warnings, sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import keras
import numpy as np
from keras.utils import plot_model
from contextlib import redirect_stdout
np.set_printoptions(threshold=np.nan)

model = keras.models.load_model("cell_nuclei_characteristics_model.hdf5")
    
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_structure.png', rankdir='TB')
#from IPython.display import SVG 
#from keras.utils.vis_utils import model_to_dot 
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

for layer in range(len(model.layers)):
    try:
        with open('params/biases/'+str(layer)+'-biases.txt', 'w') as f:
            with redirect_stdout(f):
                print(model.layers[layer].get_weights()[1])
    except IndexError:
         pass
for layer in range(len(model.layers)):
    try:
        with open('params/weights/'+str(layer)+'-weights.txt', 'w') as f:
            with redirect_stdout(f):
                print(model.layers[layer].get_weights()[0])
    except IndexError:
         pass
     
