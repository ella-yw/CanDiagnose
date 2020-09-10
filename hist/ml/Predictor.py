def pred(link, keras, pd, np, os, sp, gl, sh, visualize):
    
    keras.backend.clear_session()
    
    img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe=pd.DataFrame([[link, ""]], 
                                                                                                     columns=['filename', 'class']), 
                                                                       directory='imgs_to_predict/', target_size=(50, 50),
                                                                       color_mode='rgb')[0][0][0].reshape(1, 50, 50, 3)
    
    model = keras.models.load_model("hist/ml/HIST_Model.hdf5")
    
    prediction =  "*NO* Invasive Ductal Carcinoma Present" if np.argmax(model.predict(img)[0]) == 0 else "Invasive Ductal Carcinoma Present"
    
    sh.rmtree('imgs_to_predict')
    
    if visualize == True:
        
        path = 'hist/static/hist/visualizations/'
        for sub_path in os.listdir(path):
            if os.path.exists(os.path.join(path, sub_path)): sh.rmtree(os.path.join(path, sub_path))
    
        filters = [[] for i in range(np.shape(model.layers)[0])]
        
        for layer_idx in range(np.shape(model.layers)[0]):
            
            feature_maps = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],
                                              [model.layers[layer_idx].output])([img])[0][0]    
            
            directory = "hist/static/hist/visualizations/layer-" + str(layer_idx+1)
            os.makedirs(directory)
            
            for filter_num in range(feature_maps.shape[-1]):
                
                try: sp.misc.toimage(feature_maps[:,:,filter_num], mode = 'P').save(directory + "/{}".format(filter_num) + '.png', quality=1, optimize=True)
                except IndexError: exit
                
                if len(gl.glob(directory + "/*")) != 0: filters[layer_idx].append(filter_num)
                
        filternums = []
        for i in filters:
            if i != []: filternums.append(i)
                    
        return prediction, filternums
    
    else: return prediction