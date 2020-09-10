def pred(link, keras, pd, np, os, sp, gl, sh, jb, visualize, model):
    
    keras.backend.clear_session()
    
    img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe=pd.DataFrame([[link, ""]], 
                                                                                                     columns=['filename', 'class']), 
                                                                       directory='imgs_to_predict/', target_size=(64, 64),
                                                                       color_mode='grayscale')[0][0][0].reshape(1, 64, 64 , 1)   
    
    pbd_model = keras.models.load_model("mamm/ml/PBD_Model.hdf5")
    birads5_model = keras.models.load_model("mamm/ml/BIRADS5_Model.hdf5")
    
    y_pred_pbd = pbd_model.predict(img); y_pred_birads5 = birads5_model.predict(img);
    pbd = y_pred_pbd[0][0]; birads5 = np.argmax(y_pred_birads5[0]) + 1;
    
    if pbd >= 0 and pbd <= 16.66: categorized_pbd = 1
    elif pbd > 16.66 and pbd <= 33.33: categorized_pbd = 2
    elif pbd > 33.33 and pbd <= 50: categorized_pbd = 3
    elif pbd > 50 and pbd <= 66.66: categorized_pbd = 4
    elif pbd > 66.66 and pbd <= 83.33: categorized_pbd = 5
    elif pbd > 83.33 and pbd <= 100: categorized_pbd = 6
    
    svc = jb.load('mamm/ml/MAMM_Model.joblib')
    prediction = "Malignancy Detected" if svc.predict([[categorized_pbd, birads5]])[0] == 0 else "*NO* Malignancy Detected"
    
    sh.rmtree('imgs_to_predict')
    
    if visualize == True:
        
        path = 'mamm/static/mamm/visualizations/'
        for sub_path in os.listdir(path):
            if os.path.exists(os.path.join(path, sub_path)): sh.rmtree(os.path.join(path, sub_path))
    
        filters = [[] for i in range(np.shape(locals()[model + '_model'].layers)[0])]
        
        for layer_idx in range(np.shape(locals()[model + '_model'].layers)[0]):
            
            feature_maps = keras.backend.function([locals()[model + '_model'].layers[0].input, keras.backend.learning_phase()],
                                                  [locals()[model + '_model'].layers[layer_idx].output])([img])[0][0]    
            
            directory = "mamm/static/mamm/visualizations/layer-" + str(layer_idx+1)
            os.makedirs(directory)
            
            for filter_num in range(feature_maps.shape[-1]):
                
                try: sp.misc.toimage(feature_maps[:,:,filter_num]).save(directory + "/{}".format(filter_num) + '.jpg', quality=1, optimize=True)
                except IndexError: exit
                
                if len(gl.glob(directory + "/*")) != 0: filters[layer_idx].append(filter_num)
                
        filternums = []
        for i in filters:
            if i != []: filternums.append(i)
        
        return (str(prediction), str(pbd), str(categorized_pbd), str(birads5), filternums)
    
    else: return (str(prediction), str(pbd), str(categorized_pbd), str(birads5))