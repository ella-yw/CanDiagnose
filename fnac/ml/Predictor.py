def pred(inp, keras, pd):
   
    keras.backend.clear_session()
   
    classifier = keras.models.load_model("fnac/ml/FNAC_Model.hdf5")
    
    output = classifier.predict(pd.DataFrame([inp]))
    y_pred = [1 if output >= 0.5 else 0]
    
    if y_pred[0] == 1: return "MALIGN" + " (" + str(output[0][0] * 100) + "% Confidence)"
    else: return "BENIGN" + " (" + str(100 - (output[0][0] * 100)) + "% Confidence)"