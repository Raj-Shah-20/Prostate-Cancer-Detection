import pickle
import numpy as np
# X_test =  [[-1.30490406,0.70736518,1.18963536,1.22990484,0.12423467,1.04470792,0.73602258,-0.37138504]]
s = np.load('std.npy')
m = np.load('mean.npy')
int_features = [19,27,72,371,0.123,0.122,0.19,0.069]
final_features = (np.array([int_features]-m))/s
# final_features = scale1.transform(final_features)
loaded_model = pickle.load(open('finalized_model.sav','rb'))
result = loaded_model.predict(final_features)
print(result)