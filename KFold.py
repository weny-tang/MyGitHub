from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from random import shuffle

train_file = 'D:\\workspace\\ScarpAndCompate\\dustral\\data\\zhengqi_train.txt'
test_file = 'D:\\workspace\\ScarpAndCompate\\dustral\\data\\zhengqi_test.txt'

#载入数据
def load_train_data(file_path):
    train_data = np.loadtxt(open(file_path,'rb'), delimiter = "\t", skiprows = 1)
    return train_data

def load_test_data(file_path):
    test_data = np.loadtxt(open(file_path,'rb'), delimiter = "\t", skiprows = 1)
    return test_data
 
#构建模型
def build_model(x,y):
    kfold = KFold(y.shape[0],5)#K折交叉检验划分训练集和测试集，5份数据集（每份包括训练和测试）
    model = Ridge(normalize=True)#标准化数据并采用岭回归模型
    alpha_range = np.linspace(0.0015,0.0017,30)#生成alpha测试集
    grid_param = {"alpha":alpha_range}
    #GridSearchCV帮助我们采用一个范围内参数对模型进行训练
    #cv定义了感兴趣的交叉验证类型
    grid = GridSearchCV(estimator=model,param_grid=grid_param,cv=kfold,\
                        scoring='mean_squared_error')
    grid.fit(x,y)
    display_param_results(grid.grid_scores_)#展示均方误差平均值
    print(grid.best_params_)#打印最好的参数和评估量
    #追踪均方残差的计量用于绘制图形
    return grid.best_estimator_
     
    
#查看回归系数和截距
def view_model(model):
    #print "\n estimated alpha = %0.3f" % model.alpha_#打印模型采用的alpha值
    print("\n model coeffiecients")
    print("======================\n")
    for i,coef in enumerate(model.coef_):
        print("\t coefficent %d %0.3f" % (i+1,coef))
    print("\n\t intercept %0.3f" % (model.intercept_))
 
#模型评估
def model_worth(true_y,predicted_y):
    print("\t Mean squared error = %0.2f" % (mean_squared_error(true_y,predicted_y)))
    return mean_squared_error(true_y,predicted_y)
 
#展示参数结果
def display_param_results(param_results):
    fold = 1
    for param_result in param_results:
        print("fold %d mean squared error %0.2f" % (fold,abs(param_result[1]\
                                                             )),param_result[0])
        fold+=1
        
if __name__ == "__main__":
    train_data = load_train_data(train_file)
    test_data = load_test_data(test_file)
    shuffle(train_data)
    train = train_data[:int(len(train_data)*0.8),:]
    test = train_data[int(len(train_data)*0.8):,:]
    x_train = train[:-1]
    y_train = train[-1]
    x_test = test[:-1]
    y_test = test[-1]
    #将数据集划分为训练集和测试集
    #x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=9)
    #准备一些多项式特征
    poly_features = PolynomialFeatures(interaction_only=True)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)
    choosen_model = build_model(x_train_poly, y_train)
    predicted_y = choosen_model.predict(x_train_poly)
    model_worth(y_train, predicted_y)
    view_model(choosen_model)
    predicted_y = choosen_model.predict(x_test_poly)
    model_worth(y_test, predicted_y)    