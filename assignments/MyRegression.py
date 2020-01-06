
class MyRegression():
	def __init__(self,lam=0,a=None,b=None):
		self.a=a
		self.b=b
		self.lam=lam
		
	def fit(self,X,y):
		X, y = check_X_y(X, y, y_numeric=True)
        
		if self.lam != 0:
			pass
		else:
			pass

		one=np.ones(X.shape[0]).reshape(-1,1)
		X_=np.concatenate((one,X),axis=1)
		Lam=self.lam*np.eye(X_.shape[1])
		A=Lam+np.dot(X_.T,X_)
		X_daggar=np.dot(np.linalg.inv(A),X_.T)
		w=np.dot(X_daggar,y)
        
		self.a_=w[1:]
		self.b_=w[0]
        
		return self
    
	def predict(self,X):
		X=check_array(X)
		y=np.dot(X,self.a_)+self.b_
        
		check_is_fitted(self, "a_", "b_") # 学習済みかチェックする(推奨)
		X = check_array(X)
		return y


if __name__=="__main__":
	clf = MyRegression()
