import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, model=None, left_model=None, right_model=None, final=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.model = model
        self.left_model = left_model 
        self.right_model = right_model
        self.final = final

class DecisionTreeME():
    def __init__(self, min_samples=2, max_depth=2,n_thresholds=10):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_thresholds = n_thresholds
        self.scores = {}

    def split_data(self, dataset, feature, threshold):
        left_dataset = dataset[dataset[:, feature] <= threshold]
        right_dataset = dataset[dataset[:, feature] > threshold]
        return left_dataset, right_dataset

    def information_gain(self, parent, parent_model, left, left_model, right, right_model, ut_hp=0.001):
        rmse_parent = np.sqrt(mean_squared_error(parent[:, -1], parent_model.predict(parent[:, :-1])))
        rmse_left_left = np.sqrt(mean_squared_error(left[:, -1], left_model.predict(left[:, :-1])))
        rmse_left_right = np.sqrt(mean_squared_error(left[:, -1], right_model.predict(left[:, :-1])))
        rmse_right_left = np.sqrt(mean_squared_error(right[:, -1], left_model.predict(right[:, :-1])))
        rmse_right_right = np.sqrt(mean_squared_error(right[:, -1], right_model.predict(right[:, :-1])))

        lambda_left = len(left) / len(parent)
        lambda_right = len(right) / len(parent)

        mejora = -(rmse_parent - (lambda_left * rmse_left_left + lambda_right * rmse_right_right))
        especializacion = lambda_left * (rmse_left_right - rmse_left_left) + lambda_right * (rmse_right_left - rmse_right_right)

        gain = mejora - ut_hp * especializacion
        return gain

    def evaluate_split(self, train_dataset, val_dataset, feature_index, threshold, model_parent, alpha, ut_hp):
        left_train, right_train = self.split_data(train_dataset, feature_index, threshold)
        left_val, right_val = self.split_data(val_dataset, feature_index, threshold)

        if len(left_train) >= self.min_samples and len(right_train) >= self.min_samples \
           and len(left_val) >= self.min_samples and len(right_val) >= self.min_samples:
            
            model_left = Lasso(alpha=alpha, fit_intercept=True).fit(left_train[:, :-1], left_train[:, -1])
            model_right = Lasso(alpha=alpha, fit_intercept=True).fit(right_train[:, :-1], right_train[:, -1])

            gain = self.information_gain(train_dataset, model_parent, left_train, model_left, right_train, model_right, ut_hp)

            return {
                "feature": feature_index,
                "threshold": threshold,
                "left_train_dataset": left_train,
                "right_train_dataset": right_train,
                "left_val_dataset": left_val,
                "right_val_dataset": right_val,
                "left_model": model_left,
                "right_model": model_right,
                "model": model_parent,
                "gain": gain
            }
        return None

    def best_split(self, train_dataset, val_dataset, num_samples, num_features,
                model_parent=None, alpha=0.1, ut_hp=0.001):

      if model_parent is None:
          model_parent = Lasso(alpha=alpha, fit_intercept=True).fit(
              train_dataset[:, :-1], train_dataset[:, -1]
          )

      def best_split_for_feature(feature_index):
          """
          Busca el mejor threshold SOLO para una feature.
          """
          best_local = {'gain': -1}

          thresholds = np.unique(train_dataset[:, feature_index])
          if len(thresholds) > self.n_thresholds:
              thresholds = np.random.choice(
                  thresholds, size=self.n_thresholds, replace=False
              )

          for threshold in thresholds:
              res = self.evaluate_split(
                  train_dataset, val_dataset,
                  feature_index, threshold,
                  model_parent, alpha, ut_hp
              )
              if res is not None and res["gain"] > best_local["gain"]:
                  best_local = res

          return best_local

      # ⚠️ NO usar n_jobs=-1 en cluster
      n_jobs = min(4, num_features)

      results = Parallel(n_jobs=n_jobs)(
          delayed(best_split_for_feature)(f_idx)
          for f_idx in range(num_features)
      )

      best = {'gain': -1}
      for res in results:
          if res is not None and res["gain"] > best["gain"]:
              best = res

      return best


    def build_tree(self, train_dataset, val_dataset, current_depth=0, model_parent=None, alpha=0.1, ut_hp=0.001):
        X, y = train_dataset[:, :-1], train_dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            print(f'🔎 Profundidad {current_depth} - Buscando mejor split...')
            best_split = self.best_split(train_dataset, val_dataset, n_samples, n_features, model_parent, alpha, ut_hp)
            
            if best_split["gain"] == -1:
                print(f'⚠️ No se encontró un split válido en profundidad {current_depth}. Creando nodo hoja.')
                return Node(model=model_parent, final=True)

            print(f'✅ Mejor split en {best_split["feature"]} con threshold {best_split["threshold"]} y gain {best_split["gain"]:.4f}')
            left_node = self.build_tree(best_split["left_train_dataset"], best_split["left_val_dataset"], current_depth + 1, best_split["left_model"], alpha, ut_hp)
            right_node = self.build_tree(best_split["right_train_dataset"], best_split["right_val_dataset"], current_depth + 1, best_split["right_model"], alpha, ut_hp)

            return Node(best_split["feature"], best_split["threshold"],
                        left_node, right_node, best_split["gain"],
                        best_split["model"], best_split["left_model"], best_split["right_model"])

        print(f'🌿 Profundidad {current_depth}: Creando nodo hoja con {n_samples} muestras.')
        return Node(model=model_parent, final=True)

    def fit(self, X_train, y_train, X_val, y_val, alpha=0.1, ut_hp=0.001):
        train_dataset = np.concatenate((X_train, y_train), axis=1)
        val_dataset = np.concatenate((X_val, y_val), axis=1)
        self.root = self.build_tree(train_dataset, val_dataset, alpha=alpha, ut_hp=ut_hp)


    def predict(self, X):

        """
        Predicts for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predictions.
        """
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector.
        """
        # Si el nodo es hoja, hacer la predicción
        if node.final:
            return node.model.predict(x.reshape(1, -1))[0]
        
        # Verificar que los hijos existen antes de continuar
        if node.left is None or node.right is None:
            raise ValueError("El árbol no se construyó correctamente, hay nodos faltantes.")

        # Si el nodo no es hoja, continuar la búsqueda
        feature = x[node.feature]
        if feature <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)
        
    def retrain_leaf_nodes(self, train_dataset, node=None, param_grid=None, cv=5):

        if param_grid is None:
            param_grid = {'alpha': np.logspace(-4, 0, 10)}

        if node is None:
            node = self.root

        if node.final:
            if len(train_dataset) < cv:
                print(f"⚠️ Nodo hoja con {len(train_dataset)} muestras: insuficiente para CV ({cv}-fold). Se omite reentrenamiento.")
                return

            X_leaf = train_dataset[:, :-1]
            y_leaf = train_dataset[:, -1]

            search = GridSearchCV(Lasso(fit_intercept=True), param_grid, cv=cv, scoring='neg_mean_squared_error')
            search.fit(X_leaf, y_leaf)

            best_model = search.best_estimator_
            node.model = best_model
            best_score = -search.best_score_

            print(f"📌 Nodo hoja reentrenado con alpha={best_model.alpha:.4f}, MSE={-search.best_score_:.4f}, con {len(X_leaf):.0f}muestras")
            self.scores[len(X_leaf)] = best_score
            

            return
        

        # Si no es hoja, splitear los datos y continuar recursivamente
        left_data, right_data = self.split_data(train_dataset, node.feature, node.threshold)

        if node.left is not None:
            self.retrain_leaf_nodes(left_data, node.left, param_grid, cv)
        if node.right is not None:
            self.retrain_leaf_nodes(right_data, node.right, param_grid, cv)

