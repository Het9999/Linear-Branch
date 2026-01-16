import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yaml


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_data(config):
    data_config = config['data']
    rng = np.random.RandomState(data_config['random_state'])
    X = 2 * rng.rand(data_config['n_samples'], 1)
    y = 4 + 3 * X.ravel() + rng.randn(data_config['n_samples']) * data_config['noise_std']
    return X, y


def fit_and_evaluate(X, y, config):
    eval_config = config['evaluation']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=eval_config['test_size'], random_state=1
    )
    model_config = config['model']
    model = LinearRegression(fit_intercept=model_config['params']['fit_intercept'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_train, X_test, y_train, y_test, y_pred


def main():
    config = load_config()
    X, y = generate_data(config)
    model, mse, r2, X_train, X_test, y_train, y_test, y_pred = fit_and_evaluate(X, y, config)

    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    try:
        import matplotlib.pyplot as plt
        plt.scatter(X, y, label='data')
        xs = np.linspace(0, 2, 100).reshape(-1, 1)
        plt.plot(xs, model.predict(xs), color='red', label='model')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    except Exception:
        # matplotlib not available or running headless; that's fine
        pass


if __name__ == '__main__':
    main()
