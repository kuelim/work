import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

def plot_results(x, y, title):
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='.')
    plt.title(title)
    plt.show()

def CircleClassify():
    # 데이터 생성
    n_samples = 400
    noise = 0.02
    factor = 0.5
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    # 학습 데이터 분포 시각화
    plot_results(x_train, y_train, "Train data distribution")

    # SLP 모델 정의
    slp_model = Sequential([
        Input(shape=(2,)),
        Dense(1, activation='sigmoid')
    ])

    # SLP 모델 컴파일
    slp_model.compile(optimizer=SGD(learning_rate=0.1), loss='mse', metrics=['accuracy'])

    # SLP 모델 학습
    slp_model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=0)

    # SLP 모델 예측
    slp_predictions = (slp_model.predict(x_test) > 0.5).astype(int)

    # SLP 결과 시각화
    plot_results(x_test, slp_predictions, "SLP Classification Results")

    # MLP 모델 정의
    mlp_model = Sequential([
        Input(shape=(2,)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # MLP 모델 컴파일
    mlp_model.compile(optimizer=SGD(learning_rate=0.1), loss='mse', metrics=['accuracy'])

    # MLP 모델 학습
    mlp_model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=0)

    # MLP 모델 예측
    mlp_predictions = (mlp_model.predict(x_test) > 0.5).astype(int)

    # MLP 결과 시각화
    plot_results(x_test, mlp_predictions, "MLP Classification Results")

if __name__ == '__main__':
    CircleClassify()
