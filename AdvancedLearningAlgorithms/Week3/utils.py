# For data visualization
import matplotlib.pyplot as plt

def plot_dataset(x, y, title="input vs. target"):
    """Plot the dataset showing input vs target relationship."""
    plt.scatter(x, y, c='blue', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target"):
    """Plot the dataset showing training, CV, and test sets."""
    plt.scatter(x_train, y_train, c='blue', alpha=0.5, label='Training')
    plt.scatter(x_cv, y_cv, c='green', alpha=0.5, label='Cross Validation')
    plt.scatter(x_test, y_test, c='red', alpha=0.5, label='Test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs"):
    """Plot training and CV MSEs for different polynomial degrees."""
    plt.plot(degrees, train_mses, 'bo-', label='Training MSE')
    plt.plot(degrees, cv_mses, 'ro-', label='CV MSE')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.xticks(degrees)
    plt.grid(True)
    plt.show()

def plot_bc_dataset(x, y, title="x1 vs. x2"):
    """Plot the binary classification dataset."""
    # Separate the two classes
    class_0 = x[y.flatten() == 0]
    class_1 = x[y.flatten() == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Class 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.show()

def build_models():
    """Build neural network models with different architectures for regression."""
    import tensorflow as tf
    
    models = []
    
    # Model 1: Small network
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='model_1')
    models.append(model1)
    
    # Model 2: Medium network
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='model_2')
    models.append(model2)
    
    # Model 3: Large network
    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='model_3')
    models.append(model3)
    
    return models

def build_models_classification():
    """Build neural network models with different architectures for classification."""
    import tensorflow as tf
    
    models = []
    
    # Model 1: Small network
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ], name='model_1')
    models.append(model1)
    
    # Model 2: Medium network
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ], name='model_2')
    models.append(model2)
    
    # Model 3: Large network
    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ], name='model_3')
    models.append(model3)
    
    return models

