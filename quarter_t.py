import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import tensorflow as tf
from autograd_minimize import minimize
from autograd_minimize.tf_wrapper import tf_function_factory_training_step


# Testing TensorFlow
tests_results = [
    tf.test.is_built_with_cuda(),
    tf.test.is_built_with_gpu_support(),
    len(tf.config.experimental.list_physical_devices('GPU')),
    tf.test.gpu_device_name()
]

print(100*'=' + ' - TEST 0')
print(f'Were TensorFlow built with CUDA (GPU) support?  {tests_results[0]}')
print(100*'=' + ' - TEST 1')
print(f'Were TensorFlow built with GPU (CUDA or ROCm) support? {tests_results[0]}')
print(100*'=' + ' - TEST 2')
print(f'Num GPUs Available: {tests_results[2]}' )
print(100*'=' + ' - TEST 3')
print(f'Default GPU Devices: {tests_results[3]}')
print(100*'=' + ' - END')


# Global variables
SMALL = 1e-10


# Matplolib configuration
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



class MLP:

    def __init__(self, n_inputs, n_outputs, n_hidden_layers, n_hidden_neurons, activation='tanh'):
        self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        self.__n_hidden_layers = n_hidden_layers
        self.__n_hidden_neurons = n_hidden_neurons
        self.__activation = activation

        self.__model = self.__create_model()

    @property
    def n_inputs(self):
        return self.__n_inputs

    @property
    def n_outputs(self):
        return self.__n_outputs

    @property
    def n_hidden_layers(self):
        return self.__n_hidden_layers

    @property
    def n_hidden_neurons(self):
        return self.__n_hidden_neurons

    @property
    def activation(self):
        return self.__activation

    @property
    def model(self):
        return self.__model

    def __create_model(self):
        inputs = tf.keras.Input(shape=self.__n_inputs)

        x = tf.keras.layers.Dense(self.__n_hidden_neurons,
                                  activation=self.__activation,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                  dtype=tf.float64)(inputs)

        for _ in range(1, self.__n_hidden_layers):
            x = tf.keras.layers.Dense(self.__n_hidden_neurons,
                                      activation=self.__activation,
                                      kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      dtype=tf.float64)(x)

        outputs = tf.keras.layers.Dense(self.__n_outputs,
                                        activation=self.__activation,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        dtype=tf.float64)(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


class PINN(MLP):

    @staticmethod
    def loss_raissi(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred))

    @staticmethod
    def loss_RMSE(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    @staticmethod
    def loss_MAE(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def __init__(self, n_inputs, n_outputs, n_hidden_layers, n_hidden_neurons,
                 nu, G,
                 learning_rate=1e-02, optimizer=tf.keras.optimizers.Adam, loss_fn=None):
        super().__init__(n_inputs, n_outputs, n_hidden_layers, n_hidden_neurons)

        self.__learning_rate = learning_rate
        self.__optimizer = optimizer(learning_rate=learning_rate)
        if loss_fn is None:
            self.__loss_fn = PINN.loss_raissi
        else:
            self.__loss_fn = loss_fn
        self.__losses = []
        self.__epoch = 0
        self.__fit_name = None

        self.__tx = None
        self.__ty = None
        self.__tz = None

        self.__u = None
        self.__v = None
        self.__w = None

        self.__u_target = None
        self.__v_target = None
        self.__w_target = None

        self.__y = None
        self.__z = None

        # Eqs. parameters
        self.__nu = nu
        self.__G = G

        # Loss functions
        self.__loss_u = None
        self.__loss_v = None
        self.__loss_w = None
        self.__loss_vel = None

        self.__loss_fn_2, self.__loss_name_2 = PINN.loss_MAE, 'MAE'
        self.__loss_u_2 = None
        self.__loss_v_2 = None
        self.__loss_w_2 = None

        self.__loss = None

        self.__u_mean = None

        self.__start_time_inner = None

        self.__delta_t_inner = None

    def __read_domain(self, X_collocated, u_target):
        self.__u_target = u_target[:, 0:1]
        self.__v_target = u_target[:, 1:2]
        self.__w_target = u_target[:, 2:3]

        self.__y = tf.Variable(X_collocated[:, 0:1])
        self.__z = tf.Variable(X_collocated[:, 1:2])

    def __solve_eqs(self):
            # with tf.GradientTape(persistent=True) as diff_tape:
            #     out = self.model(tf.concat([self.__y, self.__z], axis=1), training=True)
            #     self.__u, self.__v, self.__w = out[:, 0:1], out[:, 1:2], out[:, 2:3]
            #
            #     u_y = diff_tape.gradient(self.__u, self.__y)
            #     u_z = diff_tape.gradient(self.__u, self.__z)
            #     v_y = diff_tape.gradient(self.__v, self.__y)
            #     v_z = diff_tape.gradient(self.__v, self.__z)
            #     w_y = diff_tape.gradient(self.__w, self.__y)
            #     w_z = diff_tape.gradient(self.__w, self.__z)
            # u_yy = diff_tape.gradient(u_y, self.__y)
            # u_zz = diff_tape.gradient(u_z, self.__z)
            # v_yy = diff_tape.gradient(v_y, self.__y)
            # v_zz = diff_tape.gradient(v_z, self.__z)
            # w_yy = diff_tape.gradient(w_y, self.__y)
            # w_zz = diff_tape.gradient(w_z, self.__z)
            # del diff_tape
            #
            # self.__u_mean = tf.reduce_mean(self.__u)
            #
            # self.__tx = self.__nu * (u_yy + u_zz) - (self.__v * u_y + self.__w * u_z)
            # self.__ty = self.__nu * (v_yy + v_zz) - (self.__v * v_y + self.__w * v_z)
            # self.__tz = self.__nu * (w_yy + w_zz) - (self.__v * w_y + self.__w * w_z)

            out = self.model(tf.concat([self.__y, self.__z], axis=1), training=True)
            self.__u, self.__v, self.__w = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            self.__u_mean = tf.reduce_mean(self.__u), tf.reduce_mean(self.__v), tf.reduce_mean(self.__w)

            self.__loss_u = self.__loss_fn(self.__u_target, self.__u)
            self.__loss_v = self.__loss_fn(self.__v_target, self.__v)
            self.__loss_w = self.__loss_fn(self.__w_target, self.__w)

            self.__loss_u_2 = self.__loss_fn_2(self.__u_target, self.__u)
            self.__loss_v_2 = self.__loss_fn_2(self.__v_target, self.__v)
            self.__loss_w_2 = self.__loss_fn_2(self.__w_target, self.__w)

    def __print_res(self):
        if self.__fit_name is None:
            tf.print(f'Start of the epoch {self.__epoch}')
        else:
            tf.print(f'Start of the epoch {self.__epoch}  -  ' + self.__fit_name + ' Optimization')
        tf.print(' ')

        tf.print('u_mean:', self.__u_mean, end='\n \n')

        tf.print('loss_u: ', self.__loss_u, '--', self.__loss_name_2, ':', self.__loss_u_2)
        tf.print('loss_v: ', self.__loss_v, '--', self.__loss_name_2, ':', self.__loss_v_2)
        tf.print('loss_w: ', self.__loss_w, '--', self.__loss_name_2, ':', self.__loss_w_2)
        tf.print('loss_vel: ', self.__loss_vel)
        tf.print(30 * '-')
        tf.print('loss: ', self.__loss)

        tf.print(f"Time taken: {round(self.__delta_t_inner, 2)}s")
        tf.print(60 * '=')

    def __training_step(self):
        self.__epoch += 1
        self.__start_time_inner = time.time()

        self.__solve_eqs()

        self.__loss_vel = self.__loss_u + self.__loss_v + self.__loss_w

        self.__loss = self.__loss_vel

        self.__delta_t_inner = time.time() - self.__start_time_inner

        self.__print_res()
        return self.__loss

    def fit(self, X_collocated, u_target, epochs, tol=1e-05):
        self.__read_domain(X_collocated, u_target)

        start_time_outer = time.time()
        for _ in range(epochs):
            # self.epoch += 1
            with tf.GradientTape() as tape:
                self.__loss = self.__training_step()
            self.__losses.append(self.__loss)
            grads = tape.gradient(self.__loss, self.__model.trainable_weights)
            self.__optimizer.apply_gradients(zip(grads, self.__model.trainable_weights))

            if self.__loss < tol:
                tf.print(f"Callback EarlyStopping signal received at epoch {self.__epoch}.")
                tf.print("Terminating training.")
                break

        tf.print(f"Total time: {round(time.time() - start_time_outer, 2)}s")

    def fit_scipy(self, X_collocated, u_target, epochs_scipy, method='BFGS', tol=5e-05):
        self.__read_domain(X_collocated, u_target)

        self.__fit_name = method

        func, params = tf_function_factory_training_step(self.model, self.__training_step)

        def callback(x):
            self.__epoch += 1
            tf.print(f'Start of the epoch {self.__epoch}  -  ' + self.__fit_name + ' Optimization', end='\n\n')

        _ = minimize(func,
                     params,
                     precision='float64',
                     method=method,
                     tol=tol,
                     options={'maxiter': epochs_scipy,
                              'maxfun': 50000,
                              'maxcor': 500,
                              'maxls': 500,
                              'ftol': 1.0 * np.finfo(float).eps},
                     callback=callback
                     )

    def predict(self, X_test):
        y_test = tf.Variable(X_test[:, 0:1])
        z_test = tf.Variable(X_test[:, 1:2])

        return self.model.predict(tf.concat([y_test, z_test], axis=1))

    def results(self):
        with tf.GradientTape(persistent=True) as diff_tape:
            out = self.model(tf.concat([self.__y, self.__z], axis=1), training=False)
            self.__u, self.__v, self.__w = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            u_y = diff_tape.gradient(self.__u, self.__y)
            u_z = diff_tape.gradient(self.__u, self.__z)
            v_y = diff_tape.gradient(self.__v, self.__y)
            v_z = diff_tape.gradient(self.__v, self.__z)
            w_y = diff_tape.gradient(self.__w, self.__y)
            w_z = diff_tape.gradient(self.__w, self.__z)
        u_yy = diff_tape.gradient(u_y, self.__y)
        u_zz = diff_tape.gradient(u_z, self.__z)
        v_yy = diff_tape.gradient(v_y, self.__y)
        v_zz = diff_tape.gradient(v_z, self.__z)
        w_yy = diff_tape.gradient(w_y, self.__y)
        w_zz = diff_tape.gradient(w_z, self.__z)
        del diff_tape

        self.__tx = self.__G + self.__nu * (u_yy + u_zz) - (self.__v * u_y + self.__w * u_z)
        self.__ty = self.__nu * (v_yy + v_zz) - (self.__v * v_y + self.__w * v_z)
        self.__tz = self.__nu * (w_yy + w_zz) - (self.__v * w_y + self.__w * w_z)

        tf.print('v_y + w_z = ', v_y + w_z)

        return self.__u, self.__v, self.__w, self.__tx, self.__ty, self.__tz


def lhs_array(sample_space_limits, n_points):
    sampling = LHS(xlimits=np.array(sample_space_limits))
    return sampling(n_points)


def foam_to_python(path):
    with open(path) as file:
        out = file.readlines()
    n_points_header = 21
    n_points = int(out[n_points_header - 1])
    print()
    out = out[n_points_header + 1: n_points_header + n_points + 1]
    return np.array([[float(num) for num in row[1:-2].split(sep=' ')] for row in out])


if __name__ == '__main__':
    time_0 = time.time()

    # Load data
    reynolds = '2200'
    path_of = 'E:/casos teste/Modelos_Turbulência/square_duct/quarter'
    u_of = foam_to_python(os.path.join(path_of, reynolds, '0', 'U_target'))
    X_centroids = np.genfromtxt(os.path.join(path_of, reynolds, 'centroids.txt'), delimiter=',')[:, 1:]
    t_target = foam_to_python(os.path.join(path_of, reynolds, '0', 't_target'))

    print(f'u_of: {u_of.shape}')
    print(f'X_centroids: {X_centroids.shape}')
    print(f't: {t_target.shape}')

    out_of = np.concatenate((u_of, np.sqrt(u_of[:, 1:2] ** 2 + u_of[:, 2:3] ** 2)), axis=1)


    # =============================================General information==================================================
    name = 'quarter_t'
    Path(os.path.join('results', name)).mkdir(parents=True, exist_ok=True)
    L = 2.
    nu = np.loadtxt(
        'E:\casos teste\Modelos_Turbulência\square_duct\dns_database\dns' +
        reynolds + '/nu.dat')
    G = np.loadtxt(
        'E:\casos teste\Modelos_Turbulência\square_duct\dns_database\dns' +
        reynolds + '/G.dat')


    # Hyperparameters
    n_inputs = 2
    n_outputs = 3
    n_hidden_layers = 8
    n_hidden_neurons = 20
    optimizer = tf.keras.optimizers.Adam
    lr_schedule = 1e-03
    epochs, epochs_scipy = None, 20000
    tol = 1e-05
    loss_fn = PINN.loss_raissi  # tf.keras.metrics.mean_squared_error
    method_scipy = 'BFGS' #L-BFGS-B #BFGS


    # Collocated points
    X_collocated = X_centroids
    u_test = u_of

    plt.figure(figsize=(12, 10))
    plt.scatter(X_collocated[:, 1], X_collocated[:, 0], label='Collocated')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join('results', name, 'points.png'))


    # Model training
    tf.keras.backend.set_floatx('float64')

    model_pinn = PINN(n_inputs=n_inputs,
                      n_outputs=n_outputs,
                      n_hidden_layers=n_hidden_layers,
                      n_hidden_neurons=n_hidden_neurons,
                      nu=nu, G=G,
                      learning_rate=lr_schedule, optimizer=optimizer, loss_fn=loss_fn)

    print(model_pinn.model.summary())
    tf.keras.utils.plot_model(model_pinn.model, os.path.join('results', name, 'model.png'), show_shapes=True)

    if epochs is not None:
        model_pinn.fit(X_collocated=X_collocated, u_target=u_test, epochs=epochs, tol=tol)

    if epochs_scipy is not None:
        model_pinn.fit_scipy(X_collocated=X_collocated, u_target=u_test,
                             epochs_scipy=epochs_scipy, method=method_scipy, tol=1e-08)


    # Testing
    u_pred, v_pred, w_pred, tx_pred, ty_pred, tz_pred = model_pinn.results()

    resp = np.concatenate((X_collocated, u_pred, v_pred, w_pred, tx_pred, ty_pred, tz_pred), axis=1)
    resp = pd.DataFrame(resp, columns=['y', 'z', 'u', 'v', 'w', 'tx', 'ty', 'tz'])
    resp.to_csv(os.path.join('results', name, 'resp.csv'))

    out_pred = np.concatenate((u_pred, v_pred, w_pred, np.sqrt(v_pred ** 2 + w_pred ** 2)), axis=1)
    error = np.mean(np.abs(out_of - out_pred), axis=0)
    print('error:', error)

    y, z = X_collocated[:, 0], X_collocated[:, 1]

    for cont, name_var in enumerate(['u', 'v', 'w', 'sec']):
        print('Plotting ' + name_var)
        ims = {}
        vmin = np.concatenate((out_of[:, cont], out_pred[:, cont]), axis=0).min()
        vmax = np.concatenate((out_of[:, cont], out_pred[:, cont]), axis=0).max()
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all', figsize=(16, 8), constrained_layout=True)
        ims[0, 0] = axs[0].tricontourf(z, y, out_pred[:, cont], np.linspace(vmin, vmax, 20), cmap='jet')
        axs[0].set_xlabel('Deep Learning')
        ims[0, 1] = axs[1].tricontourf(z, y, out_of[:, cont], np.linspace(vmin, vmax, 20), cmap='jet')
        axs[1].set_xlabel('DNS')
        cbar = fig.colorbar(ims[0, 0], ax=axs[-1], format='%.0e')
        fig.suptitle(f'Square duct flow ({name_var}) -  MAE: ' + "{:.2e}".format(error[cont]))
        plt.savefig(os.path.join('results', name, name_var + '.png'))

    print('Plotting t')
    ims = {}
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all', figsize=(26, 8), constrained_layout=True)
    ims[0, 0] = axs[0].tricontourf(z, y, tx_pred[:, 0], cmap='jet')
    axs[0].set_xlabel('$t_x$')
    ims[0, 1] = axs[1].tricontourf(z, y, ty_pred[:, 0], cmap='jet')
    axs[1].set_xlabel('$t_y$')
    ims[0, 2] = axs[2].tricontourf(z, y, tz_pred[:, 0], cmap='jet')
    axs[2].set_xlabel('$t_z$')
    cbar = fig.colorbar(ims[0, 0], ax=axs[0], format='%.0e')
    cbar = fig.colorbar(ims[0, 1], ax=axs[1], format='%.0e')
    cbar = fig.colorbar(ims[0, 2], ax=axs[2], format='%.0e')
    fig.suptitle(f'Square duct flow-  Predicted Reynolds force vector')
    plt.savefig(os.path.join('results', name, 't' + '.png'))

    print(f'Total time: {round(time.time() - time_0, 2)} s')
