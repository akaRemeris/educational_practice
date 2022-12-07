import numpy as np
import traceback

def regularization(model, reg_l=2):
    res = 0
    for parameters_type in model.parameters:
        for layer in model.parameters[parameters_type]:
            if reg_l == 1:
                res += np.abs(layer).sum()
            elif reg_l == 2:
                res += np.power(layer, 2).sum()
            else:
                raise(BaseException('Unknown regularization type'))
    return res

def regularization_derivative(model, layer_idx, parameters_type, reg_l=2):
    res = model.parameters[parameters_type][layer_idx].copy()
    if reg_l == 2:
        res *= 2
    elif reg_l ==1:
        res = np.sign(res)
    else:
        raise(BaseException('Unknown regularization type'))
    return res


def calculate_gradient(model, X, y, preds, reg_lambda=0, reg_type=2):
    """ 
    Calculate grad via cycle using saved in model activation and activation_derivative functions values.

    Parameters:
        model: Custom_FFN object

        X: ndarray - dtype('float64')
            2d Numpy array of observations*features

        y: ndarray - dtype('float64')
            2d Target numpy array of observations*classes

        preds: ndarray - dtype('float64')
            2d Model predictions numpy array of observations*classes
    
    Returns:
        Dictionary with two keys 'weights' and 'biases' 
        containing lists of gradients for each layer.

    """    
    grad_weights_list = []
    grad_biases_list = []
    dLoss_dLinear = model.loss_function_derivative(preds, y) # n*c
    for i in range(len(model.parameters['weights'])-1, -1, -1):

        grad_prod_part = dLoss_dLinear
        for j in range(model.n_hidden_layers-i):

            dLinear_dAct = model.parameters['weights'][model.n_hidden_layers-j]
            dAct_dLinear = model.activation_function_derivative_values[model.n_hidden_layers-j-1]

            grad_prod_part = np.matmul(grad_prod_part, dLinear_dAct.T)
            grad_prod_part *= dAct_dLinear

        regularization_derivative_weights = regularization_derivative(model, layer_idx=i, parameters_type='weights', reg_l=reg_type)
        regularization_derivative_biases = regularization_derivative(model, layer_idx=i, parameters_type='biases', reg_l=reg_type)
        regularization_derivative_biases = np.reshape(regularization_derivative_biases, (regularization_derivative_biases.shape[1],))

        if i > 0:
            grad_w = np.matmul(model.activation_functions_values[i-1].T, grad_prod_part)
        else:
            grad_w = np.matmul(X.T, grad_prod_part)
        grad_b = grad_prod_part.sum(axis=0)

        grad_w += reg_lambda * regularization_derivative_weights
        grad_b += reg_lambda * regularization_derivative_biases

        grad_weights_list.append(grad_w.copy())            
        grad_biases_list.append(grad_b.copy())

    grad_dict = {'weights': grad_weights_list[::-1], 'biases': grad_biases_list[::-1]}
    return grad_dict

def grad_step(model, grad_dict, lr):
    for parameter_type in grad_dict:
        for i, grad in enumerate(grad_dict[parameter_type]):
            model.parameters[parameter_type][i] -= lr * grad

def batch_update(model, X, y, lr, reg_lambda=0, reg_type=2):
    preds = model.forward(X)
    start_loss = model.loss_function(preds, y) + reg_lambda * regularization(model, reg_l=reg_type)

    grads = calculate_gradient(model, X, y, preds, reg_lambda, reg_type)        
    grad_step(model, grads, lr)

    preds = model.forward(X)
    end_loss = model.loss_function(preds, y) + reg_lambda * regularization(model, reg_l=reg_type)

    return (start_loss, end_loss)

def SGD(model, X, y, batch_size, 
        max_epochs=200, 
        eps=1e-6, 
        lr=0.1, 
        lr_decay=0.5,
        max_convergences=100,
        reg_lambda=0,
        reg_type=2
        ):
    
    loss_log = [0]
    
    X_ids = np.arange(X.shape[0])
    np.random.shuffle(X_ids)
    convergence_counter = 1

    for epoch in range(max_epochs):
        try:
            have_converged = True
            batch_loss = [0]
            
            for i in range(X.shape[0] - batch_size + 1):
                batch_ids = X_ids[i:i+batch_size]
                X_batch = X[batch_ids, :]
                y_batch = y[batch_ids]

                #change to optimizer step
                loss = batch_update(model, X_batch, y_batch, lr, reg_lambda, reg_type)

                if (loss[0] - loss[1]) > eps:
                    have_converged = False
                batch_loss.append(loss[1])

            loss_log.append(batch_loss[-1])

            if have_converged:
                convergence_counter += 1
            else:
                convergence_counter = 1            

            if convergence_counter%max_convergences == 0:
                    lr *= lr_decay
                    convergence_counter = 1
                    print('decayed!')
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception:
            print("Failed training!")
            print(Exception.args)
            break
    return loss_log