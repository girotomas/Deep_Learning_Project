def compute_nb_errors_pred(model, inputt, target, mini_batch_size):
    nb_errors = 0
    L = []
    for b in range(0, inputt.size(0), mini_batch_size):
        _, output  = model(inputt.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        L.append(predicted_classes)
        for k in range(mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    #print(nb_errors)
    return nb_errors, L