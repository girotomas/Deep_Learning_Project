def compute_nb_errors(model, inputt, target, mini_batch_size):
    nb_errors = 0
    L = []
    for b in range(0, inputt.size(0), mini_batch_size):
        output = model(inputt.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        L.append(predicted_classes)
        for k in range(mini_batch_size):
            if target.data[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    #print(nb_errors)
    return nb_errors, L

def compute_nb_errors_shared(model, inputt, target_0, target_1, mini_batch_size):
    nb_errors = 0
    L = []
    for b in range(0, inputt.size(0), mini_batch_size):
        output = model(inputt.narrow(0, b, mini_batch_size))
        out_0 = output[:,0:10]
        out_1 = output[:,10:20]
        _, predicted_classes_0 = out_0.data.max(1)
        _, predicted_classes_1 = out_1.data.max(1)

        L.append([predicted_classes_0, predicted_classes_1])
        for k in range(mini_batch_size):
            if target_0.data[b + k, predicted_classes_0[k]] <= 0:
                nb_errors = nb_errors + 1

            if target_1.data[b + k, predicted_classes_1[k]] <= 0:
                nb_errors = nb_errors + 1

    #print(nb_errors)
    return nb_errors, L