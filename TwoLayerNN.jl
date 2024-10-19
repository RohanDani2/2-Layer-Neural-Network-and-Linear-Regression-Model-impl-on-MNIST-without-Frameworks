
using HDF5, Distributions, LinearAlgebra, Random

# linear activation function for the output layer
hlin(a) = a
# and it's derivative
hplin(a) = 1

#sigmoid activation function to map the value between 0 and 1
hsig(a) = 1 ./ (1 .+ exp.(-a))

# used in backprop for calculating the gradients
hpsig(a) = hsig(a) .* (1 .- hsig.(a))

#output layer of the multiclassification 
softmax(x) = exp.(x) ./ sum(exp.(x))

#compute the gradient during backprop
softmaxderiv(x) = softmax(x) .* (Diagonal(ones(10,10)) .- softmax.(x))'

# activation function for output layer
hout = softmax
#derivative of the softmax as the derivative function for output layer
hpout = softmaxderiv

# sigmoid activation function for hidden layer
hhid = hsig
#derivative of sigmoid as the derivative function for the hidden layer
hphid = hpsig

function cross_entropy_loss(z, t)
    return -sum(t .* log.(z .+ 1e-9))  # Add epsilon to avoid log(0)
end

function compute_accuracy(w1, w2, input, target, idx)
    correct = 0
    for n in idx
        x = input[n]
        t = target[n]

        # Forward propagation
        hiddennode = hhid(w1 * x)
        y = [1.0; hiddennode]
        z = hout(w2 * y)

        predicted_label = argmax(z) - 1
        true_label = argmax(t) - 1

        if predicted_label == true_label
            correct += 1
        end
    end
    return correct / length(idx)
end

function test(w1, w2, input, target, idx)
    N = length(idx)
    error = 0.0

    for n = 1:N
        x = input[idx[n]]
        t = target[idx[n]]

        # Forward propagate
        y = [1.0; hhid(w1 * x)]  # Including bias node
        z = hout(w2 * y)

        error += cross_entropy_loss(z, t)
    end
    return error
end

function train(input, target)
    # Avoid divide by zero
    eps = 1e-8

    # Decay parameters
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.0019

    # Number of samples
    N = length(target)
    # Dimension of input
    D = length(input[1])

    # Number to hold out
    Nhold = round(Int64, N / 10)

    # Number in training set
    Ntrain = N - Nhold

    # Create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain + 1):N]

    # Number of hidden nodes
    M = 256

    # Batch size
    B = 256

    # Initialize weights
    w1 = 0.0007 * rand(M, D)
    w2 = 0.0007 * rand(10, M + 1)

    # Initialize Adam optimizer variables
    m1 = zeros(M, D)
    m2 = zeros(10, M + 1)
    v1 = zeros(M, D)
    v2 = zeros(10, M + 1)

    index = 1
    max_epochs = 30  # Or your desired number of epochs

    for epoch = 1:max_epochs
        shuffle!(trainidx)

        for i in 1:B:Ntrain
            batch_indices = trainidx[i:min(i + B - 1, Ntrain)]
            grad1 = zeros(M, D)
            grad2 = zeros(10, M + 1)

            for n in batch_indices
                x = input[n]
                t = target[n]

                # Forward propagation
                hiddennode = w1 * x
                y = [1.0; hhid(hiddennode)]
                outputnode = w2 * y
                z = hout(outputnode)

                # Compute gradients
                delta = z .- t  # Output layer error
                grad2 += delta * y'

                delta_hidden = (w2[:, 2:end]' * delta) .* hphid(hiddennode)
                grad1 += delta_hidden * x'
            end

            # Average gradients over batch size
            grad1 /= length(batch_indices)
            grad2 /= length(batch_indices)

            # Update weights using Adam optimizer
            m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
            v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
            mt1 = m1 ./ (1 - beta1 ^ index)
            vt1 = v1 ./ (1 - beta2 ^ index)
            w1 -= alpha .* mt1 ./ (sqrt.(vt1) .+ eps)

            m2 = beta1 .* m2 .+ (1 - beta1) .* grad2
            v2 = beta2 .* v2 .+ (1 - beta2) .* (grad2 .^ 2)
            mt2 = m2 ./ (1 - beta1 ^ index)
            vt2 = v2 ./ (1 - beta2 ^ index)
            w2 -= alpha .* mt2 ./ (sqrt.(vt2) .+ eps)

        index += 1
    end

    # Evaluate training and validation loss and accuracy
    train_loss = test(w1, w2, input, target, trainidx) / Ntrain
    val_loss = test(w1, w2, input, target, testidx) / (N - Ntrain)
    train_acc = compute_accuracy(w1, w2, input, target, trainidx)
    val_acc = compute_accuracy(w1, w2, input, target, testidx)

    # Print metrics for each epoch
    println("Epoch $(epoch): Training Loss = $(train_loss), Validation Loss = $(val_loss)")
    println("Epoch $(epoch): Training Accuracy = $(train_acc), Validation Accuracy = $(val_acc)")
end

# Final evaluation
final_loss = test(w1, w2, input, target, trainidx) / Ntrain
println("Final Training Loss = $(final_loss)")

return w1, w2, final_loss
end

function onehot(x)
    onehotVector = zeros(10)
    onehotVector[x + 1] = 1
    return onehotVector
end

function output()
    h5open("mnist.h5", "r") do file #read in mnist data
        labels = read(file, "train/labels")
        images = read(file, "train/images")

        input = []
        target = []

        N_size = size(labels)[1]

        for i = 1:N_size
            img = reshape(images[:, :, i], 784) / 255.0  # Normalize pixel values
            onehot_label = onehot(labels[i])
            push!(target, onehot_label)
            img = [1.0; img]  # Add bias term
            push!(input, img)
        end
        weight1, weight2, error = train(input, target)

        h5open("weights.h5", "w") do file #put into weights vector
            write(file, "weight1", weight1)
            write(file, "weight2", weight2)
        end
    end
end


@time output()
