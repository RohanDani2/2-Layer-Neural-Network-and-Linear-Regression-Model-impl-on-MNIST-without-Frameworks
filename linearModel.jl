import Pkg
using Plots, Images, Distributions, LinearAlgebra, SpecialFunctions
using HDF5, Random
using StatsPlots
import Base: error

# linear activation function for the output layer
hlin(a) = a
# and it's derivative
hplin(a) = 1

#output layer of the multiclassification 
softmax(x) = exp.(x) ./ sum(exp.(x))

#compute the gradient during backprop
softmaxderiv(x) = softmax(x) .* (Diagonal(ones(10,10)) .- softmax.(x))'

# activation function for output layer
hout = softmax
#derivative of the softmax as the derivative function for output layer
hpout = softmaxderiv

function onehot(x)
    onehotVector = zeros(10)
    onehotVector[x + 1] = 1
    return onehotVector
end

function cross_entropy_loss(z, t)
    return -sum(t .* log.(z .+ 1e-9))  # Add a small epsilon to avoid log(0)
end

function test(weight1, input, target, idx)
    N = length(idx)
    D = length(input[1])
    error = 0.0


    for n = 1:N
        x = input[idx[n]]
        t = target[idx[n]]
        a = weight1 * x
        z = hout(a)

        errorSums = cross_entropy_loss(z, t) #root mean square error

        error += errorSums
    end
    return error
end

function test_accuracy(weight1, input, target, idx)
    N = length(idx)
    correct_predictions = 0

    for n = 1:N
        x = input[idx[n]]
        t = target[idx[n]]
        a = weight1 * x
        z = hout(a)

        predicted_label = argmax(z) - 1  # Subtract 1 because labels are from 0 to 9
        true_label = argmax(t) - 1

        if predicted_label == true_label
            correct_predictions += 1
        end
    end
    return correct_predictions / N  # Accuracy as a fraction
end

function train(input, target)
    eps = 10^(-8)

    #exponential decay rate, moving average incorporates last 10 gradients
    beta1 = 0.9
    beta2 = 0.999
    #learning rate
    alpha = 0.001
    B = 500
    count = 0

    # number of samples
    N = size(target)[1]
    D = length(input[1])

    # number to hold out
    Nhold = round(Int64, N/3)
    # number in training set
    Ntrain = N - Nhold
    # create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]
    inputnode = zeros(D)
    outputnode = zeros(10)

    # layer 1 weights
    weight1 = 0.0007*randn(10,D)
    bestweight1 = weight1
    #creates a uniform probability that any sample can be used
    pdf = Uniform(1, Ntrain)
    error = [] 
    index = 1

    #setting initial arrays for the adam algorithm 
    m1 = zeros(10,D)
    v1 = zeros(10,D)

    max_epochs = 30
    batch_size = 128

    for epoch = 1:max_epochs
        shuffle!(trainidx)
        for i in 1:batch_size:Ntrain
            batch_indices = trainidx[i:min(i+batch_size-1, Ntrain)]
            grad1 = zeros(10, D)
            for n in batch_indices
                x = input[n]
                t = target[n]

                # Forward pass
                a = weight1 * x
                z = hout(a)

                # Gradient computation
                delta = z - t
                grad1 += delta * x'
            end
            grad1 /= length(batch_indices)
        
            #moving average of the adam
            m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
            #uses this line to counteract the first moments toward zero
            mt1 = m1 ./ (1 - (beta1^index))
            v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
            #counteract the second moments toward zero, divide by the expected value factor
            vt1 = v1 ./ (1 - (beta2^index))
            #updating weights 
            weight1 += -alpha .* mt1 ./ (sqrt.(vt1) .+ eps)
            
            index += 1
        end

        # Evaluate performance
        train_loss = test(weight1, input, target, trainidx) / Ntrain
        train_acc = test_accuracy(weight1, input, target, trainidx)
        val_acc = test_accuracy(weight1, input, target, testidx)
        println("Epoch $(epoch): Training Loss = $(train_loss), Training Accuracy = $(train_acc), Validation Accuracy = $(val_acc)")

        # Early stopping condition
        if train_acc >= 0.98
            break
        end
    end

    # Final evaluation
    final_loss = test(weight1, input, target, trainidx) / Ntrain
    println("Final Training Loss = $(final_loss)")

    return weight1, final_loss
end


function output()
    h5open("mnist.h5", "r") do file

        #read in train mnist labels and images
        labels = read(file, "train/labels")
        images = read(file, "train/images")

        input = []
        target = []

        N_size = size(labels)[1]

        for i = 1:N_size
            img = reshape(images[:,:,i], 784) / 255.0
            prepend!(img, 1)
            targetnew = onehot(labels[i])
            push!(target, targetnew)
            push!(input, img)
        end
        weight1, err = train(input, target)

        h5open("weights.h5", "w") do file
            write(file, "weight1", weight1)
        end
    end
end

@time output()
