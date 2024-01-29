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

function test(weight1, input, target, idx)
    N = length(idx)
    D = length(input[1])
    error = 0.0


    for n = 1:N
        x = input[idx[n]]
        t = target[idx[n]]
        a = weight1 * x
        z = hout(a)

        errorSums = sum(((z .- t).^2)) #root mean square error

        error += errorSums
    end
    return error
end

function train(input, target)
    eps = 10^(-8)

    #exponential decay rate, moving average incorporates last 10 gradients
    beta1 = 0.9
    beta2 = 0.999
    #learning rate
    alpha = 0.00002
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

    while count < 3000
        grad1 = zeros(10,D)
        for n = 1:B #taking a batch of 500 values
            sample = trainidx[round(Int64, rand(pdf, 1)[1])] #chooses a training sample from uniform distribution
            #x is the pixel values of the digits
            x = input[sample]
            #one hot encoded labels
            t = target[sample]

            inputnode = x
            outputnode = weight1 * inputnode

            z = hout(outputnode)

            delta = z.-t
            grad1 = (hpout(outputnode) * delta * x')
        end
        grad1 = grad1 / B
        
        #moving average of the adam
        m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
        #uses this line to counteract the first moments toward zero
        mt1 = m1 ./ (1 - (beta1 ^ index))
        v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
        #counteract the second moments toward zero, divide by the expected value factor
        vt1 = v1 ./ (1 - (beta2 ^ index))
        #updating weights 
        weight1 += -alpha .* mt1 ./ (sqrt.(vt1) .+ eps)

        errorrate = test(weight1, input, target, trainidx)/Ntrain
        push!(error, errorrate)

        println("Training Error = $(errorrate)")
        if (errorrate <= 0.12) #if error rate hits 0.12 then I stop it
            break
        end
        index = index + 1
        count = count + 1
    end
    error = test(weight1, input, target, trainidx)/Ntrain
    println("Final Testing Error = $(error)")

    return weight1, error
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
            img = reshape(images[:,:,i], 784)
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

function onehot(x)
    onehotVector = zeros(10)
    onehotVector[x + 1] = 1
    return onehotVector
end

@time output()
