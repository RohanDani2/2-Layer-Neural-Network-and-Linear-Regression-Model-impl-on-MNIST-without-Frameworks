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

function test(w1, w2, input, target, idx)

    N = length(idx)
    D = length(input[1])
    M = size(w1)[1]

    error = 0

    for n = 1:N
        x = input[idx[n]]
        t = target[idx[n]]

        # forward propagate
        y = zeros(M+1)

        y[1] = 1 # bias node
        a = w1 * x
        y[2:end] = hhid(a)
        a = 0
        a = w2 * y
        z = hout(a)

        errors = sum(((z .- t).^2))
        error += errors
    end
    return error
end

function train(input, target)
    #avoid divide by zero
    eps = 10^(-8)

    #decay parameters 
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.0019

    # number of samples
    N = length(target)
    # dimension of input
    D = length(input[1])

    # number to hold out
    Nhold = round(Int64, N/10)

    # number in training set
    Ntrain = N - Nhold

    # create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]
    #print(size(trainidx))

    #println("$(length(trainidx)) training samples")
    #println("$(length(testidx)) validation samples")

    # number of hidden nodes
    M = 256

    # batch size
    B = 256

    # input layer activation
    inputnode = zeros(D)

    # hidden layer activation
    hiddennode = zeros(M)

    # output node activation
    outputnode = zeros(10)

    # layer 1 weights
    w1 = .0007*rand(M, D)
    bestw1 = w1

    # layer 2 weights (inc bias)
    w2 = .0007*rand(10,M+1)
    bestw2 = w2

    pdf = Uniform(1,Ntrain)
    error = []

	#initialize the values of the parameters for Adam

	m1 = zeros(M, D)
	m2 = zeros(10,M+1)
	v1 = zeros(M, D)
	v2 = zeros(10,M+1)

    index = 1
    count = 1
    while count < 3000

        grad1 = zeros(M, D)
        grad2 = zeros(10,M+1)

        for n = 1:B
            sample = trainidx[round(Int64, rand(pdf, 1)[1])]

            #sample = rand(Ntrain, B)
            x = input[sample]
            t = target[sample]
            # forward propagate
            inputnode = x
            y = zeros(M+1)

            y[1] = 1 # bias node
            hiddennode = w1*inputnode
            #print(hiddennode)
            y[2:end] = hhid(hiddennode)

            # for j = 1:M+1
            #     outputnode += w2[:,j]*y[j]
            # end
            outputnode = w2*y
            z = hout(outputnode)
            # end forward propagate

            # output error
            delta = z .- t
            # for j = 1:M+1
            #     if j == 1 # bias node
            #         grad2[1,j] += delta*y[j]
            #     else
            #         grad2[1,j] += delta*y[j]*hpout(outputnode)
            #     end
            # end
            # compute layer 2 gradients by backpropagation of delta

            grad2[:,1] = delta * y[1]
            grad2[:,2:end] = hpout(z) * delta * (y[2:end])'

            # compute layer 1 gradients by backpropagation of deltaj
            for j = 1:M
                grad1[j,:] += delta'*w2[:,j+1]*hphid(hiddennode[j])*x[:]
            end
    	end


        # update weights using Adam EWMA

        #gradients calculated during backprop
        grad2 = grad2 / B
        grad1 = grad1 / B

        #momentum, using leaky average this gives bias to recent graidents, adding inertia
        m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
        mt1 = m1 ./ (1 - (beta1 ^ index))

        #exponential weighted moving average of squared gradient, square gives velocity
        v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
        vt1 = v1 ./ (1 - beta2 ^ index)

        #weight_update = learning_rate * m / (sqrt(v) + epsilon)
        w1 += -alpha .* mt1 ./ (sqrt.(vt1) .+ eps)

        m2 = beta1 .* m2 .+ (1 - beta1) .* grad2
        mt2 = m2 ./ (1 - (beta1 ^ index))
        v2 = beta2 .* v2 .+ (1 - beta2) .* (grad2 .^ 2)
        vt2 = v2 ./ (1 - beta2 ^ index)
        w2 += -alpha .* mt2 ./ (sqrt.(vt2) .+ eps)
        #println("weight 1 norm")
        #print(norm(w1))
        #println("weight 2 norm")
        #print(norm(w2))

        errorrate = test(w1, w2, input, target, trainidx)/Ntrain
        push!(error, errorrate)
        println("Training Error = $(errorrate)")
        if (errorrate <= 0.15)
            break
        end
        index = index + 1
        count = count + 1
	end
    error = test(w1, w2, input, target, trainidx)/Ntrain
    println("Final Train Error = $(error)")

    return w1, w2, error
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
            img = reshape(images[:,:,i], 784) #reshape data to 60000 by 784
            onehot_label = onehot(labels[i]) #onehot encode label
            push!(target, onehot_label)
            prepend!(img, 1)
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
