# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes);
    if numClasses<=2
        classes = reshape(classes.==classes[1], :, 1);
    else
        oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in feature]...)');
        classes = oneHot;
    end;
    return classes;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mx = maximum(dataset, dims= 1)
    mn = minimum(dataset, dims= 1)
    return (mx, mn)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mn = mean(dataset, dims = 1)
    sigma = std(dataset, dims = 1)
    return (mn, sigma)
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mx, mn = normalizationParameters
    return (dataset[:, :] .- mn) ./ (mx -mn)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    mx, mn = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, (mx,mn))
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mx, mn = normalizationParameters
    data = copy(dataset)
    return (data[:,:] .- mn) ./ (mx -mn)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    mx, mn = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax(dataset, (mx, mn))
end;


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mn, sigma = normalizationParameters
    return (dataset .- mn) ./  sigma
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    mn, sigma = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, (mn, sigma))
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mn, sigma = normalizationParameters
    data = copy(dataset)
    return (data .- mn) ./  sigma
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    mn, sigma = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean(dataset, (mn, sigma))
end;

function classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real=0.5)
    outputs = outputs.>= threshold
    return outputs;
end

function classifyOutputs(outputs::AbstractArray{<:Real, 2}; threshold::Real=0.5)
    if size(outputs)[2] == 1
        outputs = outputs[:]
        return reshape(classifyOutputs(outputs[:]; threshold), :, 1)
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs  
    end   
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(targets .== outputs)
end

function accuracy(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2})
    if size(targets)[2] == 1 && size(outputs)[2]==1
        return accuracy(outputs[:, 1], targets[:,1])
    elseif size(targets)[2] == 2 && size(outputs)[2]==2
        return accuracy(outputs[:, 1], targets[:,1])
    elseif size(targets)[2] > 2 && size(outputs)[2] > 2
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        accuracy = mean(correctClassifications)
        return accuracy
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = outputs .>= threshold
    return accuracy(targets, new_outputs)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};threshold::Real=0.5)
    if size(targets, 2) == 1 && size(outputs, 2) == 1
        outputs = outputs[:, 1]
        targets = targets[:, 1]
        return accuracy(outputs, targets; threshold)
    elseif size(targets, 2) == 2 && size(outputs, 2) == 2
        outputs = outputs[:, 1]
        targets = targets[:, 1]
        return accuracy(outputs, targets; threshold)
    elseif size(targets, 2) > 2 && size(outputs, 2) > 2
        outputs = classifyOutputs(outputs)
        return accuracy(outputs, targets) 
    end  
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain();
    numInputsLayer = numInputs
    
    for numOutputsLayer = topology 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions) ); 
        numInputsLayer = numOutputsLayer; 
    end;

    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, transferFunctions), softmax )
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
    end;
    
    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    entradas, salidas = dataset
    entradas = convert(Array{Float32,2},entradas);
    entradas_t = transpose(entradas)
    salidas_t = transpose(salidas)
    ann = buildClassANN(size(entradas_t)[1], topology, size(salidas_t)[2]; transferFunctions)
    loss(ann, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    
    loss_total = Float32[]

    optimizador = Flux.setup(Adam(learningRate), ann)

    append!(loss_total, loss(ann,entradas_t,salidas_t))

    for i in 1:maxEpochs
        Flux.train!(loss, ann, [(entradas_t, salidas_t)], optimizador);
        loss_actual = loss(ann,entradas_t,salidas_t)
        append!(loss_total, loss_actual)
        if loss_actual <= minLoss
            break
        end
        return(ann,loss_total)
    end
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    entradas, salidas = dataset
    salidas = reshape(salidas, :, 1);
    dataset = (entradas,salidas);
    trainClassANN(topology,dataset;transferFunctions)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    test = randperm(N)
    test = test[1:Int(round(N*P))]
    training = filter(x -> x ∉ test, 1:N)
    return (training,test)
end;


function holdOut(N::Int, Pval::Real, Ptest::Real)
    values = holdOut(N,Ptest)
    training = values[1]
    test = values[2]
    Pval =   Pval / (1-Ptest)
    values = holdOut(length(training),Pval)
    aux = training
    training = training[values[1]]
    validation = aux[values[2]]
    return (training,validation,test)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    entradas_val, salidas_val = validationDataset
    entradas_test, salidas_test = testDataset
    entradas_train, salidas_train = trainingDataset

    entradas_val = convert(Array{<:Float32,2}, entradas_val); 
    entradas_test = convert(Array{<:Float32,2}, entradas_test);
    entradas_train = convert(Array{Float32,2},entradas_train);
    
    entradas_val_t = transpose(entradas_val)
    salidas_val_t = transpose(salidas_val)
    
    entradas_train_t = transpose(entradas_train)
    salidas_train_t = transpose(salidas_train)

    entradas_test_t = transpose(entradas_test)
    salidas_test_t = transpose(salidas_test)
    
    ann = buildClassANN(size(entradas_train_t)[1], topology, size(salidas_train_t)[2]; transferFunctions)
   
    loss(ann, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    
    losses_train = Float32[]
    losses_val = Float32[];
    losses_test = Float32[];


    optimizador = Flux.setup(Adam(learningRate), ann)

    loss_inic_train =  loss(ann,entradas_train_t,salidas_train_t)
    push!(losses_train, loss_inic_train)

    if !isempty(entradas_val)
        loss_inic_val = loss(ann,entradas_val_t,salidas_val_t)
        push!(losses_val, loss_inic_val)
    end
    
    if !isempty(entradas_test)
        loss_inic_test = loss(ann,entradas_test_t,salidas_test_t)
        push!(losses_test, loss_inic_test)
    end

    best_val_loss = Inf32
    lim_epoch_cicles = 0
    ann_final = deepcopy(ann)

    for i in 1:maxEpochs

        Flux.train!(loss, ann, [(entradas_train_t, salidas_train_t)], optimizador);
        loss_train =  loss(ann,entradas_train_t,salidas_train_t)
        push!(losses_train, loss_train)
        
        if !isempty(entradas_val)
            loss_val = loss(ann,entradas_val_t,salidas_val_t)
            push!(losses_val, loss_val)
            if loss_val < best_val_loss
                best_val_loss = loss_val;
                lim_epoch_cicles = 0;
                ann_final = deepcopy(ann);
            else
                lim_epoch_cicles += 1;

                if lim_epoch_cicles == 20
                    break
                end 
            end;
        else
            ann_final = ann
        end

        if !isempty(entradas_test)
            push!(losses_test, loss(ann, entradas_test_t, salidas_test_t))
        end
    end
    return (ann_final, losses_train, losses_val, losses_test)
end
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    entradas_train, salidas_train = trainingDatraset
    entradas_val, salidas_val = validationDataset
    entradas_test, salidas_test = testDataset

    salidas_train = reshape(salidas_train, :, 1);
    salidas_val = reshape(salidas_val, :, 1);
    salidas_test = reshape(salidas_test, :, 1);

    trainingDataset = (entradas_train, salidas_train);
    validationDataset = (entradas_val, salidas_val);
    testDataset = (entradas_test, salidas_test);
    
    trainClassANN(topology, trainingDataset; validationDataset, testDataset, transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;
