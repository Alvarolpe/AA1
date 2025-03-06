# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes);
    if numClasses<=2
        classes = reshape(feature.==classes[1], :, 1);
    else
        oneHot = convert(BitArray{2}, hcat([classes .== f for f in feature]...)');
        classes = oneHot;
    end;
    return classes;
end;


oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = return reshape(feature, :, 1)


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mx = maximum(dataset, dims= 1)
    mn = minimum(dataset, dims= 1)
    return (mn, mx)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mn = mean(dataset, dims = 1)
    sigma = std(dataset, dims = 1)
    return (mn, sigma)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mn, mx = normalizationParameters
    dataset .-= mn
    dataset ./= (mx.-mn)
    dataset = dataset[:,vec(mn .== mx)] .= 0
    return dataset;
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
end;


function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    data = copy(dataset)
    normalizeMinMax!(data, normalizationParameters)
end;



function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    data = copy(dataset)
    normalizeMinMax!(data, normalizationParameters)
end;


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mu, sigma = normalizationParameters
    dataset .-= mu
    dataset ./= sigma
    dataset = dataset[:,vec(sigma .== 0)] .= 0
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    data = copy(dataset)
    normalizeZeroMean!(data,normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    data = copy(dataset)
    normalizeZeroMean!(data, normalizationParameters)
end;

function classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real=0.5)
    outputs = outputs.>= threshold
    return outputs;
end

function classifyOutputs(outputs::AbstractArray{<:Real, 2}; threshold::Real=0.5)
    if size(outputs)[2] == 1
        outputs = outputs[:]
        return reshape(classifyOutputs(outputs[:]; threshold = threshold), :, 1)
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs  
    end   
end




function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(targets .== outputs)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(targets)[2] <= 2 && size(outputs)[2]<=2
        return accuracy(outputs[:, 1], targets[:,1])
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        precision = mean(correctClassifications)
        return precision
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = outputs .>= threshold
    return accuracy(new_outputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(targets)[2] <= 2 && size(outputs)[2] <= 2
        outputs = outputs[:, 1]
        targets = targets[:, 1]
        return accuracy(outputs, targets; threshold = threshold)
    else
        outputs = classifyOutputs(outputs)
        return accuracy(outputs, targets) 
    end  
end;



function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain();
    numInputsLayer = numInputs
    i = 1
    for numOutputsLayer = topology 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i])); 
        numInputsLayer = numOutputsLayer; 
        i += 1
    end;
    if numOutputs > 2
        ann = Chain(ann..., Dense(topology[end], numOutputs), softmax)
    else
        ann = Chain(ann..., Dense(topology[end], 1, transferFunctions[end]))
    end
    return ann
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    entradas, salidas = dataset
    entradas = convert(Array{Float32,2},entradas);
    entradas_t = transpose(entradas)
    salidas_t = transpose(salidas)
    ann = buildClassANN(size(entradas_t)[1], topology, size(salidas_t)[2]; transferFunctions=transferFunctions)
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
    end
    return (ann,loss_total)
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    entradas, salidas = dataset
    salidas = reshape(salidas, :, 1);
    dataset = (entradas,salidas);
    trainClassANN(topology,dataset;transferFunctions=transferFunctions)
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
    
    #Obtención de datos de entrenamiento, validación y test
    entradas_val, salidas_val = validationDataset
    entradas_test, salidas_test = testDataset
    entradas_train, salidas_train = trainingDataset

    #Conversión a Float32
    entradas_val = convert(Array{Float32,2}, entradas_val); 
    entradas_test = convert(Array{Float32,2}, entradas_test);
    entradas_train = convert(Array{Float32,2},entradas_train);
    
    #Transponemos las matrices de entradas y salidas de cada conjunto
    entradas_val_t = transpose(entradas_val)
    salidas_val_t = transpose(salidas_val)
    
    entradas_train_t = transpose(entradas_train)
    salidas_train_t = transpose(salidas_train)

    entradas_test_t = transpose(entradas_test)
    salidas_test_t = transpose(salidas_test)
    
    #Creamos la RNA
    ann = buildClassANN(size(entradas_train_t)[1], topology, size(salidas_train_t)[1]; transferFunctions = transferFunctions)
   
    #Definimos la función de pérdida
    loss(ann,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    
    #Creamos las listas de losses
    losses_train = Float32[];
    losses_val = Float32[];
    losses_test = Float32[];

    #Definimos el optimizador
    optimizador = Flux.setup(Adam(learningRate), ann)


    #Calculamos los losses iniciales (antes de entrenar)
    loss_inic_train =  loss(ann,entradas_train_t,salidas_train_t)
    push!(losses_train, loss_inic_train)

    #Definimos las varibles necesarias para el citerio de parada
    best_val_loss = Inf32
    lim_epoch_cicles = 0
    ann_final = deepcopy(ann)

    #print(loss_inic_train)
    if !isempty(entradas_val)
        loss_inic_val = loss(ann,entradas_val_t,salidas_val_t)
        push!(losses_val, loss_inic_val)
        best_val_loss = loss_inic_val
    end
    
    if !isempty(entradas_test)
        loss_inic_test = loss(ann,entradas_test_t,salidas_test_t)
        push!(losses_test, loss_inic_test)
    end

    #Iniciamos los ciclos de entrenamiento

    for i in 1:maxEpochs
        #print(i)
        #print(" ")
        Flux.train!(loss, ann, [(entradas_train_t, salidas_train_t)], optimizador);
        loss_train =  loss(ann,entradas_train_t,salidas_train_t)
        push!(losses_train, loss_train)
       
        if !isempty(entradas_test)
            push!(losses_test, loss(ann, entradas_test_t, salidas_test_t))
        end

        #Criterio de parada
        if !isempty(entradas_val)
            loss_val = loss(ann,entradas_val_t,salidas_val_t)
            push!(losses_val, loss_val)

            if loss_val < best_val_loss
                best_val_loss = loss_val;
                lim_epoch_cicles = 0; #Reinicio de los ciclos
                ann_final = deepcopy(ann);
            else
                #print(loss_val)
                lim_epoch_cicles += 1;
                if lim_epoch_cicles == maxEpochsVal
                    print(loss(ann_final,entradas_train_t,salidas_train_t))
                    return (ann_final, losses_train, losses_val, losses_test)
                end 
            end;
        else
            if loss_train <= minLoss
                break
            end
        end
    end
    #print("el los final es")
    #print(loss(ann,entradas_train_t,salidas_train_t))
    return (ann, losses_train, losses_val, losses_test)
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    entradas_val, salidas_val = validationDataset
    entradas_test, salidas_test = testDataset
    entradas_train, salidas_train = trainingDataset

    trainClassANN(topology, (entradas_train, reshape(salidas_train, length(salidas_train), 1));
    validationDataset=(entradas_val, reshape(salidas_val, length(salidas_val), 1)),
    testDataset=(entradas_test, reshape(salidas_test, length(salidas_test), 1)),
    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
end
# Ejercicio 4

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VN = sum(outputs .== 0 .&& targets .== 0)
    FP = sum(outputs .== 0 .&& targets .== 1)
    VP = sum(outputs .== 1 .&& targets .== 1)
    FN = sum(outputs .== 1 .&& targets .== 0)

    precision = (VN + VP)/(VN + VP + FN + FP)
    failure = (FN + FP)/(VN + VP + FN + FP)
    recall = VP/(FN+VP)
    especificity = VN/(FP+VN)
    positivity = VP/(VP+FP)
    negativity = VN/(VN+FN)
    F1_score = 2 * (positivity * recall) / (positivity + recall)
    confusion_matrix = reshape([VN, FN, FP, VP], 2, 2)
    output = [precision,failure,recall,especificity ,positivity, negativity, F1_score]
    output = map(x -> isnan(x) ? 0.0 : x, output)
    precision,failure,recall,especificity ,positivity, negativity, F1_score = Tuple(output)
    return (precision,failure,recall,especificity ,positivity, negativity, F1_score,confusion_matrix)
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
