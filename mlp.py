import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2        # l2 regularization reduces overfitting?
import matplotlib.pyplot as plt
import numpy as np
import math, sys
from colorama import Fore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# splits data dict into different dicts based on the ratios given
def dictSplit(data, trainingRatio, testingRatio, validationRatio=0.0, randomSeed=42):
    # make sure the ratios add up to 1
    if trainingRatio + testingRatio + validationRatio != 1:
        print(f"{Fore.RED}ERROR: Ratios for data split do not add up to 1{Fore.WHITE}")
        return
    if trainingRatio <= 0 or testingRatio < 0 or validationRatio < 0:
        print(f"{Fore.RED}ERROR: No ratios can be less than 0 & training ratio can't be 0{Fore.WHITE}")
        return

    numRounds = len(data['scores']) // 2
    
    # set seed for reproducible splits
    np.random.seed(randomSeed)
    
    # randomly shuffle the indices of the rounds
    roundIndices = np.random.permutation(numRounds)

    # calculate actual counts
    numTesting = math.floor(numRounds * testingRatio)
    numValidation = math.floor(numRounds * validationRatio)
    numTraining = numRounds - numTesting - numValidation

    # slways allocate test set first to ensure consistency
    # This ensures the same test set regardless of validation set presence
    testingIndices = roundIndices[:numTesting]
    
    if validationRatio > 0:
        # if validation set exists, take it from the remaining indices
        validationIndices = roundIndices[numTesting:numTesting + numValidation]
        trainingIndices = roundIndices[numTesting + numValidation:]
    else:
        # no validation set - all remaining indices go to training
        validationIndices = roundIndices[numTesting:numTesting]  # empty slice
        trainingIndices = roundIndices[numTesting:]

    # create new dictionaries for each set
    trainingDict = {key: [] for key in data.keys()}
    testingDict = {key: [] for key in data.keys()}
    validationDict = {key: [] for key in data.keys()}

    # populate the new dictionaries
    for i in range(numRounds):
        startIdx = i * 2
        endIdx = startIdx + 2
        if i in trainingIndices:
            targetDict = trainingDict
        elif i in testingIndices:
            targetDict = testingDict
        else:
            targetDict = validationDict
        
        for key in data.keys():
            targetDict[key].extend(data[key][startIdx:endIdx])

    # check if adjacent indices correspond to the same round
    for dataDict in [trainingDict, testingDict, validationDict]:
        for i in range(0, len(dataDict['scores']), 2):
            if len(dataDict['scores']) > i + 1:  # ensure we have a pair
                if dataDict['name'][i] != dataDict['nameO'][i+1] or dataDict['nameO'][i] != dataDict['name'][i+1]:
                    print(f"{Fore.RED}ERROR: Adjacent rounds don't match (indices {i} & {i+1}){Fore.WHITE}")
                if dataDict['round'][i] != dataDict['round'][i+1] or dataDict['date'][i] != dataDict['date'][i+1]:
                    print(f"{Fore.RED}ERROR: Adjacent rounds don't match (indices {i} & {i+1}){Fore.WHITE}")

    print("\nCompleted:\tData split\n")

    print(f"Rounds in training set:\t\t{len(trainingDict['scores'])//2}")
    print(f"Rounds in testing set:\t\t{len(testingDict['scores'])//2}")
    print(f"Rounds in validation set:\t{len(validationDict['scores'])//2}")

    return trainingDict, testingDict, validationDict

# takes in split data dictionaries and returns np arrays to be used in MLP
def preprocessData(trainingDict, testingDict, validationDict, params, judgeName, randomSeed=42):

    # helper function to turn data dictionaries int np arrays
    # results in 6 data points per round (3 for each boxer) if judgeName is None
    def processDict(dataDict):
        # calculate +/- differences for each parameter (like gradient descent)
        features = []
        for param in params:
            if 'total' in param or 'Share' in param:
                features.append(param)
            else:
                features.append(param + '_diff')
        
        # create the feature array
        featureMatrix = np.empty((len(dataDict['scores']), len(features)))
        for i in range(len(dataDict['scores'])):
            for j, param in enumerate(params):
                if 'total' in param or 'Share' in param:
                    featureMatrix[i, j] = dataDict[param][i]
                else:
                    fighterValue = dataDict[param][i]
                    opponentValue = dataDict[param + 'O'][i]
                    featureMatrix[i, j] = fighterValue - opponentValue

        # store original combined scores for evaluation
        originalScores = np.array(dataDict['scores'])

        # make data['scores'] binary
        if judgeName is None:
            # set seed for any random operations in this function
            np.random.seed(randomSeed)
            # create np array of scores of length numRounds*3 (dict already has 2 entries points for each round)
            scores = np.empty(len(dataDict['scores']) * 3)
            # iterate through each entry (two per round)
            for i in range(0, len(dataDict['scores'])):
                # add one value to np array for each judge
                for j in range(1,4):
                    if dataDict['score'+str(j)][i] == 1:
                        scores[3*i+(j-1)] = 1
                    elif dataDict['score'+str(j)][i] == -1:
                        scores[3*i+(j-1)] = 0
                    # judge scores should only be 1 or -1
                    else:
                        print(f"{Fore.RED}ERROR: Invalid judge score value (index: {i},  judge: {j}){Fore.WHITE}")

            # expand featureMatrix to match the expanded scores
            featureMatrix = np.repeat(featureMatrix, 3, axis=0)
            # expand original scores to match
            originalScores = np.repeat(originalScores, 3)

        # if using data from 1 specific judge, don't expand each entry into 3 binary scores
        else:
            scores = np.array([(score + 1) / 2 for score in dataDict['scores']])

        # make sure featureMatrix and scores have the same number of samples
        assert featureMatrix.shape[0] == scores.shape[0], f"Mismatch between number of samples in features ({featureMatrix.shape[0]}) and scores ({scores.shape[0]})"

        return featureMatrix, scores, originalScores

    # process each dictionary
    xTrain, yTrain, originalTrainScores = processDict(trainingDict)
    xTest, yTest, originalTestScores = processDict(testingDict)
    xValidation, yValidation, originalValidationScores = processDict(validationDict)

    # normalize the stats
    scaler = StandardScaler()
    xTrainNormalized = scaler.fit_transform(xTrain)
    xTestNormalized = scaler.transform(xTest)
    xValidationNormalized = scaler.transform(xValidation)

    return (xTrainNormalized, xTestNormalized, xValidationNormalized, 
            yTrain, yTest, yValidation, 
            originalTrainScores, originalTestScores, originalValidationScores, 
            scaler)


def buildModel(inputShape):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(inputShape,), 
                          kernel_regularizer=l2(0.001)),  # reduced from 0.01
        keras.layers.Dropout(0.2),  # reduced from 0.3
        keras.layers.Dense(16, activation='relu', 
                          kernel_regularizer=l2(0.001)),  # reduced from 0.01
        keras.layers.Dropout(0.1),  # reduced from 0.2
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

# rescale MLP predictions from [0,1] to [-1,1] scale
def rescalePredictions(predictions, judgeName, originalScores=None, returnAsNumpy=False):
    """
    rescale MLP predictions from [0,1] to [-1,1] scale
    
    Args:
        predictions: model predictions in [0,1] scale
        judgeName: judge name (None for all judges, string for specific judge)
        originalScores: optional original scores to also convert
        returnAsNumpy: if True, return numpy arrays; if False, return lists
    
    Returns:
        if originalScores is None: converted predictions only
        if originalScores provided: (converted predictions, converted original scores)
    """
    convertedPredictions = []
    convertedOriginalScores = []
    
    if judgeName is None:
        # average every 3 predictions and convert to [-1,1] scale
        for i in range(0, len(predictions), 3):
            avgPrediction = np.mean(predictions[i:i+3])
            convertedPredictions.append(2 * avgPrediction - 1)
            
            if originalScores is not None:
                # take the original score (should be same for all 3 entries)
                convertedOriginalScores.append(originalScores[i])
    else:
        # convert predictions to [-1,1] scale
        for pred in predictions.flatten():
            convertedPredictions.append(2 * pred - 1)
            
        if originalScores is not None:
            convertedOriginalScores = originalScores.tolist()
    
    # convert to numpy arrays if requested
    if returnAsNumpy:
        convertedPredictions = np.array(convertedPredictions)
        if originalScores is not None:
            convertedOriginalScores = np.array(convertedOriginalScores)
    
    # return based on what was requested
    if originalScores is not None:
        return convertedPredictions, convertedOriginalScores
    else:
        return convertedPredictions
    

def calculateCustomMetrics(convertedPredictions, convertedOriginalScores):
    """
    calculate MSE and sign agreement accuracy on [-1,1] scale
    """
    # calculate MSE on [-1,1] scale
    mse = np.mean((convertedPredictions - convertedOriginalScores) ** 2)
    
    # calculate sign agreement accuracy
    signAgreement = np.sum(np.sign(convertedPredictions) == np.sign(convertedOriginalScores))
    accuracy = signAgreement / len(convertedPredictions)
    
    return mse, accuracy

def train(model, xTrain, yTrain, xValidation, yValidation, originalTrainScores, originalValidationScores, judgeName, epochs=100, batchSize=32):
    earlyStop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-7, verbose=1)

    history = model.fit(
        xTrain, yTrain, 
        epochs=epochs, 
        batch_size=batchSize, 
        validation_data=(xValidation, yValidation),
        callbacks=[earlyStop, reduceLr],  # updated this line
        verbose=1
    )
    
    # get the epoch with the best validation loss
    bestEpoch = np.argmin(history.history['val_loss'])
    
    # get predictions for custom metric calculation
    trainPredictions = model.predict(xTrain)
    valPredictions = model.predict(xValidation)
    
    # convert to original scale and calculate custom metrics
    convertedTrainPreds, convertedTrainScores = rescalePredictions(trainPredictions, judgeName, originalTrainScores, returnAsNumpy=True)
    convertedValPreds, convertedValScores = rescalePredictions(valPredictions, judgeName, originalValidationScores, returnAsNumpy=True)
    
    trainMSE, trainAccuracy = calculateCustomMetrics(convertedTrainPreds, convertedTrainScores)
    valMSE, valAccuracy = calculateCustomMetrics(convertedValPreds, convertedValScores)
    
    print("\nTraining Performance")
    print(f"\tKeras Accuracy    : {(100*history.history['accuracy'][bestEpoch]):.2f}%")
    print(f"\tKeras Loss        : {history.history['loss'][bestEpoch]:.4f}")
    print(f"\tSign Agreement    : {(100*trainAccuracy):.2f}%")
    print(f"\tMSE [-1,1] Scale  : {trainMSE:.4f}")
    
    print("Validation Performance")
    print(f"\tKeras Accuracy    : {100*(history.history['val_accuracy'][bestEpoch]):.2f}%")
    print(f"\tKeras Loss        : {history.history['val_loss'][bestEpoch]:.4f}")
    print(f"\tSign Agreement    : {(100*valAccuracy):.2f}%")
    print(f"\tMSE [-1,1] Scale  : {valMSE:.4f}")
    
    return history

def predict(model, xTest):
    return model.predict(xTest)

def evalModel(model, xTest, yTest, originalTestScores, judgeName):
    print("\nEvaluating on test set...")
    
    # get keras evaluation
    loss, accuracy = model.evaluate(xTest, yTest)
    
    # get predictions for custom evaluation
    predictions = model.predict(xTest)
    
    # convert to original scale and calculate custom metrics
    convertedPreds, convertedScores = rescalePredictions(predictions, judgeName, originalTestScores, returnAsNumpy=True)
    testMse, testAccuracy = calculateCustomMetrics(convertedPreds, convertedScores)
    
    # calculate unanimous vs split round breakdown
    unanimousTotal = 0
    unanimousCorrect = 0
    splitTotal = 0
    splitCorrect = 0
    
    # iterate through each round (convertedScores has one entry per round after rescaling)
    for i in range(len(convertedScores)):
        roundScore = convertedScores[i]
        roundPrediction = convertedPreds[i]
        
        # determine if unanimous or split decision
        if abs(roundScore) == 1:  # unanimous decision
            unanimousTotal += 1
            if np.sign(roundPrediction) == np.sign(roundScore):
                unanimousCorrect += 1
        else:  # split decision (abs(roundScore) == 1/3)
            splitTotal += 1
            if np.sign(roundPrediction) == np.sign(roundScore):
                splitCorrect += 1
    
    print("\nTesting Performance")
    print(f"\tUnanimous Rounds:\t\t\t\t{unanimousTotal/2}")
    print(f"\tSplit Rounds:\t\t\t\t\t{splitTotal/2}")
    print(f"\tKeras Accuracy    :\t\t\t\t\t{(100*accuracy):.2f}%")
    print(f"\tKeras Loss        :\t\t\t\t\t{loss:.4f}")
    
    # print unanimous vs split accuracy
    if unanimousTotal > 0:
        unanimousAccuracy = 100 * unanimousCorrect / unanimousTotal
        print(f"\tMajority Agreement for Unanimous Rounds:\t{unanimousAccuracy:.3f}%")
    
    if splitTotal > 0:
        splitAccuracy = 100 * splitCorrect / splitTotal
        print(f"\tMajority Agreement for Split Rounds:\t\t{splitAccuracy:.3f}%")
    
    print(f"\tSign Agreement    :\t\t\t\t\t{(100*testAccuracy):.2f}%")
    print(f"\tMSE [-1,1] Scale  :\t\t\t\t\t{testMse:.4f}")
    
    # calculate individual judge agreement (equivalent to keras accuracy)
    totalDecisions = 3 * (unanimousTotal + splitTotal)
    correctDecisions = (3 * unanimousCorrect + 2 * splitCorrect + 1 * (splitTotal - splitCorrect))
    if totalDecisions > 0:
        individualJudgeAccuracy = 100 * correctDecisions / totalDecisions
        print(f"\tIndividual Judge Agreement:\t\t\t{individualJudgeAccuracy:.3f}%")
    
    return testMse, testAccuracy

        



def mlp(trainingDict, testingDict, validationDict, params, judgeName, splitRatio=0.1, randomSeed=42):

    # preprocess the already split data dictionaries
    (xTrain, xTest, xValidation, yTrain, yTest, yValidation, 
     originalTrainScores, originalTestScores, originalValidationScores, scaler) = preprocessData(
        trainingDict, testingDict, validationDict, params, judgeName, randomSeed)

    print(f"Parameters: {params}")
    print(f"Training data shape: {xTrain.shape}")
    print(f"Training labels shape: {yTrain.shape}")
    print(f"Test data shape: {xTest.shape}")
    print(f"Test labels shape: {yTest.shape}")

    # build model
    inputShape = xTrain.shape[1]  # get the number of features
    model = buildModel(inputShape)
    print("\nCompleted:\tModel built\n")

    # train model
    history = train(model, xTrain, yTrain, xValidation, yValidation, 
                   originalTrainScores, originalValidationScores, judgeName, 
                   epochs=200, batchSize=64)

    # evaluate model based on test data
    loss, accuracy = evalModel(model, xTest, yTest, originalTestScores, judgeName)

    print("\nCompleted:\tTraining of Multi-layer Perceptron Model\n")

    # make predictions on testing data only
    print("Making predictions for testing rounds...")
    testingPredictions = predict(model, xTest)
    convertedTestingPredictions = rescalePredictions(testingPredictions, judgeName)
    
    # make predictions on all data
    print("Making predictions for all rounds...")
    # NOTE xAll & allPredictions have 6 entries per round
    xAll = np.concatenate((xTrain, xTest, xValidation), axis=0)
    allPredictions = predict(model, xAll)

    print(f"\nCompleted:\t{len(allPredictions)//6} Rounds Predicted Based On MLP\n")

    # convert data and predictions back to original format

    # create a new dictionary with all data and converted predictions
    # NOTE mlpData is built using non-normalized dicts so there are 2 entries per round
    mlpData = {}
    for key in trainingDict.keys():
        mlpData[key] = trainingDict[key] + testingDict[key] + validationDict[key]

    convertedPredictions = rescalePredictions(allPredictions, judgeName)

    # ensure that the number of predictions matches the number of scores
    assert len(convertedPredictions) == len(mlpData['scores']), f"Mismatch between number of predictions \
    ({len(convertedPredictions)}) and scores ({len(mlpData['scores'])})"

    # ensure that the number of testing predictions matches the number of testing scores
    assert len(convertedTestingPredictions) == len(testingDict['scores']), f"Mismatch between number of testing predictions \
    ({len(convertedTestingPredictions)}) and testing scores ({len(testingDict['scores'])})"

    return mlpData, convertedPredictions, convertedTestingPredictions