import numpy as np
from colorama import Fore
from itertools import combinations
import multiprocessing as mp
from functools import partial
import time
import pickle
import os

# gradient descent functions with new scoring equation
def sigmoidActivation(score):
    # clip extreme values to prevent overflow in exp()
    clippedScore = np.clip(score, -500, 500)
    return 2 / (1 + np.exp(-clippedScore)) - 1


def calculatePointScore(params, data, selectedParams, fighterIndex):
    """
    calculate point score for a single fighter using the weighted sum approach
    
    Args:
        params: weight parameters for each metric
        data: data dictionary containing performance metrics
        selectedParams: list of parameter names to include
        fighterIndex: index of the round (0-based)
    
    Returns:
        point score for the fighter
    """
    pointScore = 0
    for param, metric in zip(params, selectedParams):
        # use raw values for total and ratio stats, +/- for others
        if 'total' in metric or 'Share' in metric:
            pointScore += param * data[metric][fighterIndex]
        else:
            pointScore += param * data[metric][fighterIndex]
    
    return pointScore


def heuristic(params, data, selectedParams, dampener=50.0, sharpness=9.0):
    """
    robust heuristic function with comprehensive parameter flattening
    """
    
    def flattenParams(params):
        """
        recursively flatten any nested parameter structure
        """
        flatList = []
        
        for p in params:
            if isinstance(p, (list, tuple)):
                # recursively flatten lists/tuples
                flatList.extend(flattenParams(p))
            elif isinstance(p, np.ndarray):
                # flatten numpy arrays and convert each element
                if p.size == 1:
                    # single element array
                    flatList.append(float(p.item()))
                else:
                    # multi-element array - flatten and add all elements
                    flatList.extend([float(x) for x in p.flatten()])
            elif hasattr(p, '__len__') and not isinstance(p, str):
                # other sequence types
                flatList.extend(flattenParams(p))
            else:
                # scalar value
                try:
                    flatList.append(float(p))
                except (ValueError, TypeError):
                    print(f"WARNING: could not convert parameter to float: {p} (type: {type(p)})")
                    flatList.append(0.0)
        
        return flatList
    
    # debug output to see what we're working with
    # print(f"DEBUG: original params type: {type(params)}")
    # print(f"DEBUG: original params length: {len(params)}")
    
    # handle different input types
    if hasattr(params, 'tolist'):
        # convert numpy array to list first
        params = params.tolist()
    elif not isinstance(params, (list, tuple)):
        # single parameter - wrap in list
        params = [params]
    
    # flatten all parameters
    flatParams = flattenParams(params)
    
    # print(f"DEBUG: flattened params length: {len(flatParams)}")
    # print(f"DEBUG: selectedParams length: {len(selectedParams)}")
    # print(f"DEBUG: first few flat params: {flatParams[:5] if len(flatParams) > 5 else flatParams}")
    
    # check if dampener and sharpness are included as parameters
    if len(flatParams) == len(selectedParams) + 2:
        weights = flatParams[:-2]
        dampener = float(flatParams[-2])
        sharpness = float(flatParams[-1])
        # print(f"DEBUG: using optimized dampener: {dampener}, sharpness: {sharpness}")
    elif len(flatParams) == len(selectedParams):
        weights = flatParams
        # print(f"DEBUG: using default dampener: {dampener}, sharpness: {sharpness}")
    elif len(flatParams) == len(selectedParams) + 1:
        # handle case where we have 40 params instead of expected 39 or 41
        weights = flatParams[:len(selectedParams)]
        # print(f"DEBUG: got {len(flatParams)} params, using first {len(selectedParams)} as weights")
        # print(f"DEBUG: using default dampener: {dampener}, sharpness: {sharpness}")
    else:
        # try to use the first N parameters as weights where N = len(selectedParams)
        print(f"WARNING: unexpected parameter count: {len(flatParams)} (expected {len(selectedParams)} or {len(selectedParams)+2})")
        if len(flatParams) >= len(selectedParams):
            weights = flatParams[:len(selectedParams)]
            # print(f"DEBUG: using first {len(selectedParams)} params as weights, ignoring the rest")
            # print(f"DEBUG: using default dampener: {dampener}, sharpness: {sharpness}")
        else:
            raise ValueError(f"Not enough parameters: got {len(flatParams)}, need at least {len(selectedParams)}")
    
    # print(f"DEBUG: final weights length: {len(weights)}")
    
    predictions = []
    
    # process each round (every 2 entries represent one round)
    for i in range(0, len(data[selectedParams[0]]), 2):
        # calculate point scores for both fighters
        redPointScore = 0.0
        bluePointScore = 0.0
        
        for j, metric in enumerate(selectedParams):
            weight = weights[j]
            
            # red fighter (index i)
            if 'total' in metric or 'Share' in metric:
                redPointScore += weight * data[metric][i]
            else:
                redPointScore += weight * data[metric][i]
            
            # blue fighter (index i+1)
            if 'total' in metric or 'Share' in metric:
                bluePointScore += weight * data[metric][i+1]
            else:
                bluePointScore += weight * data[metric + 'O'][i]
        
        # ensure non-negative point scores
        minScore = min(redPointScore, bluePointScore)
        if minScore < 0:
            redPointScore -= minScore
            bluePointScore -= minScore
        
        # calculate ratio and apply scoring equation
        ratio = (redPointScore + dampener) / (bluePointScore + dampener)
        ratioToS = ratio ** sharpness
        redPrediction = ratioToS / (ratioToS + 1)
        redPrediction = 2 * redPrediction - 1
        
        predictions.append(redPrediction)
        predictions.append(-redPrediction)
    
    return np.array(predictions)

def costCalc(params, data, scores, selectedParams, meanBool, dampener=50.0, sharpness=9.0):
    """
    calculate cost using new heuristic function
    """
    predictions = heuristic(params, data, selectedParams, dampener, sharpness)

    if meanBool:
        cost = np.mean((predictions - np.array(scores)) ** 2)
    else:
        cost = np.median((predictions - np.array(scores)) ** 2)
    
    return cost


def gradientDescent(data, scores, selectedParams, learningRate=0.00001, iterations=20000, 
                   clipValue=0.5, momentum=0.9996, earlyStopThreshold=0.0, randomSeed=0,
                   includeMappingParams=False, dampener=150.0, sharpness=9.0,
                   dampenerLearningRate=0.1, sharpnessLearningRate=0.0005):

    # suppress print statements for parallel processing
    suppressPrint = mp.current_process().name != 'MainProcess'
    
    if not suppressPrint:
        print(f"Learning Rate: {learningRate}    Momentum: {momentum}")
        if includeMappingParams:
            print(f"Optimizing mapping parameters (D, S)")
            print(f"Dampener Learning Rate: {dampenerLearningRate}    Sharpness Learning Rate: {sharpnessLearningRate}")
        else:
            print(f"Fixed mapping parameters - D: {dampener}, S: {sharpness}")
        print()

    np.random.seed(randomSeed)
    
    # determine parameter count
    if includeMappingParams:
        paramSize = len(selectedParams) + 2  # weights + D + S
        params = np.random.uniform(-0.5, 0.5, size=len(selectedParams))
        params = np.append(params, [dampener, sharpness])  # add initial D and S values
    else:
        paramSize = len(selectedParams)
        params = np.random.uniform(-0.5, 0.5, size=paramSize)
    
    # initialize momentum for each parameter
    prevGradients = np.zeros_like(params)
    prevCost = float('inf')
    minCost2 = float('inf')
    bestParams = np.zeros_like(params)
    stagnationCount = 0
    
    for i in range(iterations):
        # calculate predictions and errors
        if includeMappingParams:
            predictions = heuristic(params, data, selectedParams)
        else:
            predictions = heuristic(params, data, selectedParams, dampener, sharpness)
        
        errors = predictions - np.array(scores)
        
        # calculate gradients for weights
        gradients = []
        for j, metric in enumerate(selectedParams):
            if 'total' in metric or 'Share' in metric:
                gradient = np.mean(errors * np.array(data[metric]))
            else:
                metricDiff = np.array(data[metric]) - np.array(data[metric + 'O'])
                gradient = np.mean(errors * metricDiff)
            gradients.append(gradient)
        
        # add gradients for mapping parameters if optimizing them
        if includeMappingParams:
            # use larger epsilon for mapping parameters to improve gradient calculation
            epsilonMapping = 1e-3
            currentCost = np.mean(errors ** 2)
            
            # gradient for dampener D
            tempParams = params.copy()
            tempParams[-2] += epsilonMapping
            tempPredictions = heuristic(tempParams, data, selectedParams)
            tempCost = np.mean((tempPredictions - np.array(scores)) ** 2)
            dGradient = (tempCost - currentCost) / epsilonMapping
            gradients.append(dGradient)
            
            # gradient for sharpness S
            tempParams = params.copy()
            tempParams[-1] += epsilonMapping
            tempPredictions = heuristic(tempParams, data, selectedParams)
            tempCost = np.mean((tempPredictions - np.array(scores)) ** 2)
            sGradient = (tempCost - currentCost) / epsilonMapping
            gradients.append(sGradient)
        
        gradients = np.array(gradients)
        
        # gradient clipping
        gradients = np.clip(gradients, -clipValue, clipValue)
        
        # create learning rate array with different rates for different parameter types
        if includeMappingParams:
            learningRates = np.ones_like(params) * learningRate
            learningRates[-2] = dampenerLearningRate  # dampener D
            learningRates[-1] = sharpnessLearningRate  # sharpness S
        else:
            learningRates = learningRate
        
        # update momentum with parameter-specific learning rates
        if includeMappingParams:
            prevGradients = momentum * prevGradients + learningRates * gradients
        else:
            prevGradients = momentum * prevGradients + learningRate * gradients
        
        # update parameters using momentum
        params -= prevGradients

        # clip parameters to reasonable bounds
        params[:len(selectedParams)] = np.clip(params[:len(selectedParams)], -40, 40)
        if includeMappingParams:
            params[-2] = np.clip(params[-2], 1.0, 1000.0)  # dampener bounds
            params[-1] = np.clip(params[-1], 0.1, 10.0)    # sharpness bounds
        
        if i % 10 == 0:
            if includeMappingParams:
                cost = costCalc(params, data, scores, selectedParams, True)
            else:
                cost = costCalc(params, data, scores, selectedParams, True, dampener, sharpness)
            
            if cost < minCost2:
                minCost2 = cost
                bestParams = params.copy()
                stagnationCount = 0
            else:
                stagnationCount += 1
        
        # early stopping logic
        if i % 1000 == 0 and not suppressPrint:
            if includeMappingParams:
                currentCost = costCalc(params, data, scores, selectedParams, True)
                print(f"Iteration {i}: Cost {currentCost:.6f}, D={params[-2]:.2f}, S={params[-1]:.2f}")
            else:
                currentCost = costCalc(params, data, scores, selectedParams, True, dampener, sharpness)
                print(f"Iteration {i}: Cost {currentCost:.6f}")
            
            if abs(prevCost - currentCost) < earlyStopThreshold and i > 0:
                if not suppressPrint:
                    print(f"Early stopping at iteration {i} due to cost convergence.")
                break
            prevCost = currentCost
        
        # extended stagnation check
        if stagnationCount > 5000:
            if not suppressPrint:
                print(f"Early stopping at iteration {i} due to extended stagnation.")
            break
    
    return bestParams, minCost2


def isValidCombination(combo):
    aggressionCancels = {'aggressionCombinations', 'aggressionExchanges', 'aggressionPower'}
    pressureCancels = {'pressureDistance', 'pressureMovement', 'pressurePosition'}
    highImpactCancels = {'high', 'max'}
    missedCancels = {'lowCommitMiss', 'highCommitMiss', 'minMissed', 'lowMissed', 'midMissed', 'highMissed', 'maxMissed'}
    aggressionCombinationsCancels = {'singles', 'doubles', 'triples', 'quadsPlus'}
    pressureDistanceCancels = {'outside', 'midrange', 'inside', 'clinch'}

    if 'aggression' in combo and aggressionCancels.intersection(combo):
        return False
    elif 'pressure' in combo and pressureCancels.intersection(combo):
        return False
    elif 'highImpact' in combo and highImpactCancels.intersection(combo):
        return False
    elif 'missed' in combo and missedCancels.intersection(combo):
        return False
    elif 'aggressionCombinations' in combo and aggressionCombinationsCancels.intersection(combo):
        return False
    elif 'pressureDistance' in combo and pressureDistanceCancels.intersection(combo):
        return False
    return True


def processCombo(args):
    """
    worker function for parallel processing of parameter combinations
    """
    combo, data, scores, comboIndex, totalCombos, dampener, sharpness = args
    
    # run gradient descent for this combination
    optimizedParams = gradientDescent(data, scores, list(combo), dampener=dampener, sharpness=sharpness)
    
    # calculate the cost for this optimized set
    cost = costCalc(optimizedParams, data, scores, list(combo), True, dampener, sharpness)
    
    return (cost, combo, optimizedParams, comboIndex, totalCombos)


def sortByCost(item):
    """
    helper function to sort by cost (replaces lambda for pickling)
    """
    return item[0]


def updateProgress(result, bestResults, lock, progressCounter):
    """
    callback function to update progress and maintain top results
    """
    cost, combo, optimizedParams, comboIndex, totalCombos = result
    
    with lock:
        # increment progress counter
        progressCounter.value += 1
        currentProgress = progressCounter.value
        
        # if we have less than 20 results, or if this result is better than the worst we've seen
        if len(bestResults) < 20 or cost < bestResults[-1][0]:
            bestResults.append((cost, combo, optimizedParams))
            bestResults.sort(key=sortByCost)
            if len(bestResults) > 20:
                bestResults.pop()
        
        # print progress and save checkpoint every 100 combinations
        if currentProgress % 100 == 0 or (len(bestResults) > 0 and cost <= bestResults[0][0]):
            percentage = (currentProgress / totalCombos) * 100
            if len(bestResults) > 0:
                print(f"Progress: {currentProgress}/{totalCombos} ({percentage:.1f}%) - Current best cost: {bestResults[0][0]:.6f}")
                
                # save checkpoint every 500 combinations
                if currentProgress % 500 == 0:
                    checkpointData = {
                        'bestResults': list(bestResults),
                        'progress': currentProgress,
                        'totalCombos': totalCombos
                    }
                    with open('combos_checkpoint.pkl', 'wb') as f:
                        pickle.dump(checkpointData, f)
                    print(f"Checkpoint saved at {currentProgress}/{totalCombos}")
            else:
                print(f"Progress: {currentProgress}/{totalCombos} ({percentage:.1f}%)")


def combos(data, parameters, paramCount, numProcesses=None, dampener=150.0, sharpness=9.0):
    """
    find the best combination of paramCount parameters using parallel processing
    """
    if numProcesses is None:
        numProcesses = mp.cpu_count() - 1
    
    print(f"Starting parallel gradient descent cycling with {paramCount} parameters...")
    print(f"Using {numProcesses} processes...")
    print(f"Mapping parameters - D: {dampener}, S: {sharpness}")
    
    scores = data['scores']
    
    # calculate the actual number of valid combinations
    validCombinations = [combo for combo in combinations(parameters, paramCount) if isValidCombination(combo)]
    totalCombinations = len(validCombinations)
    
    print(f"Total valid combinations to test: {totalCombinations}")
    
    # prepare arguments for parallel processing
    args = [(combo, data, scores, i+1, totalCombinations, dampener, sharpness) 
            for i, combo in enumerate(validCombinations)]
    
    # shared data structure for results
    manager = mp.Manager()
    bestResults = manager.list()
    lock = manager.Lock()
    progressCounter = manager.Value('i', 0)
    
    # create callback function with shared data
    updateCallback = partial(updateProgress, bestResults=bestResults, lock=lock, progressCounter=progressCounter)
    
    startTime = time.time()
    
    # run parallel processing
    with mp.Pool(processes=numProcesses) as pool:
        # submit all jobs
        jobs = []
        for arg in args:
            job = pool.apply_async(processCombo, (arg,), callback=updateCallback)
            jobs.append(job)
        
        # wait for all jobs to complete
        for job in jobs:
            job.wait()
    
    endTime = time.time()
    
    # convert manager list to regular list and sort
    finalResults = list(bestResults)
    finalResults.sort(key=sortByCost)
    
    print(f"\nCompleted in {endTime - startTime:.2f} seconds")
    print(f"\nTop 20 Parameter Combinations:")
    for i, (cost, combo, optimizedValues) in enumerate(finalResults[:20], 1):
        print(f"\n{i}. Parameters: {combo}")
        print(f"   Cost: {cost:.6f}")
        print("   Optimized values:")
        for param, value in zip(combo, optimizedValues):
            print(f"      {param}: {value:.7f}")
    
    return finalResults[:20]


def processComboStart(args):
    """
    worker function for parallel processing of parameter combinations with starting params
    """
    additionalCombo, startParams, data, scores, comboIndex, totalCombos, dampener, sharpness = args
    
    currentCombo = list(startParams) + list(additionalCombo)
    
    # run gradient descent for this combination
    optimizedParams = gradientDescent(data, scores, currentCombo, dampener=dampener, sharpness=sharpness)
    
    # calculate the cost for this optimized set
    cost = costCalc(optimizedParams, data, scores, currentCombo, True, dampener, sharpness)
    
    return (cost, currentCombo, optimizedParams, comboIndex, totalCombos)


def updateProgressStart(result, bestResults, lock, progressCounter):
    """
    callback function for comboStart to update progress and maintain top results
    """
    cost, combo, optimizedParams, comboIndex, totalCombos = result
    
    with lock:
        # increment progress counter
        progressCounter.value += 1
        currentProgress = progressCounter.value
        
        # if we have less than 20 results, or if this result is better than the worst we've seen
        if len(bestResults) < 20 or cost < bestResults[-1][0]:
            bestResults.append((cost, combo, optimizedParams))
            bestResults.sort(key=sortByCost)
            if len(bestResults) > 20:
                bestResults.pop()
        
        # print progress every 50 combinations or for the best result so far
        if currentProgress % 50 == 0 or (len(bestResults) > 0 and cost <= bestResults[0][0]):
            percentage = (currentProgress / totalCombos) * 100
            if len(bestResults) > 0:
                print(f"Progress: {currentProgress}/{totalCombos} ({percentage:.1f}%) - Current best cost: {bestResults[0][0]:.6f}")
            else:
                print(f"Progress: {currentProgress}/{totalCombos} ({percentage:.1f}%)")


def comboStart(data, startParams, allParams, paramCount, numProcesses=None, dampener=150.0, sharpness=9.0):
    """
    take a starting set of parameters and find best params to add to it using parallel processing
    """
    if numProcesses is None:
        numProcesses = mp.cpu_count() - 1
    
    print(f"Starting parallel gradient descent with {len(startParams)} set parameters and {paramCount} total parameters...")
    print(f"Using {numProcesses} processes...")
    print(f"Mapping parameters - D: {dampener}, S: {sharpness}")
    
    scores = data['scores']
    
    # remove startParams from allParams to avoid duplicates
    remainingParams = list(set(allParams) - set(startParams))
    
    # calculate the number of additional parameters needed
    additionalParamCount = paramCount - len(startParams)
    
    if additionalParamCount <= 0:
        print(f"{Fore.RED}ERROR: paramCount ({paramCount}) must be greater than the number of startParams ({len(startParams)}){Fore.WHITE}")
        return
    
    # calculate the actual number of valid combinations
    validCombinations = [combo for combo in combinations(remainingParams, additionalParamCount) 
                          if isValidCombination(list(startParams) + list(combo))]
    totalCombinations = len(validCombinations)
    
    print(f"Total valid combinations to test: {totalCombinations}")
    print(f"Starting parameters: {startParams}")
    
    # prepare arguments for parallel processing
    args = [(combo, startParams, data, scores, i+1, totalCombinations, dampener, sharpness) 
            for i, combo in enumerate(validCombinations)]
    
    # shared data structure for results
    manager = mp.Manager()
    bestResults = manager.list()
    lock = manager.Lock()
    progressCounter = manager.Value('i', 0)
    
    # create callback function with shared data
    updateCallback = partial(updateProgressStart, bestResults=bestResults, lock=lock, progressCounter=progressCounter)
    
    startTime = time.time()
    
    # run parallel processing
    with mp.Pool(processes=numProcesses) as pool:
        # submit all jobs
        jobs = []
        for arg in args:
            job = pool.apply_async(processComboStart, (arg,), callback=updateCallback)
            jobs.append(job)
        
        # wait for all jobs to complete
        for job in jobs:
            job.wait()
    
    endTime = time.time()
    
    # convert manager list to regular list and sort
    finalResults = list(bestResults)
    finalResults.sort(key=sortByCost)
    
    print(f"\nCompleted in {endTime - startTime:.2f} seconds")
    print(f"\nTop 20 Parameter Combinations:")
    for i, (cost, combo, optimizedValues) in enumerate(finalResults[:20], 1):
        print(f"\n{i}. Parameters: {combo}")
        print(f"   Cost: {cost:.6f}")
        print("   Optimized values:")
        for param, value in zip(combo, optimizedValues):
            print(f"      {param}: {value:.7f}")

    return finalResults[:20]