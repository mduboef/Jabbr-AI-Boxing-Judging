import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from colorama import Fore
import matplotlib.pyplot as plt
import seaborn as sns

# data fed into l1 log reg needs to be binary, so there are 3 points per round
# data is all plus/minus data (except for total and share stats)
# data is also normalized so its all on the same scale
# ? is the total and share stats on a different scale even if they are normalized?
# ? is there an issues in comaring importance of percentage based stats to raw counts or is this handled by normalization?
def prepareDataForL1Logistic(data, parameters):
    """
    prepare data for L1 logistic regression by expanding to individual judge decisions
    and calculating plus/minus features
    
    Args:
        data: data dictionary with round-level stats
        parameters: list of parameter names to include as features
    
    Returns:
        featureMatrix: normalized feature matrix (n_samples x n_features)
        binaryOutcomes: binary judge decisions (0=loss, 1=win)
        featureNames: list of feature names
        judgeNames: list of judge names for each decision
        roundInfo: list of (fighter_name, opponent_name, round_number) for each decision
    """
    
    featureMatrix = []
    binaryOutcomes = []
    judgeNames = []
    roundInfo = []
    
    # iterate through each round (step by 2 since we have fighter and opponent perspectives)
    for i in range(0, len(data['scores']), 2):
        
        # calculate plus/minus features for this round
        roundFeatures = []
        for param in parameters:
            if 'total' in param or 'Share' in param:
                # use raw values for total and share stats
                roundFeatures.append(data[param][i])
            else:
                # use plus/minus differential
                plusMinus = data[param][i] - data[param + 'O'][i]
                roundFeatures.append(plusMinus)
        
        # expand to individual judge decisions
        for judgeNum in range(1, 4):
            judgeScore = data[f'score{judgeNum}'][i]
            judgeName = data[f'judge{judgeNum}'][i]
            
            # convert judge score to binary (1 for win, 0 for loss)
            binaryOutcome = 1 if judgeScore > 0 else 0
            
            featureMatrix.append(roundFeatures.copy())
            binaryOutcomes.append(binaryOutcome)
            judgeNames.append(judgeName)
            roundInfo.append((data['name'][i], data['nameO'][i], data['round'][i]))
    
    # convert to numpy arrays
    featureMatrix = np.array(featureMatrix)
    binaryOutcomes = np.array(binaryOutcomes)
    
    # normalize features
    scaler = StandardScaler()
    featureMatrix = scaler.fit_transform(featureMatrix)
    
    print(f"prepared data: {len(binaryOutcomes)} judge decisions across {len(binaryOutcomes)//3} rounds")
    print(f"features: {len(parameters)} parameters")
    print(f"class balance: {np.mean(binaryOutcomes):.3f} win rate")
    
    return featureMatrix, binaryOutcomes, parameters, judgeNames, roundInfo, scaler


def runL1LogisticRegression(featureMatrix, binaryOutcomes, featureNames, 
                           cValue=1.0, useCrossValidation=True, testSize=0.2, randomState=42):
    """
    run L1 logistic regression with optional cross-validation for regularization parameter selection
    
    Args:
        featureMatrix: normalized feature matrix
        binaryOutcomes: binary outcomes (0/1)
        featureNames: list of feature names
        cValue: regularization parameter (inverse of regularization strength)
        useCrossValidation: whether to use CV to select optimal C
        testSize: proportion of data to hold out for testing
        randomState: random seed for reproducibility
    
    Returns:
        model: fitted LogisticRegression model
        results: dictionary with performance metrics and feature importance
    """
    
    np.random.seed(randomState)
    
    # split data into train/test
    nSamples = len(binaryOutcomes)
    nTest = int(nSamples * testSize)
    indices = np.random.permutation(nSamples)
    
    testIndices = indices[:nTest]
    trainIndices = indices[nTest:]
    
    xTrain = featureMatrix[trainIndices]
    yTrain = binaryOutcomes[trainIndices]
    xTest = featureMatrix[testIndices]
    yTest = binaryOutcomes[testIndices]
    
    print(f"\ntraining set: {len(yTrain)} samples")
    print(f"test set: {len(yTest)} samples")
    
    # cross-validation for optimal C parameter
    if useCrossValidation:
        print(f"\n{Fore.YELLOW}performing cross-validation to select optimal C parameter...{Fore.WHITE}")
        
        # define parameter grid (focused on stronger regularization for feature selection)
        cValues = np.logspace(-6, -2, 20)
        
        # use stratified k-fold to maintain class balance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
        
        # grid search with cross-validation
        gridSearch = GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', random_state=randomState, max_iter=1000),
            param_grid={'C': cValues},
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        gridSearch.fit(xTrain, yTrain)
        bestC = gridSearch.best_params_['C']
        bestCvScore = gridSearch.best_score_
        
        print(f"optimal C: {bestC:.4f}")
        print(f"cross-validation accuracy: {bestCvScore:.3f}")
        
        cValue = bestC
    
    # fit final model with optimal (or provided) C
    model = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=cValue, 
        random_state=randomState,
        max_iter=1000
    )
    
    model.fit(xTrain, yTrain)
    
    # make predictions
    yTrainPred = model.predict(xTrain)
    yTestPred = model.predict(xTest)
    yTrainProba = model.predict_proba(xTrain)[:, 1]
    yTestProba = model.predict_proba(xTest)[:, 1]
    
    # calculate performance metrics
    trainAccuracy = accuracy_score(yTrain, yTrainPred)
    testAccuracy = accuracy_score(yTest, yTestPred)
    trainAuc = roc_auc_score(yTrain, yTrainProba)
    testAuc = roc_auc_score(yTest, yTestProba)
    
    # feature importance (coefficients) - keep ALL features, don't sort by absolute value yet
    coefficients = model.coef_[0]
    featureImportance = pd.DataFrame({
        'feature': featureNames,
        'coefficient': coefficients,
        'absCoefficient': np.abs(coefficients)
    })
    
    # count non-zero coefficients (selected features)
    nSelectedFeatures = np.sum(coefficients != 0)
    
    print(f"\n{Fore.GREEN}L1 Logistic Regression Results{Fore.WHITE}")
    print("=" * 50)
    print(f"regularization parameter (C): {cValue:.4f}")
    print(f"selected features: {nSelectedFeatures}/{len(featureNames)}")
    print(f"training accuracy: {trainAccuracy:.3f}")
    print(f"test accuracy: {testAccuracy:.3f}")
    print(f"training AUC: {trainAuc:.3f}")
    print(f"test AUC: {testAuc:.3f}")
    
    # print ALL selected features instead of just top 10
    selectedFeatures = featureImportance[featureImportance['coefficient'] != 0].sort_values('absCoefficient', ascending=False)
    print(f"\n{Fore.CYAN}All {len(selectedFeatures)} Selected Features:{Fore.WHITE}")
    print("-" * 50)
    for i, row in selectedFeatures.iterrows():
        print(f"{row['feature']:<25}: {row['coefficient']:>8.4f}")
    
    # compile results
    results = {
        'model': model,
        'featureImportance': featureImportance,  # contains ALL features
        'trainAccuracy': trainAccuracy,
        'testAccuracy': testAccuracy,
        'trainAuc': trainAuc,
        'testAuc': testAuc,
        'nSelectedFeatures': nSelectedFeatures,
        'cValue': cValue,
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest,
        'yTestPred': yTestPred,
        'yTestProba': yTestProba
    }
    
    return model, results


def plotFeatureImportance(results, topN=20, saveFile=None):
    """
    plot feature importance from L1 logistic regression - shows all features including unselected ones
    """
    
    featureImportance = results['featureImportance']
    
    # sort by absolute coefficient value to get most important features at top
    sortedFeatures = featureImportance.sort_values('absCoefficient', ascending=False)
    
    # take top N features (including those with zero coefficients)
    topFeatures = sortedFeatures.head(topN)
    
    if len(topFeatures) == 0:
        print("no features to plot")
        return
    
    plt.figure(figsize=(10, max(6, len(topFeatures) * 0.3)))
    
    # create color map for bars (positive = green, negative = red, zero = gray)
    barColors = []
    for coef in topFeatures['coefficient']:
        if coef > 0:
            barColors.append('green')
        elif coef < 0:
            barColors.append('red') 
        else:
            barColors.append('lightgray')  # zero coefficients in light gray
    
    # create the bar chart
    plt.barh(range(len(topFeatures)), topFeatures['coefficient'], color=barColors, alpha=0.7)
    
    # create y-tick labels
    plt.yticks(range(len(topFeatures)), topFeatures['feature'])
    
    # apply colors to y-tick labels (red for unselected features with coefficient = 0)
    ax = plt.gca()
    for i, (tickLabel, coef) in enumerate(zip(ax.get_yticklabels(), topFeatures['coefficient'])):
        if coef == 0:
            tickLabel.set_color('red')  # unselected features in red
        else:
            tickLabel.set_color('black')  # selected features in black
    
    plt.xlabel('Coefficient Value')
    plt.title(f'L1 Logistic Regression Feature Importance\n({results["nSelectedFeatures"]} of {len(topFeatures)} features selected, unselected features shown in red)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # invert y-axis to show most important at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if saveFile:
        plt.savefig(saveFile, dpi=300, bbox_inches='tight')
        print(f"feature importance plot saved to {saveFile}")
    
    plt.show()


def compareL1ToOtherMethods(data, parameters, gdPredictions=None):
    """
    compare L1 logistic regression results to gradient descent and correlation analysis
    """
    
    print(f"\n{Fore.MAGENTA}Comparing L1 Logistic Regression to Other Methods{Fore.WHITE}")
    print("=" * 60)
    
    # prepare data and run L1 logistic regression
    featureMatrix, binaryOutcomes, featureNames, judgeNames, roundInfo, scaler = prepareDataForL1Logistic(data, parameters)
    model, l1Results = runL1LogisticRegression(featureMatrix, binaryOutcomes, featureNames)
    
    # get top L1 features
    selectedFeatures = l1Results['featureImportance'][l1Results['featureImportance']['coefficient'] != 0]
    l1TopFeatures = selectedFeatures['feature'].tolist()
    
    print(f"\n{Fore.CYAN}All {len(selectedFeatures)} L1 Selected Features:{Fore.WHITE}")
    for i, (index, row) in enumerate(selectedFeatures.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25}: {row['coefficient']:>8.4f}")
    
    # if gradient descent predictions provided, compare
    if gdPredictions is not None:
        print(f"\n{Fore.YELLOW}note: for full comparison with gradient descent weights,")
        print(f"you'll need to manually compare the L1 selected features")
        print(f"with your gradient descent parameter weights{Fore.WHITE}")
    
    return model, l1Results


def runL1Analysis(data, parameters):
    """
    main function to run L1 logistic regression analysis
    """
    
    print(f"{Fore.BLUE}Starting L1 Logistic Regression Analysis{Fore.WHITE}")
    print("=" * 50)
    
    # prepare data
    featureMatrix, binaryOutcomes, featureNames, judgeNames, roundInfo, scaler = prepareDataForL1Logistic(data, parameters)
    
    # run L1 logistic regression with cross-validation
    model, results = runL1LogisticRegression(
        featureMatrix, 
        binaryOutcomes, 
        featureNames, 
        useCrossValidation=True
    )
    
    # plot feature importance for all features (up to the total number of parameters)
    # this will show both selected and unselected features
    plotFeatureImportance(results, topN=len(parameters), saveFile='l1_feature_importance.png')
    
    return model, results, scaler