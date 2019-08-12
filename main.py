import pandas as pd
import numpy as np
import math

#pin: 8930655055542458
#submit at:
#http://www.cse.scu.edu/~yfang/coen169/submit.html

THRESHOLD_MODE = 0
KNN_MODE = 1

def main():


    train = pd.read_csv('C:/Users/phili/PycharmProjects/MovieRecommender/train.txt',sep='\t',header=None)
    train = train.values
    train = np.insert(train,0,0,axis=0)

    #resize training set to make movieId and userId align with indeces
    trainUserVectors = []
    for row in train:
        row = [0] + row.tolist()
        trainUserVectors.append(row)

    train = pd.DataFrame(data=trainUserVectors)

    test5 = pd.read_csv('C:/Users/phili/PycharmProjects/MovieRecommender/test5.txt',sep=' ',header=None)
    test10 = pd.read_csv('C:/Users/phili/PycharmProjects/MovieRecommender/test10.txt',sep=' ',header=None)
    test20 = pd.read_csv('C:/Users/phili/PycharmProjects/MovieRecommender/test20.txt',sep=' ',header=None)
    test5.columns = test10.columns = test20.columns = ['User','Movie','Rating']

    print("test5...")
    generateCosineSimilarityRatings(test5,train,"result5-cosineSimilarity3.txt",0.7,50,2,THRESHOLD_MODE)
    print("test10...")
    generateCosineSimilarityRatings(test10,train,"result10-cosineSimilarity3.txt",0.7,50,2,THRESHOLD_MODE)
    print("test20...")
    generateCosineSimilarityRatings(test20,train,"result20-cosineSimilarity3.txt",0.7,50,2,THRESHOLD_MODE)

    # print("test5...")
    # generatePearsonCorrelationRatings(test5,train,"result5-pearsonCorrelation.txt",1.5)
    # print("test10...")
    # generatePearsonCorrelationRatings(test10,train,"result10-pearsonCorrelation.txt",1.5)
    # print("test20...")
    # generatePearsonCorrelationRatings(test20,train,"result20-pearsonCorrelation.txt",1.5)

    # print("test5...")
    # generateItemBasedCFRatings(test5,train,"result5-itemBasedCF.txt",2.5)
    # print("test10...")
    # generateItemBasedCFRatings(test10,train,"result10-itemBasedCF.txt",2.5)
    # print("test20...")
    # generateItemBasedCFRatings(test20,train,"result20-itemBasedCF.txt",2.5)

def generateCosineSimilarityRatings(test,train,fileName,similarityThreshold,k,rho,mode):
    simCount = 0
    avgCount = 0
    lenCount = 0
    delIndeces = []             #keeps track of indeces in train to delete at the end
    readingNewVector = False            #readingNewVector used to keep track of when new user vector is being read
    for index, row in test.iterrows():
        if (row['Rating'] != 0):
            if (readingNewVector == False):
                tmpUserVector = np.zeros(len(train.columns))            #if at new user, initialize user vector to all 0s
                readingNewVector = True
            if (readingNewVector == True):
                tmpUserVector[row['Movie']] = row['Rating']  # set value of index(movieID) to user's rating of movieID
                delIndeces.append(index)
        else:
            readingNewVector = False
            similarities = getUserNeighbors(tmpUserVector, train, row['Movie'], similarityThreshold,k,rho,mode)        #retrieve neighborhood of user
            if (len(similarities) == 0):            #if no neighbors, get average movie rating. If no average movie rating, get average user rating.
                avgCount += 1
                rating = getAvgMovieRating(row['Movie'], train)
                if (rating == 0):
                    rating = getAvgUserRating(tmpUserVector)
            else:
                lenCount += len(similarities)
                rating = getUserNeighborhoodRating(row['Movie'], train, similarities)           #replace 0 with neighborhood rating
                simCount += 1
            test.at[index, 'Rating'] = int(round(rating))

    test = test.drop(test.index[delIndeces])            #drop indeces with ratings
    print("Similarity ratings: %f, Avg ratings: %f" % (simCount/(simCount + avgCount),avgCount/(simCount + avgCount)))
    print("Avg neighborhood length: %f" % (lenCount/simCount))
    test.to_csv(fileName, sep=' ', index=False, header=False)

#generates file with item based CF ratings
def generateItemBasedCFRatings(test,train,fileName,rho):
    delIndeces = []
    readingNewVector = False
    for index, row in test.iterrows():
        if (row['Rating'] != 0):
            if (readingNewVector == False):
                tmpMovieList = []           #keeps track of list of movieIds
                readingNewVector = True
            if (readingNewVector == True):
                tmpMovieList.append((row['Movie'],row['Rating']))           #add movieId to tmpMovieList
                delIndeces.append(index)
        else:
            readingNewVector = False
            rating = getItemBasedCFRating(row['Movie'],tmpMovieList,train,rho)      #get itemBased CF rating
            test.at[index, 'Rating'] = int(round(rating))

    test = test.drop(test.index[delIndeces])
    test.to_csv(fileName, sep=' ', index=False, header=False)

def generatePearsonCorrelationRatings(test,train,fileName,rho):
    delIndeces = []
    avgCount = 0
    pearsonCount = 0
    readingNewVector = False
    for index, row in test.iterrows():
        if (row['Rating'] != 0):
            if (readingNewVector == False):
                tmpUserVector = np.zeros(len(train.columns))
                readingNewVector = True
            if (readingNewVector == True):
                tmpUserVector[row['Movie']] = row['Rating']
                delIndeces.append(index)
        else:
            readingNewVector = False
            rating = getPearsonCorrelationRating(row['Movie'], tmpUserVector, train, getPearsonCorrelations(tmpUserVector,train,row['Movie'],rho))          #get pearson correlation rating
            if(math.isnan(rating)):         #if the rating is Nan because no users found with common movies rated, get average movie or user rating
                rating = getAvgMovieRating(row['Movie'], train)
                avgCount += 1
                if (isValid(rating) == False):
                    rating = getAvgUserRating(tmpUserVector)
                rating = int(round(rating))
            elif(round(rating) <= 0):           #otherwise if pearson rating is valid, round to nearest valid rating
                rating = 1
                pearsonCount += 1
            elif(round(rating) >= 6):
                rating = 5
                pearsonCount += 1
            else:
                rating = int(round(rating))
                pearsonCount += 1
            test.at[index, 'Rating'] = rating


    test = test.drop(test.index[delIndeces])
    print("Pearson ratings: %f, Avg ratings: %f" % (pearsonCount/(pearsonCount + avgCount),avgCount/(pearsonCount + avgCount)))

    test.to_csv(fileName, sep=' ', index=False, header=False)


#returns neighborhood based rating given neighborhood (a list of tuples (userId,similarity))
def getUserNeighborhoodRating(movieID,train,neighborhood):
    sumOfSimilarities = np.sum([x[1] for x in neighborhood])
    rating = 0
    for elt in neighborhood:
        rating += (elt[1]/sumOfSimilarities) * train.iloc[elt[0],movieID]           #accumulates a weighted average of user ratings based on similarity
    return rating

#returns pearson correlation rating given correlations (a list of tuples (userId,pearsonCoefficient))
def getPearsonCorrelationRating(movieID,userVector,train,correlations):
    sumOfCorrelations = np.sum([np.abs(x[1]) for x in correlations])
    offset = 0
    for elt in correlations:
        offset += (train.iloc[elt[0],movieID] - getAvgUserRating(train.iloc[elt[0],:])) * elt[1]        #gets weighted average of differnce b/w movie rating and avg user rating
    offset = offset/sumOfCorrelations
    return getAvgUserRating(userVector) + offset            #add weighted average offset to subject user's avg rating


#return item based collaborative filtering rating given movieRatingList (a list of tuples(movieId,rating))
def getItemBasedCFRating(movieId,movieRatingList,train,rho):
    activeMovie = train.iloc[:,movieId]         #get movie vector from movieId
    movieList = [train.iloc[:,id[0]] for id in movieRatingList]         #get list of movie vectors from list of movieIds
    similarities = getMovieSimilarities(activeMovie,movieList,rho)      #get list of similarities b/w active movie and movieList
    sumOfSimilarities = np.sum(similarities)
    if(sumOfSimilarities == 0):
        return np.mean([x[1] for x in movieRatingList])          #if movie had no ratings in common with other movies, return average of their ratings
    rating = 0
    for i in range(0,len(similarities)):
        rating += (similarities[i]/sumOfSimilarities) * movieRatingList[i][1]           #get weighted avg of movie ratings based on similarity
    return rating

# def getDimensionalRating

#returns list of tuple pairs of userid,similarity for k most similar users to user userVector that have rated movieID
def getUserNeighbors(userVector,train,movieId,similarityThreshold,k,rho,mode):
    similarities = []               #keeps track of similariites as list of tuples (userId,similarity)
    for i in range(1,len(train.index)):         #loop through all users
        if(train.iloc[i,movieId] != 0):         #make sure user has rated subject movie
            trainUserVector = train.iloc[i,:]
            if(mode == THRESHOLD_MODE):
                similarity = getSimilarity(userVector,trainUserVector,False)              #get similarity b/w subject user and user being visited
                if(similarity >= similarityThreshold):                 #add similarity to list if above threshold
                    similarity = similarity * math.pow(similarity, rho - 1)         #perform case amplification
                    similarities.append((i,similarity))
            else:
                similarity = getSimilarity(userVector,trainUserVector,True)
                if(similarity > 0):
                    similarity = similarity * math.pow(similarity, rho - 1)  # perform case amplification
                    similarities.append((i, similarity))
    if(mode == KNN_MODE):
        similarities.sort(key=lambda x: x[1])         #sort list of similariites in reverse order
        similarities.reverse()
    # print(similarities)
    if(mode == THRESHOLD_MODE):
        return similarities
    return similarities[0:k]



#gets list of tuples (userId,pearson correlation)
def getPearsonCorrelations(userVector,train,movieID,rho):
    correlations = []
    for i in range(1, len(train.index)):
        if (train.iloc[i,movieID] != 0):
            newUserVector, newTrainUserVector = getCommonEltsVectors(userVector, train.iloc[i,:])
            if (len(newUserVector) > 1):
                nuvAvg = getAvgUserRating([x[1] for x in newUserVector])              #get average ratings for users
                ntuvAvg = getAvgUserRating([x[1] for x in newTrainUserVector])
                newUserVector = [(x[1] - nuvAvg)*getIUF(train,x[0]) for x in newUserVector]           #new user vectors with IUF
                newTrainUserVector = [(x[1] - ntuvAvg)*getIUF(train,x[0]) for x in newTrainUserVector]
                # newUserVector = [(x[1] - nuvAvg) for x in newUserVector]              #new user vectors without IUF
                # newTrainUserVector = [(x[1] - ntuvAvg) for x in newTrainUserVector]
                # make sure denominator is != 0. Doesn't harm results since coefficient should = 0 for such case, which doesn't affect result
                if(getVectorMagnitude(newTrainUserVector) != 0 and getVectorMagnitude(newUserVector) != 0):
                    correlation = np.dot(newUserVector, newTrainUserVector) / (getVectorMagnitude(newUserVector) * getVectorMagnitude(newTrainUserVector))
                    correlation = correlation * math.pow(np.abs(correlation), rho - 1)          #case amplification
                    correlations.append((i,correlation))
    # print(len(correlations))
    return correlations

#return list of similarities b/w movie and list of movies, w/ case amplification
def getMovieSimilarities(movieVector,movieVectorList,rho):
    similarities = []
    for tmpVector in movieVectorList:
        similarity = getSimilarity(movieVector,tmpVector,False)
        similarity = similarity * math.pow(similarity, rho - 1)
        similarities.append(similarity)
    return similarities


#get cosine similarity b/w two vectors. uses only common elts.
def getSimilarity(vector1,vector2,withLengthModification):
    newVector1,newVector2 = getCommonEltsVectors(vector1,vector2)           #get new vectors w/ common elements
    newVector1 = [x[1] for x in newVector1]         #extract ratings
    newVector2 = [x[1] for x in newVector2]
    if(getVectorMagnitude(newVector1) != 0 and getVectorMagnitude(newVector2) != 0):            #make sure not empty vectors
        if(len(newVector1) == 1):           #if 1-D vectors, use this formula for similarity
            return 1 - (np.abs(newVector1[0] - newVector2[0]) / 20)
        similarity = np.dot(newVector1, newVector2) / (
                getVectorMagnitude(newVector1) * getVectorMagnitude(newVector2))                #otherwise use cosine similarity formula
        if(withLengthModification == True):
            similarity = similarity * math.log(len(newVector1),len(vector1))            #make similarity more/less similar based on if common vector is longer/shorter
        print(similarity)
        return similarity
    return 0


def getVectorMagnitude(vector):
    sum = 0
    for elt in vector:
        sum += math.pow(elt,2)
    return math.sqrt(sum)

#returns list of tuples (index,val) of common elts in vector1 and vector2
def getCommonEltsVectors(vector1,vector2):
    newVector1 = []
    newVector2 = []
    for i in range(1,len(vector1)):
        if(vector1[i] != 0 and vector2[i] != 0):
            newVector1.append((i,vector1[i]))
            newVector2.append((i,vector2[i]))
    return newVector1,newVector2

def getAvgMovieRating(movieID,train):
    sum = 0
    count = 0
    for i,elt in train.iterrows():
        if(elt[movieID] != 0):
            sum += elt[movieID]
            count = count + 1
    if(count == 0):
        return 0
    return sum/count

def getAvgUserRating(userVector):
    sum = 0
    count = 0
    for rating in userVector:
        if(rating != 0):
            sum += rating
            count += 1
    return sum/count

#returns if a rating is valid
def isValid(rating):
    if(math.isnan(rating) or rating <= 0.5 or rating >= 5.5):
        return False
    return True

#return inverse user frequency of a given movie
def getIUF(train,movieID):
    numRated = 0
    for i,row in train.iterrows():
        if(row[movieID] != 0):
            numRated += 1
    if(numRated != 0):
        return math.log((len(train.index)-1) / numRated, 200)
    return 1


if __name__ == '__main__':
    main()