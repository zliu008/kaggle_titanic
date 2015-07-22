library(caret)
##read the training and testing data
train<-read.csv('train.csv', stringsAsFactor = FALSE)
test<-read.csv('test.csv', stringsAsFactor = FALSE)
trainLen <- dim(train)[1]
testLen <- dim(test)[1]

##combine the training and testing data into a full set
test$Survived <- rep(NA, testLen)
alldata<-rbind(train,test)
alldataLen <- trainLen + testLen

##fill in the missing values of Fare and Embarked
alldata$Fare[is.na(alldata$Fare)] = sum(alldata$Fare[!is.na(alldata$Fare)])/sum(!is.na(alldata$Fare))
alldata$Embarked[alldata$Embarked==''] <- 'S';

#process the ticket information
TicketNumArray <- sapply(alldata$Ticket, FUN=function(x) {strsplit(x, split=" ")})
TicketNum0 <- rep(NA, alldataLen) #the text part
TicketNum1 <- rep(NA, alldataLen) #the first 3 digits
TicketNum2 <- rep(NA, alldataLen) #all the digits
for( k in 1:alldataLen) {
    ind_loc <- length(TicketNumArray[[k]])
    end_pos <- nchar(TicketNumArray[[k]][ind_loc])
    if(end_pos >3) {
        end_pos <- 3
    }
    if( length(TicketNumArray[[k]]) > 1 ) {
         TicketNum0[k] <- TicketNumArray[[k]][1]
    } else {
         TicketNum0[k] <- 'Nil'
    }
    TicketNum1[k] <- as.numeric(substr(TicketNumArray[[k]][ind_loc], 1, end_pos)) 
    TicketNum2[k] <- (TicketNumArray[[k]][ind_loc]) 
    
}

alldata$TicketNum0 = as.factor(TicketNum0)
alldata$TicketNum1 = TicketNum1
alldata$TicketNum2 = TicketNum2

#process the Name and title information
NameTitle <- strsplit(as.character(alldata$Name), split='[,.]')
titlePerson <- rep(NA, alldataLen)

for (i in 1:alldataLen) {
    titlePerson[i] <- NameTitle[[i]][2]
}
#merge some 'small titles' to major stream titles, these small titles does not help in prediction 
titlePerson <- sub(' ','', titlePerson);
titlePerson[titlePerson %in% c('Mme', 'Mlle')] <- 'Ms'
titlePerson[titlePerson %in% c('Capt', 'Don', 'Major', 'Sir','Jonkheer')] <- 'Mr'
titlePerson[titlePerson %in% c('Dona', 'Lady', 'the Countess')] <- 'Ms'
alldata$titlePerson <- as.factor(titlePerson)

#convert the read-in strings to factor
alldata$Sex <- as.factor(alldata$Sex)
alldata$Embarked <- as.factor(alldata$Embarked)

#grouping and encoding Surnames into digital feature 
Surnames <- rep(NA, alldataLen)
for (i in 1:alldataLen) {
    Surnames[i] <- NameTitle[[i]][1]
} 
surnames_lookup <- levels(as.factor(Surnames))
surnames_lookup_code <- 1:length(levels(as.factor(Surnames)))
surname_codes <- rep(NA, alldataLen)
for (i in 1:alldataLen) {
     surname_codes[i] <- surnames_lookup_code[Surnames[i] == surnames_lookup]
} 
alldata$surname_codes <- surname_codes; 


## encode TicketNums as digital feature
##Note that for the text header of the ticket, both its encoded digital number and the original text are used
##to build different random forest
TicketNum0_lookup <- unique(TicketNum0)
TicketNum0_lookup_code <- 1:length(unique(TicketNum0))
TicketNum0_codes <- rep(NA, alldataLen)
for (i in 1:alldataLen) {
    TicketNum0_codes[i] <- TicketNum0_lookup_code[TicketNum0[i] == TicketNum0_lookup]
}
alldata$TicketNum0_codes <- TicketNum0_codes

TicketNum2_lookup <- unique(TicketNum2)
TicketNum2_lookup_code <- 1:length(unique(TicketNum2))
TicketNum2_codes <- rep(NA, alldataLen)
for (i in 1:alldataLen) {
    TicketNum2_codes[i] <- TicketNum2_lookup_code[TicketNum2[i] == TicketNum2_lookup]
}
alldata$TicketNum2_codes <- TicketNum2_codes

#define the age for kids
alldata$isKid <- (alldata$Age < 14)

##remove the ticket number of "LINE"
NATicketNum <- sum(is.na(alldata$TicketNum1))
alldata <- alldata[!is.na(alldata$TicketNum1),]
##prepaire the data for isKid identification
sub_alldata <- alldata[!is.na(alldata$isKid),]
#Age model to identify whether the passenger is a kid
set.seed(11000)
AgeModel <- train( as.factor(isKid) ~ Pclass + Sex + SibSp + Parch + Fare + titlePerson + TicketNum1, 
                    method = "rf", importance= TRUE, data = sub_alldata);

print("age model done!")
isKid_pred <- predict(AgeModel, alldata);
#for the data with missing ages, (alldata$Age < 14) will produce NA, so fill the NA with the predicted values
alldata$isKid[is.na(alldata$isKid)] <- as.logical(isKid_pred[is.na(alldata$isKid)])

#list out special cases for Kids and female in large family since this reduces their survive chance
#isMaleLargeFamily is not very useful
alldata$isFemaleLargeFamily <- (alldata$SibSp > 1 | alldata$Parch >1) & (alldata$Sex == 'female') &(!alldata$isKid)
alldata$isKidLargeFamily <- (alldata$SibSp > 1 | alldata$Parch >1) & (alldata$isKid)
print("family grouped")

#convert to factor 
alldata$titlePerson <- as.factor(alldata$titlePerson);
alldata$Embarked <- as.factor(alldata$Embarked);
alldata$Pclass <- as.factor(alldata$Pclass); 

#split back to train and test with engineered data
trainLen <- trainLen - NATicketNum; 
train <- alldata[1:trainLen,]
testStartInd <- trainLen + 1
alldataLen <- alldataLen - NATicketNum; 
test <- alldata[testStartInd:alldataLen,]

##cross check with surname_codes and TicketNum2_codes to make sure they are
##in both training set and testing set.  If they are not, then throw their code into a different code where the impurity
##level is very low, because of low predictive information for these codes. 
for (i in 1:trainLen) {
    if( sum(train$surname_codes[i] == test$surname_codes) == 0 ) {
	    train$surname_codes[i] <- -100; 
    }

    if( sum(train$TicketNum2_codes[i] == test$TicketNum2_codes) == 0) {
	    train$TicketNum2_codes[i] <- -1111;
    }
} 

for (i in 1:testLen) {
    if(sum(test$surname_codes[i] == train$surname_codes) == 0 ) {
	    test$surname_codes[i] <- -100; 
    }
    if( sum(test$TicketNum2_codes[i] == train$TicketNum2_codes) == 0) {
	    test$TicketNum2_codes[i] <- -1111;
    }
} 

set.seed(11000)
#train 3 different rf models, using different features
modSurvive1 <- train(as.factor(Survived) ~ Pclass + Fare+ surname_codes + TicketNum0 + Embarked + Sex + isKid + isKidLargeFamily + isFemaleLargeFamily, 
		     method = "rf", importance = TRUE, data = train, ntree = 12000)
modSurvive2 <- train(as.factor(Survived) ~ Pclass + Fare+ surname_codes + Embarked + Sex + isKid + isKidLargeFamily + isFemaleLargeFamily, 
		     method = "rf", importance = TRUE, data = train, ntree = 12000)
modSurvive3 <- train(as.factor(Survived) ~ Pclass + Fare+ surname_codes + TicketNum0_codes + TicketNum2_codes + Embarked + Sex + isKid + isKidLargeFamily + isFemaleLargeFamily, 
		     method = "rf", importance = TRUE, data = train, ntree = 12000)
 

print('finished training'); 
predictedSurvive1 <- as.numeric(predict(modSurvive1, test)) - 1; 
predictedSurvive2 <- as.numeric(predict(modSurvive2, test)) - 1; 
predictedSurvive3 <- as.numeric(predict(modSurvive3, test)) - 1; 
#voting of the 3 rf models
predictedSurvive <- (predictedSurvive1 + predictedSurvive2 + predictedSurvive3)/3

test$Survived <- round(predictedSurvive)
submit<- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "model_final2.csv", row.names = FALSE)
