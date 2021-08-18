import os,json,csv
from sklearn import metrics

with open('predictionResult.json') as f:
    jsonData = json.load(f)

labelList=set()

# i = 0
def getLabels ():

    labelFolder = os.path.join(os.getcwd(),'correctLabels')
    labelMap = {}
    filePaths = []
    filePaths.append(os.path.join(labelFolder,'1-SignIn','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'2-SignUp','LS-annotations.csv'))




    labelCount={}
    for filePath in filePaths:
        with open(filePath) as f:
            csvReader = csv.DictReader(f,delimiter=",")
            for counter,row in enumerate(csvReader):
                # print(row)
                imageName = row['screen'].strip().split("/")[-1]
                label = row['tag_screen']
                labelMap[imageName] = label
                labelList.add(label)
                if labelCount.get(label):
                    count = labelCount.get(label)
                    labelCount[label] = count+1
                else:
                    labelCount[label]=1

    print("UNIQUE LABELS WE ARE USING "+str(len(labelList)))

    print(str(len(labelMap))+" tagged number of screen category")

    return labelMap

# labelMapping = getLabels()
#
# realLabel =[]
# noTag=[]
# predLabel= []
#
# for query, result in jsonData.items():
#     getQueryName = query.split("/")[-1].split('.jpg')[0]+"-screen.jpg" # Need to change <path>/buzzfeed-signup-1-bbox-1808.jpg to buzzfeed-signup-1-bbox-1808-screen.jpg
#     getQueryName= getQueryName.replace('-long','')
#     resultName = result[0].split("/")[-1].split('.jpg')[0]+"-screen.jpg"
#     resultName =resultName.replace('-long','')
#     if labelMapping.get(getQueryName) and labelMapping.get(resultName):
#         realLabel.append(labelMapping[getQueryName])
#         predLabel.append(labelMapping[resultName])
#
#     else:
#         noTag.append(getQueryName)
#
# # print(noTag)
# print(metrics.accuracy_score(realLabel, predLabel))
