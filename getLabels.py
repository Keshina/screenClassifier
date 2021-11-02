import os,json,csv
from sklearn import metrics

# with open('predictionResult.json') as f:
#     jsonData = json.load(f)

labelList=set()
labelCounter ={}

# i = 0
def getLabels ():

    labelFolder = os.path.join(os.getcwd(),'correctLabels')
    labelMap = {}
    filePaths = []
    filePaths.append(os.path.join(labelFolder,'1-SignIn','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'2-SignUp','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'3-Category','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'4-Search','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'5-Terms','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'6-Account','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'7-Detail','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'8-Menu','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'9-About','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'10-Contact','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'11-Help','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'12-AddCart','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'13-RemoveCart','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'14-Address','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'15-Filter','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'16-AddBookmark','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'17-RemoveBookmark','LS-annotations.csv'))
    filePaths.append(os.path.join(labelFolder,'18-Textsize','LS-annotations.csv'))




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
    print("LABEL STATUS---------------")
    print(labelCount)

    return labelMap

labelMapping = getLabels()


def filterOutSmallCategories():
    filterList = []

    filterList.append("account_guest")

    filterList.append("pwd_assistant")
    filterList.append("username")
    filterList.append("signin_google")
    filterList.append("signin_password")
    filterList.append("menu_bookmark")
    filterList.append("signin_email")
    filterList.append("keep_signin")
    filterList.append("home_signin_or_signup")
    filterList.append("signin_amazon")
    filterList.append("signin_tosignin")
    filterList.append("back")
    filterList.append("menu_account")
    filterList.append("signin_fb")
    filterList.append("terms")
    filterList.append("interests")
    filterList.append("about")
    filterList.append("help")
    filterList.append("conact")
    filterList.append("alert")
    filterList.append("bookmark")
    filterList.append("checkout")
    filterList.append("confirm_remove")address_add
    filterList.append("address_add")
    filterList.append("state_spinner")
    filterList.append("continue")
    filterList.append("to_search")
    return filterList

    # problemImages = [allLabels.index(item) for item in filterList]
    # problemImages = [bels.index(item) for item in allLabels if not item]
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
