import sys
import os
from xml_to_json_convertor import convert_to_json

count = 0

outputLocationonDevice= '/Users/kesina/Documents/screenClassifier' #change this for your device
# '/Users/kesina/Documents/AutoSeq-Android-Testing/uiClassification'

outputFolder =os.path.join(outputLocationonDevice,'classificationInput')
os.mkdir(outputFolder)
# inputFolder = sys.argv[1] #give REMAUI output here eg/ 1-SignIn_REMAUI_output
inputs =[]
scriptLocation = os.getcwd()
print(scriptLocation)
inputs.append('1-SignIn')
inputs.append('9-About')
inputs.append('10-Contact')
inputs.append('13-RemoveCart')
inputs.append('14-Address')
inputs.append('15-Filter')
inputs.append('16-AddBookmark')
inputs.append('17-RemoveBookmark')
inputs.append('18-Textsize')
inputs.append('2-SignUp')
inputs.append('3-Category')
inputs.append('4-Search')
inputs.append('5-Terms')
inputs.append('6-Account')
inputs.append('7-Detail')
inputs.append('8-Menu')
inputs.append('11-Help')
inputs.append('12-AddCart')

for inputFolder in inputs:
    outputLabelSubFolder = os.path.join(outputFolder,inputFolder)
    os.mkdir(outputLabelSubFolder)

    inputFolderPath = os.path.join(scriptLocation,"REMAUIOutput")
    inputFolderPath = os.path.join(inputFolderPath,inputFolder)
    appList = os.listdir(inputFolderPath)
    for app in appList:
        appPath = os.path.join(inputFolderPath,app)
        outputAppSubFolder= os.path.join(outputLabelSubFolder,app)
        os.mkdir(outputAppSubFolder)
        if '.DS_Store' not in appPath:
            screenList = os.listdir(appPath)
            for screen in screenList:
                screenPath = os.path.join(appPath,screen)
                output_path= os.path.join(outputAppSubFolder,screen)

                convert_to_json(screenPath, output_path)


    # for trace in os.listdir(dir_name):
    #     if os.path.isdir(os.path.join(dir_name,trace)):
    #         print("---------trace started------------")
    #         for screen in os.listdir(os.path.join(dir_name,trace)):
    #             count = count+1
    #             print(screen)
    #             output_path = trace+"_"+screen
    #             convert_to_json(os.path.join(dir_name,trace,screen), output_path)



        print(count)
