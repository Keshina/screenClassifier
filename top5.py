import os, json

predictionfolder = "C:/Users/leony/Documents/UsageTesting/screenClassifier"


for file in os.listdir(predictionfolder):
    if file.startswith("predictionResult"):

        data = []
        total = 0
        with open(file, 'r') as data_file:
            data = json.load(data_file)


        correct_predictions = 0
        with open(file, 'w+') as data_file:
            json.dump(data, data_file, indent=4)
            
            for image in data:
                if (("time" in image) or ("score" in image) or ("badResultCounter" in image)):
                    continue

                elif image.startswith("num of imgs"):
                    total = data["num of imgs"]

                else:
                    for correctlabel in data[image]:
                        for predictedlabel in data[image][correctlabel]:
                            if predictedlabel == correctlabel:
                                correct_predictions += 1
                                break
                

        print(file, "TOP 5 ACCURACY:", (correct_predictions/total))

        
