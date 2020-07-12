import glob, os
import sys

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = current_dir+"/" + sys.argv[1]+"/"


print("fdasfasfdasfasfasfdsfa::"+path_data)

# Percentage of images to be used for the test set
percentage_test = 1;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt

label_list = glob.glob(os.path.join(current_dir+"/"+ sys.argv[1], "*.txt"))
label_list = [  i.split('/')[-1].split('.')[0] for i in label_list ]
test_data_count  =int( (len(label_list) *percentage_test) /100.  )

test_list = label_list[0:test_data_count]
train_list = label_list[test_data_count:-1]


print(len(test_list))
print(len(train_list))

for pathAndFilename in glob.glob(os.path.join(current_dir+"/"+ sys.argv[1],"*.png")):
    
    print(pathAndFilename)
   
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    
    print(title)
    
    if title in test_list:
        
        file_test.write(pathAndFilename + "\n")
    elif title in train_list:
        file_train.write(pathAndFilename + "\n")


for pathAndFilename in glob.glob(os.path.join(current_dir+"/"+ sys.argv[1],"*.jpg")):
    
    print(pathAndFilename)
   
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    
    print(title)
    
    if title in test_list:
        
        file_test.write(pathAndFilename + "\n")
    elif title in train_list:
        file_train.write(pathAndFilename + "\n")

