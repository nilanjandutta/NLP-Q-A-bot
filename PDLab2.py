print("Please wait, while dependencies loading")
from Retrieval import Model as RM
from Query import Query as Q
import re
import sys

if len(sys.argv) == 1:
	print(">No data given")
	print("> Please rerun as")
	print("\t\t$ python task.py <FileName>")
	print("Bot> Save any information paragraph as .txt file for dataset input")
	print("Bot> See you again")
	exit()

datasetName = sys.argv[1]
try:
	datasetFile = open(datasetName,"r")
except FileNotFoundError:
	print("Bot> Dataset \"" + datasetName + "\" not found")
	exit()

paragraphs = []
for para in datasetFile.readlines():
	if(len(para.strip()) > 0):
		paragraphs.append(para.strip())

drm = RM(paragraphs,True,True)

print("Bot> Hey there! I am ready for your queries")
print("Bot> Say bye to stop")


greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)

isActive = True
while isActive:
	userQuery = input("You> ")
	if(not len(userQuery)>0):
		print("Bot> You need to ask something")

	elif greetPattern.findall(userQuery):
		response = "Hello!"
	elif userQuery.strip().lower() == "bye":
		response = "Bye Bye!"
		isActive = False
	else:
		pq = Q(userQuery,True,False,True)

		response =drm.query(pq)
	print("Bot>",response)

p=input("You are out")
