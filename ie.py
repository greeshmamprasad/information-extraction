import sys
import re

print ('Hi! This is an information extractor. Please enter your file like:\n')
cmdLineArgs = raw_input('infoextract input.txt\n')
cmdLineArgs = cmdLineArgs.split()

# def getId(article):
#     return id

try:
    with open(cmdLineArgs[1], 'r') as file:
        fileContent = file.read().strip()

        #split based on id pattern
        articles = list(filter(None, re.split(r"(((DEV-MUC3-)|(TST1-MUC3-)|(TST2-MUC4-))[0-9]{4})", fileContent)))
    file.close()

except BaseException as e:
    if str(e)!="":
        print('Unable to open the file OR file doesn\'t exist. \n Detailed exception is: \n'+str(e)+"\n")
        sys.exit()

#Create a dictionary of ids and articles
news_dictionary = {}
outputFile = open(cmdLineArgs[1] + ".template", "w")

for article in articles:
    id_pattern = re.compile("(((DEV-MUC3-)|(TST1-MUC3-)|(TST2-MUC4-))[0-9]{4})")
    if id_pattern.match(article):
      id = article
      continue
    half_id_pattern = re.compile("((DEV-MUC3-)|(TST1-MUC3-)|(TST2-MUC4-))")
    if half_id_pattern.match(article):
      continue
    news_dictionary[id] = article

    outputFile.write("ID: \t" + id + "\n")
outputFile.close()
   # incident = getIncident(article)
   # weapon = getWeapon(article)
   # perp_indiv = getPerpIndiv(article)
   # perp_org = getPerpOrg(article)
   # tartget = getTartget(article)
   # victim = getVictim(article)
   # print ("INCIDENT: \t"+id)
   # print ("WEAPON: \t"+id)
   # print ("PER INDIV: \t"+id)
   # print ("PERP ORG: \t"+id)
   # print ("TARGET: \t"+id)
   # print ("VICTIM: \t"+id)
