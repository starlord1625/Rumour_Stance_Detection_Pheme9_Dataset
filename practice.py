import os
import codecs
import json

stance_path = "C:\\Users\\Harshavardhan\\IR-term-project\\paper-1\\Verified-Summarization-master_TreeLSTM\\all-rnr-annotated-threads\\stance.json"

stance_file = codecs.open(stance_path, 'r', 'utf-8')

stance = stance_file.read()
d = json.loads(stance)

stance_file.close()

switcher = {
    "comment" : [1],
    "deny"    : [2],
    "query"   : [3],
    "support" : [4]
}



print(switcher.get(d["9479743137051729934"],[0,0,0]))