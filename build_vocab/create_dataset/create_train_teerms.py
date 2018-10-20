
import json
import tqdm
with open("test_terms.json") as json_file:
    json_data = json.load(json_file)

with open("train_sent.txt") as file :
	content = file.readlines()
text = " ".join(content)
text =text.lower()
with open("test_sent.txt") as file :
	content_test = file.readlines()
text_test = " ".join(content_test)
classes = ["fabric",'fashion_strings',"article_types","attribute_types"]
train_terms ={}
for i in classes :
	term_dict = json_data[i]
	temp_dict ={}
	for j in term_dict.keys():
		if (text.find(j)!=-1):
			temp_dict[j] = term_dict[j]
	train_terms[i]=temp_dict
test_terms ={}
unseen_terms ={}
for i in classes :
	term_dict = json_data[i]
	temp_dict ={}
	for j in term_dict.keys():
		if (text_test.find(j)!=-1):
			temp_dict[j] = term_dict[j]
	test_terms[i]=temp_dict
	test_keys = list(set(train_terms[i].keys())-set(test_terms[i].keys())&set(train_terms[i].keys()))
	temp_dict2={}
	for j in test_keys :
		temp_dict2[j] = term_dict[j]
	unseen_terms[i]=temp_dict2
# with open('train_terms_new2.json', 'w') as outfile:
#     json.dump(train_terms, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('test_terms_new2.json', 'w') as outfile:
    json.dump(unseen_terms, outfile, sort_keys=True, indent=4, separators=(',', ': '))




