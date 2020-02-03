new_data = open('new.txt', 'r')
new_name_data = open('names_new.txt', 'w')
name_data = open('names.txt', 'r')
new_temp_data = open('temp_new.txt', 'w')
temp_data = open('temp.txt', 'r')

names = []
for name in name_data:
    names.append(name.strip().lower())
for name in new_data:
    for seg in name.split():
        names.append(seg.strip().lower())
    names.append(name.strip().lower())

names = list(set(names))

for i in names:
    new_name_data.write(i+'\n')

tmp = []
for t in temp_data:
    if t not in names:
        tmp.append(t.strip().lower())

tmp = list(set(tmp))
for i in tmp:
    new_temp_data.write(i+'\n')

