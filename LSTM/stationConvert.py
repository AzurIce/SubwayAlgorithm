import csv

i = 0
with open('stationID.csv','r') as f:
    flag = []
    ids = []
    reader = csv.reader(f)
    for row in reader:
        i+=1
        if(i==1):
            continue
        flag.append(1)
        ids.append(int(row[0]))
    print(ids)
    i-=1
with open('new_traffic_data.csv','r') as f:
    reader = csv.reader(f)
    with open('stationConvert.csv','w',newline='') as w:
        writer = csv.writer(w)
        writer.writerow( ['id','Neighborhood Size','Structure'])
        count = 0
        j = 0
        for row in reader:
            j+=1
            if(j==1):
                continue
            if flag[int(row[1])]==1:
                flag[int(row[1])] = 0
                writer.writerow([int(row[1]),int(row[12]),int(row[3])])
                count+=1

