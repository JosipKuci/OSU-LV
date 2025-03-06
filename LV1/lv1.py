#Zadatak 1.4.1
def total_euro():
    sati=float(input("Unesite radne sate: "))
    satnica=float(input("Unesite satnicu: "))
    ukupno=satnica*sati
    print(f"Radni sati: {sati}h \neura/h: {satnica} \nUkupno:{ukupno} eura")

#Zadatak 1.4.2
def ocjena():
    try:
        ocjena=float(input("Unesite ocjenu izmedu 0 i 1: "))
    except:
        print("Kriva ocjena")
    if(ocjena >= 0.9):
        print("A")
    elif(ocjena >= 0.8):
        print("B")
    elif(ocjena >= 0.7):
        print("C")
    elif(ocjena >= 0.6):
        print("D")
    elif(ocjena < 0.6 and ocjena >= 0):
        print("F")
    else:
        print("Ocjena nije u rasponu!")


#Zadatak 1.4.3
def infinite_loop():
    lista=[]
    sum=0
    unos=input("Unesite broj ili upisite Done: ")
    while(unos != "Done"):
        if(unos.isnumeric()):
            lista.append(int(unos))
            sum+=int(unos)
        unos=input("Unesite broj ili upisite Done: ")
    print(lista)
    print(f"Max u listi: {max(lista)}")
    print(f"Min u listi: {min(lista)}")
    print(f"Srednja vrijednost: {sum/len(lista)}")


#Zadatak 1.4.4
def lyrics():
    counter = {}
    file = open("song.txt","r")
    for line in file:
        for word in line.split():
            if word not in counter:
                counter[word]=1
            else:
                counter[word]+=1
    file.close()
    print("Rijeci koje se samo jednom pojavljuju: ")
    for key in counter.keys():
        if counter[key]==1:
            print(key)

#Zadatak 1.4.5
def spam_collection():
    file = open("SMSSpamCollection.txt", "r")
    ham_sum=0
    spam_sum=0
    ham_messages=0
    spam_messages=0
    num_of_exclamations=0
    for line in file:
        current_line_elements=line.split()
        if current_line_elements[0]=="ham":
            ham_sum+=len(current_line_elements)
            ham_messages+=1
        elif current_line_elements[0]=="spam":
            spam_sum+=len(current_line_elements)
            spam_messages+=1
            if(current_line_elements[-1][-1]=='!'):
                num_of_exclamations+=1
    print(f"Average spam message word count: {spam_sum/spam_messages}")
    print(f"Average ham message word count: {ham_sum/ham_messages}")
    print(f"Number of exclamation marks at ends of messages: {num_of_exclamations}")



