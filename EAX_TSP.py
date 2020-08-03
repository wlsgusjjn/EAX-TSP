from random import *
from tkinter import *

totalCities = -1

cities = []
order = []
edges_origin = []
edges_A = []
edges_B = []
edges_M = []
edge_sub = []

bestOrderA = []
bestCostA = 0.0
bestOrderB = []
bestCostB = 0.0

greedy_list = []

pre_cost = []
next_cost = []

esets = []
inter_edges = []

Eset = []
intermediate = []
next_generation = []

def set():    
    canvas.delete("all")
    edges_origin.clear()
    edges_A.clear()
    edges_B.clear()
    edges_M.clear()
    edge_sub.clear()

    bestOrderA.clear()
    bestCostA = 0.0
    bestOrderB.clear()
    bestCostB = 0.0

    pre_cost.clear()
    next_cost.clear()

    esets.clear()
    inter_edges.clear()

    Eset.clear()
    intermediate.clear()
    next_generation.clear()    

    cPa['text'] = ""
    cPb['text'] = ""

    cO1['text'] = ""
    cO2['text'] = ""
    cO3['text'] = ""

def add(arr,e):
    for i in range(0,len(e)-1,2):
        a = e[i]
        b = e[i+1]
        c = -1
        for j in range(len(arr)-1):
            if (a == arr[j] and b == arr[j+1]) or (a == arr[j+1] and b == arr[j]):
                c = 1
        if c == -1:
            return False
    return True

def reset(flg):
    global totalCities,times,Eset,greedy_list


    if flg == 0:
        totalCities = int(Ncities.get())

        cities.clear()
        order.clear()

    set()

    if totalCities < 6:
        if totalCities < 3:
            return

        if flg == 0:
            createCities(totalCities)
            greedy_list = [[] for _ in range(totalCities)]
            greedy(order)
        keepCities()
        min = float('inf')
        x = -1
        val = -1
        for i in range(len(greedy_list)):
            if len(greedy_list[i]) > 0:
                val = calcost(greedy_list[i])
                if min > val:
                    min = val
                    x = i
                
        drawPath([greedy_list[x]])
        cO1['text'] = min

        return

    if flg == 0:
        createCities(totalCities)
        greedy_list = [[] for _ in range(totalCities)]
        
    greedy(order)
    mergePath(bestOrderA,bestOrderB)
    while len(Eset) == 0:
        run()
        
    for i in range(len(Eset)):
        intermediate.append(inter_sol(bestOrderA,Eset[i]))

    temp = []
    for i in range(len(Eset)):
        temp = new_connection(inter_sol(bestOrderA,Eset[i]))
        next_generation.append(temp.copy())
        esets.append([Eset[i].copy()])
        
        for j in range(len(next_generation)):
            if add(next_generation[j],Eset[i]):
                next_generation[j] = new_connection(inter_sol(next_generation[j],Eset[i]))
                esets[j].append(Eset[i].copy())

    for i in range(len(next_generation)):
        next_cost.append(calcost(next_generation[i]))


    if len(next_cost) > 3:
        min = float('inf')
        max = float('-inf')
        a = -1
        b = -1
        
        for i in range(len(next_cost)):
            if min > next_cost[i]:
                min = next_cost[i]
                a = i
            if i < 3 and max < next_cost[i]:
                max = next_cost[i]
                b = i

        if min > bestCostA or min > bestCostB:
            reset(1)
        else:
            c = intermediate[a].copy()
            intermediate[a] = intermediate[b].copy()
            intermediate[b] = c

            d = next_generation[a].copy()
            next_generation[a] = next_generation[b].copy()
            next_generation[b] = d

            e = next_cost[a]
            next_cost[a] = next_cost[b]
            next_cost[b] = e

            f = esets[a].copy()
            esets[a] = esets[b].copy()
            esets[b] = f
        

def run():
    global times,Eset,gre_candi
    
    set()
    
    keepCities()
    greedy(order)
    mergePath(bestOrderA,bestOrderB)
    Eset = mergeCycles(bestOrderA,bestOrderB)

def calcost(arr):
    sum = 0
    for i in range(len(order) - 1):
        x = cities[arr[i]][0] - cities[arr[i+1]][0]
        y = cities[arr[i]][1] - cities[arr[i+1]][1]
        sum += (x**2 + y**2)**0.5
    return sum

def calDistance(a,b):
    sum = 0
    x = cities[a][0] - cities[b][0]
    y = cities[a][1] - cities[b][1]
    sum += (x**2 + y**2)**0.5
    return sum

def greedy(arr):
    global bestOrderA, bestOrderB, bestCostA, bestCostB, greedy_list
    
    n = len(arr)
    candi = []
    tmp = []
    
    a = randrange(0,n-1)
    b = randrange(0,n-1)

    randCh = [a,b]
    for i in range(2):
        if len(greedy_list[randCh[i]]) == n:
            continue
        
        chk = [0 for _ in range(n-1)]
        chk[randCh[i]] = 1
        candi.append(arr[randCh[i]])

        while(1):
            min = float('inf')
            nextv = -1
            for j in range(n-1):
                if candi[len(candi)-1] == arr[j]:
                    continue

                if chk[j] == 0:
                    temp = calDistance(candi[len(candi)-1],arr[j])
                    if min > temp:
                        min = temp
                        nextv = j
                    
            candi.append(arr[nextv])
            chk[nextv] = 1

            k = 0
            while(k < n-1):
                if chk[k] == 0:
                    break
                k += 1

            if k == n-1:
                break

        k = 0
        for j in range(len(candi)):
            if candi[j] == 0:
                k = j

        for j in range(len(candi)):
            tmp.append(candi[(j+k)%(len(candi))])

        tmp.append(0)

        greedy_list[randCh[i]] = tmp.copy()

        tmp.clear()
        candi.clear()

    bestOrderA = greedy_list[a].copy()
    bestCostA = calcost(bestOrderA)
    bestOrderB = greedy_list[b].copy()
    bestCostB = calcost(bestOrderB)

    drawTwoPath(bestOrderA,bestOrderB)
    pre_cost.append(bestCostA)
    pre_cost.append(bestCostB)

    cPa['text'] = pre_cost[0]
    cPb['text'] = pre_cost[1]

def mergeCycles(A,B):
    global totalCities
    
    n = len(A)
    AnB = A + B
    x = -1
    ABcyc = []
    candi = []
    for i in range(n - 1):
        ABcyc.clear()
        prsnt = A[i]
        nextv = A[i + 1]
        indi = 1
        
        while(nextv != A[i]):
            ABcyc.append(prsnt)
            temp = -1

            if indi == 1:
                x = n
            elif indi == -1:
                x = 0
                
            for j in range(n-1):
                if AnB[j+x] == nextv:
                    temp = j
                    break

            if random() < 0.5:
                if prsnt == AnB[(temp+n-2)%(n-1)+x]:
                    prsnt = nextv
                    nextv = AnB[(temp+1)%(n-1)+x]
                else:
                    prsnt = nextv
                    nextv = AnB[(temp+n-2)%(n-1)+x]
            else:
                if prsnt == AnB[(temp+1)%(n-1)+x]:
                    prsnt = nextv
                    nextv = AnB[(temp+n-2)%(n-1)+x]
                else:
                    prsnt = nextv
                    nextv = AnB[(temp+1)%(n-1)+x]
                    
            indi *= -1
                    
        ABcyc.append(prsnt)
        ABcyc.append(nextv)
        if len(ABcyc) % 2 != 0 and len(ABcyc) < totalCities and len(ABcyc) > 3 and cycle(ABcyc):
            if not(distinct(candi,ABcyc)):
                candi.append(ABcyc.copy())

    return candi

def cycle(ABcyc):
    for i in range(1,len(ABcyc)-1):
        for j in range(i):
            if ABcyc[i] == ABcyc[j]:
                return False
    return True

def distinct(candi,ABcyc):
    n = len(ABcyc)
    for i in range(len(candi)):
        if n == len(candi[i]):
            temp = -1
            for j in range(n-1):
                if ABcyc[0] == candi[i][j]:
                    temp = j
                    break
            for j in range(n-1):
                if ABcyc[j] != candi[i][(temp+j)%(n-1)]:
                    temp = -1
                    break
            if temp != -1:
                return True
    return False

def sameEdge(used,arr):
    n = len(arr)
    if len(used) == 0:
        return False
    
    for i in range(n-1):
        for j in range(len(used)):
            for k in range(len(used[j])-1):
                if arr[i] == used[j][k] and arr[i+1] == used[j][k+1]:
                    return True
                if arr[i+1] == used[j][k] and arr[i] == used[j][k+1]:
                    return True
    return False

def inter_conn(sub,li,chk,flg):
    j = 0
    while(j < len(li)):
        temp = -1
        a = -1
        b = -1
        if flg == 1 or flg == 2:
            temp = len(sub)-1
            a = 0
            b = 1
        elif flg == -1 or flg == -2:
            temp = 0
            a = 1
            b = 0

        if sub[temp] == li[j][a]:
            if chk[j] == 0 or chk[j] == -1:
                if flg == 1 or flg == 2:
                    sub.append(li[j][b])
                elif flg == -1 or flg == -2:
                    sub.insert(0,li[j][b])
                chk[j] = 1
                j = 0
            else:
                j += 1
                            
        elif sub[temp] == li[j][b]:
            if flg == 2 or flg == -2:
                if chk[j] == 0 or chk[j] == -1:
                    if flg == 2:
                        sub.append(li[j][a])
                    elif flg == -2:
                        sub.insert(0,li[j][a])
                    chk[j] = 1
                    j = 0
                else:
                    j += 1                    
            else:
                if chk[j] == -1:
                    if flg == 1:
                        sub.append(li[j][a])
                    elif flg == -1:
                        sub.insert(0,li[j][a])
                    chk[j] = 1
                    j = 0
                else:
                    j += 1
                            
        else:
            j += 1

    return sub, li, chk
    

def inter_sol(A,E):
    global inter_edges
    inter = []
    sub = []
    edges = []
    Y = A.copy()
    n = len(Y)

    for i in range(n-1):
        tmp = [A[i],A[i+1]]
        edges.append(tmp.copy())
        tmp.clear()

    inter_edges = edges.copy()
            
    li = edges.copy()
    for j in range(0,len(E)-1,2):
        for k in range(len(li)):
            if E[j] == li[k][0]:
                if E[j+1] == li[k][1]:
                    li.remove([li[k][0],li[k][1]])
                    break
            if E[j+1] == li[k][0]:
                if E[j] == li[k][1]:
                    li.remove([li[k][0],li[k][1]])
                    break

    chk = [0 for _ in range(len(li))]
                        
    for j in range(1,len(E),2):
        edgeB = [E[j],E[j+1]]
        li.append(edgeB.copy())
        chk.append(-1)
        edgeB.clear()

    sub.append(li[0][0])
    sub.append(li[0][1])

    chk[0] = 1
            
    while(1):
        sub,li,chk = inter_conn(sub,li,chk,1)
        sub,li,chk = inter_conn(sub,li,chk,-1)
        sub,li,chk = inter_conn(sub,li,chk,2)
        sub,li,chk = inter_conn(sub,li,chk,-2)

        inter.append(sub.copy())
        sub.clear()

        j = 0
        while(j < len(li)):
            if chk[j] == 0:
                sub.append(li[j][0])
                sub.append(li[j][1])
                chk[j] = 1
                break
            j += 1

        if j == len(li):
            break
                    
    return inter.copy()
                
def new_connection(arr):
    if len(arr) == 1:
        return arr[0]
    
    a = -1
    b = -1
    c = -1
    d = 0
    tmpLi = []
    for i in range(len(arr)-1):
        min = float('inf')
        for j in range(len(arr[0])-1):
            for k in range(1,len(arr)):
                for l in range(len(arr[k])-1):
                    dele = calDistance(arr[0][j],arr[0][j+1]) + calDistance(arr[k][l],arr[k][l+1])

                    temp = calDistance(arr[0][j],arr[k][l]) + calDistance(arr[0][j+1],arr[k][l+1])
                    temp -= dele

                    if min > temp:
                        min = temp
                        a = j
                        b = k
                        c = l
                        d = 1

                    temp = calDistance(arr[0][j],arr[k][l+1]) + calDistance(arr[0][j+1],arr[k][l])
                    temp -= dele

                    if min > temp:
                        min = temp
                        a = j
                        b = k
                        c = l
                        d = -1

        for j in range(len(arr[0])):
            tmpLi.append(arr[0][j])
            if j == a:
                if d == 1:
                    for k in range(len(arr[b])-1):
                        tmpLi.append(arr[b][(c+len(arr[b])-1-k)%(len(arr[b])-1)])
                elif d == -1:
                    for k in range(len(arr[b])-1):
                        tmpLi.append(arr[b][(c+k)%(len(arr[b])-1)])

        del arr[b]
        del arr[0]
        arr.insert(0,tmpLi.copy())
        tmpLi.clear()

    return arr[0]
    

def show_cycle():  
    #print("Esets",esets)
    #print()
    drawSub(esets)

def show_inter():
    drawInter(inter_edges,esets)
    '''
    print("Eset" + str(v2), Eset[v2])
    print("inter" + str(v2), intermediate[v2])
    print()
    '''
    
def show_newGene():
    drawPath(next_generation)

    cO1['text'] = next_cost[0]
    cO2['text'] = next_cost[1]
    cO3['text'] = next_cost[2]
        

def all_cost():
    print("old A", bestOrderA)
    print("old B", bestOrderB)
    print("pre costA ",pre_cost[0],"\npre cost B ",pre_cost[1])
    for i in range(len(next_generation)):
        print("new"+str(i), next_generation[i])
    print("new ", next_cost)
    print(cities)


# UI


def title():
    canvas.create_text(110,220, fill = "black", text="Parent A")
    canvas.create_text(110,450, fill = "black", text="Parent B")
    canvas.create_text(110,690, fill = "black", text="Generate AB")

    canvas.create_text(710,20, fill = "red", text="(1)")
    canvas.create_text(710,230, fill = "red", text="(2)")
    canvas.create_text(710,470, fill = "red", text="(3)")

    canvas.create_text(340,690, fill = "black", text="E-set")
    canvas.create_text(580,690, fill = "black", text="intermediate")
    canvas.create_text(830,690, fill = "black", text="offspring")

def createCities(n):
    global bestCostA,bestCostB
    r = 3
    title()
    for i in range(n):
        city = []
        randX = randrange(20,200)
        randY = randrange(20,200)
        cities.append([randX,randY])
        order.append(i)
        
    order.append(0)

def keepCities():
    global bestCostA,bestCostB,totalCities
    r = 3
    title()
    for i in range(totalCities):
        X = cities[i][0]
        Y = cities[i][1]
        II = i
        s = StringVar(root,II)
        
        canvas.create_oval(X - r, Y - r, X + r, Y + r)
        canvas.create_oval(X - r, Y - r + 230, X + r, Y + r + 230)
        canvas.create_oval(X - r, Y - r + 470, X + r, Y + r + 470)

        canvas.create_oval(X - r + 230, Y - r, X + r + 230, Y + r)
        canvas.create_oval(X - r + 230, Y - r + 230, X + r + 230, Y + r + 230)
        canvas.create_oval(X - r + 230, Y - r + 470, X + r + 230, Y + r + 470)

        canvas.create_oval(X - r + 470, Y - r, X + r + 470, Y + r)
        canvas.create_oval(X - r + 470, Y - r + 230, X + r + 470, Y + r + 230)
        canvas.create_oval(X - r + 470, Y - r + 470, X + r + 470, Y + r + 470)

        canvas.create_oval(X - r + 720, Y - r, X + r + 720, Y + r)
        canvas.create_oval(X - r + 720, Y - r + 230, X + r + 720, Y + r + 230)
        canvas.create_oval(X - r + 720, Y - r + 470, X + r + 720, Y + r + 470)

def mergePath(A,B):
    for i in range(len(A)-1):
        edge = canvas.create_line(cities[A[i]][0],cities[A[i]][1] + 470,cities[A[i+1]][0],cities[A[i+1]][1] + 470, fill = "red")
        edges_M.append(edge)
        edge = canvas.create_line(cities[B[i]][0],cities[B[i]][1] + 470,cities[B[i+1]][0],cities[B[i+1]][1] + 470, fill = "green")
        edges_M.append(edge)

def drawInter(arr,e):
    intermd = []
    for i in range(len(e)):
        temp = arr.copy()
        for j in range(len(e[i])):
            for k in range(0,len(e[i][j])-1,2):
                for l in range(len(temp)):
                    if e[i][j][k] == temp[l][0]:
                        if e[i][j][k+1] == temp[l][1]:
                            temp.remove([temp[l][0],temp[l][1]])
                            break
                    if e[i][j][k+1] == temp[l][0]:
                        if e[i][j][k] == temp[l][1]:
                            temp.remove([temp[l][0],temp[l][1]])
                            break
                        
            for k in range(1,len(e[i][j]),2):
                edgeB = [e[i][j][k],e[i][j][k+1]]
                temp.append(edgeB.copy())
                edgeB.clear()

        intermd.append(temp.copy())
        temp.clear()

    for i in range(len(intermd)):
        if i == 0:
            z = 0
        elif i == 1:
            z = 230
        elif i == 2:
            z = 470
        else:
            break
        for j in range(len(intermd[i])):
            edge = canvas.create_line(cities[intermd[i][j][0]][0] + 470,cities[intermd[i][j][0]][1] + z,cities[intermd[i][j][1]][0] + 470,cities[intermd[i][j][1]][1] + z)
            edges_origin.append(edge)

def drawPath(arr):
    for j in range(len(arr)):
        if j == 0:
            z = 0
        elif j == 1:
            z = 230
        elif j == 2:
            z = 470
        else:
            break
        for i in range(len(arr[j])-1):
            edge = canvas.create_line(cities[arr[j][i]][0] + 720,cities[arr[j][i]][1] + z,cities[arr[j][i+1]][0] + 720,cities[arr[j][i+1]][1] + z)
            edges_origin.append(edge)

def drawSub(arr):
    for j in range(len(arr)):
        toggle = 1
        if j == 0:
            z = 0
        elif j == 1:
            z = 230
        elif j == 2:
            z = 470
        else:
            break
        for k in range(len(arr[j])):
            for i in range(len(arr[j][k])-1):
                if(toggle == 1):
                    edge = canvas.create_line(cities[arr[j][k][i]][0] + 230,cities[arr[j][k][i]][1] + z,cities[arr[j][k][i+1]][0] + 230,cities[arr[j][k][i+1]][1] + z, fill = "red")
                if(toggle == -1):
                    edge = canvas.create_line(cities[arr[j][k][i]][0] + 230,cities[arr[j][k][i]][1] + z,cities[arr[j][k][i+1]][0] + 230,cities[arr[j][k][i+1]][1] + z, fill = "green")
                toggle *= -1
                edge_sub.append(edge)

def drawTwoPath(arr1,arr2):
    for i in range(len(arr1)-1):
        edgeA = canvas.create_line(cities[arr1[i]][0],cities[arr1[i]][1],cities[arr1[i+1]][0],cities[arr1[i+1]][1], fill = "red")
        edgeB = canvas.create_line(cities[arr2[i]][0],cities[arr2[i]][1] + 230,cities[arr2[i+1]][0],cities[arr2[i+1]][1] + 230, fill = "green")
        edges_A.append(edgeA)
        edges_B.append(edgeB)
        

root = Tk()
root.geometry('1200x730')
canvas = Canvas(root, width = 950, height = 700, bg = "white")
Ncities = Entry(root,width=10)
resetButton = Button(root,width=5,text="Start",command=lambda: reset(0))
runButton = Button(root,width=6,text="Others",command=lambda: reset(1))
input_cities = Label(root, text="# of cities:")
nextSub = Button(root,width=5,text="E-set",command=show_cycle)
newInterButton = Button(root,width=10,text='Intermediate',command=show_inter)
newGeneButton = Button(root,width=8,text='Offspring',command=show_newGene)

Pa = Label(root, text="Cost(Parent A):")
Pb = Label(root, text="Cost(Parent B):")

cPa = Label(root, text="")
cPb = Label(root, text="")

O1 = Label(root, text="Cost(Offspring 1):")
O2 = Label(root, text="Cost(Offspring 2):")
O3 = Label(root, text="Cost(Offspring 3):")

cO1 = Label(root, text="")
cO2 = Label(root, text="")
cO3 = Label(root, text="")

canvas.grid(row = 1, rowspan=10, column=0, columnspan = 15)

input_cities.grid(row=0,column=0)
Ncities.grid(row=0,column=1)
resetButton.grid(row=0, column=2)
runButton.grid(row=0, column=3)
nextSub.grid(row=0, column=4)
newInterButton.grid(row=0,column=5)
newGeneButton.grid(row=0,column=6)

Pa.grid(row=1,column=16)
Pb.grid(row=2,column=16)

cPa.grid(row=1,column=17)
cPb.grid(row=2,column=17)

O1.grid(row=4,column=16)
O2.grid(row=5,column=16)
O3.grid(row=6,column=16)

cO1.grid(row=4,column=17)
cO2.grid(row=5,column=17)
cO3.grid(row=6,column=17)

root.mainloop()
