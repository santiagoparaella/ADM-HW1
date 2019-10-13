# py-introduction

# 1 - Say "Hello, World!" With Python

print("Hello, World!")

#2 - Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2!=0:
        print('Weird')
    else:
        if n>= 2 and n<=5:
            print('Not Weird')
        elif n>=6 and n<=20:
            print('Weird')
        elif n>20:
            print('Not Weird')


# 3 - Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# 4 - Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# 5 - Loops

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

# 6 - Write a function

def is_leap(year):
    leap = False

    # Write your logic here
    if year%4==0:
        leap=True
        if year%100==0:
            leap=False
            if year%400==0:
                leap=True

    return leap

year = int(input())
print(is_leap(year))

# 7 - Print Function

if __name__ == '__main__':
    n = int(input())
    print(''.join(map(str, range(1, n+1))))


# py-basic-data-types

# 1 - Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))

# 2 - Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for _ in range(N):
        comando=input()
        cS=comando.split()

        if cS[0]=='insert':
            l.insert(int(cS[1]), int(cS[2]))
        elif cS[0]=='print':
            print(l)
        elif cS[0]=='remove':
            l.remove(int(cS[1]))
        elif cS[0]=='append':
            l.append(int(cS[1]))
        elif cS[0]=='sort':
            l.sort()
        elif cS[0]=='pop':
            l.pop()
        else:
            l.reverse()


# 3 - Finding the percentage
def mySum(l):
    tot=0
    for i in l:
        try:
            tot+=i
        except TypeError:
            print('I can only sum numerical type :( sorry...')
    return tot


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    #print(student_marks)
    query_name = input()
    voti=list(student_marks[query_name])
    avg=mySum(voti)/len(voti)
    print('{:.2f}'.format(avg))

# 4 - Nested Lists
if __name__ == '__main__':
    d=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        d.append([name, score])
    l=[]
    for i in d:
        l.append(i[1])
    lset=set(l)
    l=list(lset)
    l.sort()
    second=l[1]
    f=[]
    for i in d:
        if i[1]==second:
            f.append(i[0])
    f.sort()
    #f.reverse()
    end=False #per compatibilità con l'output richiesto
    for i in range(len(f)):
        print(f[i], end='')
        if i!=len(f)-1:
            print()

# 5 - Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    s=set(arr)
    l=list(s)
    l.sort()
    #print(l)
    print(l[len(l)-2])

# 6 - List Comprehensions
if __name__ == '__main__':
    X = int(input())
    Y = int(input())
    Z = int(input())
    N = int(input())
    r = [[x, y, z] for x in range(X+1) for y in range(Y+1) for z in range(Z+1) if ((x+y+z)!=N)]
    print(r)



# py-strings

# 1 - String Split and Join

def split_and_join(line):
    # write your code here
    return('-'.join(line.split()))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# 2 - sWAP cASE

def swap_case(s):
    sl=list(s)
    for i in range(len(sl)):
        if sl[i].isalpha():
            if sl[i].islower():
                sl[i]= sl[i].upper()
            else:
                sl[i]= sl[i].lower()

    return ''.join(sl)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# 3 - What's Your Name?

def print_full_name(a, b):
    print("Hello {} {}! You just delved into python.".format(a, b))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# 4 - Mutations

def mutate_string(string, position, character):
    sl=list(string)
    sl[position]=character
    return ''.join(sl)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# 5 - Find a string

def count_substring(string, sub_string):
    n=0
    while string.find(sub_string)!=-1:
        n+=1
        pos=string.find(sub_string)
        sl=list(string)
        sl.pop(pos)
        string=''.join(sl)
    return n

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

# 6 - String Validators

if __name__ == '__main__':
    s = input()
    alfa=False
    digit=False
    alphadigit=False
    up=False
    low=False
    for c in s:
        if c.isalpha():
            alfa=True
        if c.isdigit():
            digit=True
        if c.isalnum():
            alphadigit=True
        if c.isupper():
            up=True
        if c.islower():
            low=True

    print(alphadigit)
    print(alfa)
    print(digit)
    print(low)
    print(up)

    #ero stupido io.. controllavo prima se fosse un carattere maiuscolo e
    #poi se era minuscolo. L'esercizio chiedeva il contrario


# 7 - Text Alignment

#Replace all ______ with rjust, ljust or center.

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# 8 - Text Wrap

import textwrap

def wrap(string, max_width):
    sl=list(string)
    sr=[]
    while len(sl)-max_width>0:
        k=[]
        for _ in range(max_width):
            k.append(sl.pop(0))
        sr.append(''.join(k))
        sr.append('\n')
    sr.append(''.join(sl[0:]))
    return ''.join(sr)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# 9 - Capitalize!

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(s):
    sl=list(s)
    for i in range(len(sl)):
        if i==0:
            sl[i]=sl[i].capitalize()
        else:
            if sl[i]==' ' and i<len(sl)-1:
                sl[i+1]=s[i+1].capitalize()
    return ''.join(sl)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


# 10 - Merge the Tools!

def merge_the_tools(string, k):
    # your code goes here
    listString=list(string)
    stringDivise=[]
    j=0
    s=[]
    for i in range(len(listString)):
        if j>=k:
            stringDivise.append(s)
            s=[]
            j=1
            s.append(listString[i])
        else:
            s.append(listString[i])
            j+=1
    stringDivise.append(s)

    p=[]
    for i in range(len(stringDivise)):
        for c in stringDivise[i]:
            if c not in p:
                p.append(c)
        stringDivise[i]=p
        p=[]


    for s in stringDivise:
        print(''.join(s))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# Sets

#1 - Introduction to Sets

def average(array):
    # your code goes here
    return '{:.3f}'.format(sum(set(array))/len(set(array)))



if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# 2 - Symmetric Difference

# Enter your code here. Read input from STDIN. Print output to STDOUT

m= int(input())
mset = set(map(int, input().split()))

n= int(input())
nset = set(map(int, input().split()))

sd=list(mset.symmetric_difference(nset))
sd.sort()

for v in sd:
    print(v)



#3 - Set .add()

# Enter your code here. Read input from STDIN. Print output to STDOUT
s=set()
for _ in range(int(input())):
    c=input()
    s.add(c)

print(len(s))

# 4 - Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
nc = int(input())
comandi=list()
for _ in range(nc):
    comandi.append(input().split())
#print(comandi)

for i in range(len(comandi)):
    #print(i)
    if comandi[i][0]=='remove':
        #print('in remove')
        if int(comandi[i][1]) in s:
            s.remove(int(comandi[i][1]))
    elif comandi[i][0]=='discard':
        #print('in discard')
        s.discard(int(comandi[i][1]))
    elif comandi[i][0]=='pop':
        #print('in pop')
        if len(s)!=0:
            s.pop()

print(sum(s))

# 5 - Set .union() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
sn = set(map(int, input().split()))
m=int(input())
sm = set(map(int, input().split()))

print(len(sn.union(sm)))

# 6 - Set .intersection() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
sn = set(map(int, input().split()))
m=int(input())
sm = set(map(int, input().split()))

print(len(sn.intersection(sm)))


# 7 - Set .difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
sn = set(map(int, input().split()))
m=int(input())
sm = set(map(int, input().split()))

print(len(sn.difference(sm)))

# 8 - Set .symmetric_difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
sn = set(map(int, input().split()))
m=int(input())
sm = set(map(int, input().split()))

print(len(sn.symmetric_difference(sm)))

# 9 - Set Mutations

# Enter your code here. Read input from STDIN. Print output to STDOUT
na=int(input())
a=set(map(int, input().split()))
nothers = int(input())
for _ in range(nothers):
    op_num = input().split()
    b=set(map(int, input().split()))

    if op_num[0]=='update':
        a.update(b)

    if op_num[0]=='intersection_update':
        a.intersection_update(b)

    if op_num[0]=='difference_update':
        a.difference_update(b)

    if op_num[0]=='symmetric_difference_update':
        a.symmetric_difference_update(b)

print(sum(a))

# 10 - The Captain's Room

# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())
l=list(map(int, input().split()))
s=set(l)
sums=(sum(s)*k)
suml=sum(l)
dif=sums-suml
print(dif//(k-1))

# 11 - Check Subset

# Enter your code here. Read input from STDIN. Print output to STDOUT

ntests=int(input())
for _ in range(ntests):
    na=int(input())
    a=set(map(int, input().split()))
    nb=int(input())
    b=set(map(int, input().split()))
    subset=True
    for e in a:
        if e not in b:
            subset=False
    print(subset)

# 12 - Check Strict Superset

# Enter your code here. Read input from STDIN. Print output to STDOUT
a=set(map(int, input().split()))
ofAll=True
for _ in range(int(input())):
    b=set(map(int, input().split()))
    ofAll = ofAll and a.issuperset(b)
print(ofAll)

# 13 - Min and Max

import numpy as np

n, m = map(int, input().split())
l=[]
for _ in range(n):
    l.append(list(map(int, input().split())))
array=np.array(l)

ax1=np.min(array, axis=1)
print(max(ax1))

# 14 - Mean, Var, and Std
import numpy as np
n, m = map(int, input().split())
l=[]
for _ in range(n):
    l.append(list(map(int, input().split())))
array=np.array(l)

mean=np.mean(array, axis=1)
var=np.var(array, axis=0)
std=np.std(array)
print(mean)
print(var)
print(std, end='')

# 15 - Dot and Cross

import numpy as np

n=int(input())

a=[]
b=[]
for _ in range(n):
    a.append(list(map(int, input().split())))
a=np.array(a)

for _ in range(n):
    b.append(list(map(int, input().split())))
b=np.array(b)


m=np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        m[i][j]=np.dot(a[i], b[:,[j]])

print(m)

# 16 - Inner and Outer

import numpy as np

a=np.array(list(map(int, input().split())))
b=np.array(list(map(int, input().split())))
print(np.inner(a, b))
print(np.outer(a, b))

# 17 - Polynomials

import numpy as np

coefficents=list(map(float, input().split()))
point=int(input())

print(np.polyval(coefficents, point))

# 18 - Linear Algebra

import numpy as np
n=int(input())

a=[]
for _ in range(n):
    a.append(list(map(float, input().split())))
m=np.array(a)
det=np.linalg.det(m)

#print('{:.2f}'.format(det))
print(round(det, 2))

# 19 - itertools.product()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import product

a=list(map(int, input().split()))
b=list(map(int, input().split()))
p=list(product(a, b))
p.sort()
for e in p:
    print(e, end=' ')

# 20 - No Idea!

n, m = map(int, input().split())
ar=list(map(int, input().split()))
a=set(map(int, input().split()))
b=set(map(int, input().split()))

h=0

for e in ar:
    if e in a:
        h+=1
    if e in b:
        h-=1

print(h)

# - py-collections

# 1 - DefaultDict Tutorial
n, m = map(int, input().split())

a=[]
b=[]
for _ in range(n):
    a.append(input())
for _ in range(m):
    b.append(input())

for e in b:
    pos=[]
    for i in range(n):
        if e==a[i]:
            pos.append(i+1)
    print( ' '.join(map(str, pos)) if len(pos)>0 else '-1' )
    pos=[]


# 2 - Collections.namedtuple()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
n = int(input())
colonne=list(input().split())
Student = namedtuple('Student', ' '.join(colonne))
students=[]
for _ in range(n):
    students.append(Student._make(input().split()))
sum = 0
for e in students:
    sum+=int(e.MARKS)
print('{:.2f}'.format(sum/n))

# 3 - Collections.OrderedDict()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict

n = int(input())

dic=OrderedDict()

for _ in range(n):
    nomeQuant = input().split()
    nome=' '.join(nomeQuant[:-1])
    quant=nomeQuant[-1]

    dic[nome]=dic[nome]+int(quant) if nome in dic.keys() else int(quant)

for e in dic:
    print(e, dic[e])


# py_date_time

# 1 - Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar as c
month, day, year = map(int, input().split())
print(c.day_name[c.weekday(year, month, day)].upper())

# py_exceptions

# 1 - Exceptions

# Enter your code here. Read input from STDIN. Print output to STDOUT

n=int(input())
for _ in range(n):
    a, b = input().split()
    if b=='0':
        try:

            print(int(a)//int(b))
        except ZeroDivisionError as e1:
            print("Error Code:", e1)

    else:
        try:
            print(int(a)//int(b))
        except ValueError as e2:
            print("Error Code:", e2)


# Built-ins

# 1 - Zipped!

# Enter your code here. Read input from STDIN. Print output to STDOUT

n, x = map(int, input().split())
l=[]
for _ in range(x):
    l=l+[list(map(float, input().split()))]

other=zip(*l)
for t in other:
    print(sum(t)/x)


# 2 -ginortS

# Enter your code here. Read input from STDIN. Print output to STDOUT

s=input()

lo=[]
up=[]
odd=[]
even=[]

for c in s:
    if c.islower():
        lo.append(c)
    elif c.isupper():
        up.append(c)
    elif c.isdigit():
        if int(c)%2==0:
            even.append(c)
        else:
            odd.append(c)

up.sort()
lo.sort()
odd.sort()
even.sort()

print(''.join(lo+up+odd+even))

# map-and-lambda-expression

# 1 - Map and Lambda Function
cube = lambda x: x**3# complete the lambda function

def fibonacci(n):
    if n==0:
        return []
    elif n==1:
        return [0]
    else:
        l=[-1 for i in range(n)]
        l[0]=0
        l[1]=1
        for i in range(2, n):
            l[i]=l[i-1]+l[i-2]
    return(l)

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# - xml

# 1 - XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    a=0
    a+=len(node.attrib)
    #print(len(node.attrib), node.text , node.attrib)
    for child in node:
        a+=len(child.attrib)
        #print(len(child.attrib), child.text, child.attrib)

        for ch in child:
            a+=len(ch.attrib)
    return a
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# 2 - XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    maxdepth=level+1 if maxdepth<=level+1 else maxdepth
    if len(elem.getchildren())>0:
        for c in elem.getchildren():
            depth(c, level+1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

# - closures-and-decorator

# 1 - Closures and Decorators

def wrapper(f):
    def fun(l):
    # complete the function
        n=len(l)
        myl=['' for i in range(n)]
        for i in range(n):
            myl[i]=l[i][-10:]
        myl.sort()

        for i in range(n):
                print(''.join(['+','9','1', ' ']+list(myl[i][:5])+[' ']+list(myl[i][5:])))
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# - numpy

# 1 - Arrays
import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    a=[float(x) for x in arr]
    a.reverse()
    return(numpy.array(a))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# 2 -









#PARTE 2

# 1 - Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    m=max(ar)
    talles=[x for x in ar if x==m]
    return len(talles)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()

# out - Time Conversion


#!/bin/python3

import os
import sys

#
# Complete the timeConversion function below.
#
def timeConversion(s):
    #
    # Write your code here.
    #

    if s[8:] == "AM" and s[:2] == "12":
        return "00" + s[2:8]

    elif s[8:] == "AM":
        return s[:8]

    elif s[8:] == "PM" and s[:2] == "12":
        return s[:8]

    else:
        return str(int(s[:2]) + 12) + s[2:8]

timeConversion('12:45:54PM')


if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = timeConversion(s)

    f.write(result + '\n')

    f.close()


# 2 - Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    v=arr[len(arr)-1]
    i=len(arr)-1
    while arr[i-1]>v and i>0:
        arr[i]=arr[i-1]
        print(' '.join(map(str, arr)))
        i-=1
    arr[i]=v
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# 3 - Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys


def myinsertionSort(n, arr):
    v=arr[len(arr)-1]
    i=len(arr)-1
    while arr[i-1]>v and i>0:
        arr[i]=arr[i-1]
        i-=1
    arr[i]=v
    return(arr)

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    subArr= []
    #print(' '.join(map(str, arr)))
    for i in range(1, len(arr)):
        arr=myinsertionSort(i, arr[:i+1])+arr[i+1:]
        print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

# 4 - Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    digitOfn=0
    for c in n:
        digitOfn+=int(c)
    digitOfnk=str(digitOfn*k)

    if len(digitOfnk)==1:
        return(digitOfnk)
    else:
        return(superDigit(digitOfnk, 1))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# 5 - Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def myViralAdvertising(day, prec, cum):
    if day==0:
        return(cum)
    else:
        prec=int(prec*3/2)
        cum+=prec
        return(myViralAdvertising(day-1, prec, cum))

def viralAdvertising(n):
    prec=2
    cum=2
    return(myViralAdvertising(n-1, prec, cum))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# 6 - Kangaroo

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if x1>x2 and v1>v2:
        return('NO')
    elif x2>x1 and v2>v1:
        return('NO')
    elif x1==x2 and v1==v2:
            return('YES')
    elif x1!=x2 and v1==v2:
        return('NO')
    else:
        piùAvanti=x1 if x1>x2 else x2
        velPiùAvanti=v1 if piùAvanti==x1 else v2
        piùIndietro=x1 if piùAvanti==x2 else x2
        velPiùIndietro=v1 if piùIndietro==x1 else v2
        print(piùAvanti, velPiùAvanti, piùIndietro, velPiùIndietro)

        while piùAvanti>=piùIndietro:
            piùAvanti+=velPiùAvanti
            piùIndietro+=velPiùIndietro
            if piùAvanti==piùIndietro:
                return('YES')
        return('NO')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
