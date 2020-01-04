a = ['ciao', 'ciao', 'cia']
if all('cia' in elem for elem in a):
    print("cia in tutti")
else:
    print("non sai usare all()")

if all('ciao' in elem for elem in a):
    print("non sai usare all()")
else:
    print("ciao non è in tutti")

if any('ciao' in elem for elem in a):
    print("ciao in alcuni")
else:
    print("non sai usare all()")
